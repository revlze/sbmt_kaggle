"""Fast leakage-safe distance-cache framework for ``svc_multikernel.ipynb``.

Target-derived train features are inner-OOF.  Outer-valid rows, scalers, PCA,
gamma estimates and pairwise distances never use outer-valid labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from svc_experiments import CoordinateLikelihoodFeatures, GeometryFeatures, PPCA_COMPONENTS


BLOCKS = ("raw", "geometry", "aggregates", "density", "knn", "coord", "pca128")
COMPOSITE_COMPONENTS = ("current_best", "fine6", *BLOCKS)
OLD_AUX_PARTS = ("geometry", "aggregates", "old_density", "knn", "coord")
AGG = ("total", "count_zeros", "relative_std", "max_share", "max", "log1p_sum")
CACHE_VERSION = 2
FEATURE_BUILDER_CONFIG = "enhanced_ppca_coord_inner_oof_v2"
INNER_SEED_OFFSET = 10_000


@dataclass
class DistanceFold:
    repeat: int
    fold: int
    seed: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    d2_train: dict[str, Path | np.ndarray]
    d2_valid: dict[str, Path | np.ndarray]
    features_train: dict[str, Path | np.ndarray]
    features_valid: dict[str, Path | np.ndarray]
    gamma0: dict[str, float]
    cache_key: str


@dataclass
class DistanceCache:
    folds: list[DistanceFold]
    seeds: tuple[int, ...]
    n_samples: int
    cache_dir: Path
    cache_hits: int
    cache_builds: int


def _geometry_families(frame):
    names = list(frame.columns)
    geometry = [c for c in names if c in {"d_euclid_diff", "proj_w", "cos_w", "lda_score"}
                or c.startswith(("mahal_", "d_mahal_", "rda_", "mid_", "fisher_"))]
    density = [c for c in names if c in {"qda_score", "pca_subspace_diff"} or c.startswith("ppca_")]
    old_density = [c for c in names if c in {"qda_score", "pca_subspace_diff"}]
    knn = [c for c in names if c.startswith("knn_")]
    blocks = {
        "geometry": frame[geometry].to_numpy(),
        "aggregates": frame[list(AGG)].to_numpy(),
        "density": frame[density].to_numpy(),
        "old_density": frame[old_density].to_numpy(),
        "knn": frame[knn].to_numpy(),
    }
    names_by_block = {"geometry": geometry, "aggregates": list(AGG), "density": density,
                      "old_density": old_density, "knn": knn}
    return blocks, names_by_block


def _scale_pair(train, valid):
    scaler = StandardScaler()
    return scaler.fit_transform(train), scaler.transform(valid), scaler


def make_feature_blocks(X_train, y_train, X_valid, builder_seed, n_bins=8, alpha=5.0,
                        pca_components=128):
    """Create all views once; supervised train views are produced by inner OOF."""
    X_train = np.asarray(X_train, dtype=float)
    X_valid = np.asarray(X_valid, dtype=float)
    y_train = np.asarray(y_train)
    z_train, z_valid = np.log1p(X_train), np.log1p(X_valid)

    geometry = GeometryFeatures("enhanced", PPCA_COMPONENTS, (), 5, builder_seed)
    geometry_oof = geometry.fit_transform(z_train, y_train)
    geometry_valid = geometry.transform(z_valid)
    train, names = _geometry_families(geometry_oof)
    valid, _ = _geometry_families(geometry_valid)

    coord = CoordinateLikelihoodFeatures(n_bins, alpha, True, True, 5, builder_seed)
    train["coord"] = coord.fit_transform(z_train, y_train).to_numpy()
    valid["coord"] = coord.transform(z_valid).to_numpy()
    train["raw"], valid["raw"] = z_train, z_valid

    raw_for_pca = StandardScaler()
    A, B = raw_for_pca.fit_transform(z_train), raw_for_pca.transform(z_valid)
    n_components = min(int(pca_components), len(A) - 1, A.shape[1])
    pca = PCA(n_components=n_components, random_state=builder_seed).fit(A)
    train["pca128"], valid["pca128"] = pca.transform(A), pca.transform(B)

    scaled_train, scaled_valid, scalers = {}, {}, {}
    for block in (*BLOCKS, "old_density"):
        scaled_train[block], scaled_valid[block], scalers[block] = _scale_pair(train[block], valid[block])

    old_train = np.column_stack([train[b] for b in OLD_AUX_PARTS])
    old_valid = np.column_stack([valid[b] for b in OLD_AUX_PARTS])
    scaled_train["old_aux"], scaled_valid["old_aux"], scalers["old_aux"] = _scale_pair(old_train, old_valid)
    names["coord"] = [f"coord_{i}" for i in range(train["coord"].shape[1])]
    names["raw"] = [f"raw_{i}" for i in range(X_train.shape[1])]
    names["pca128"] = [f"pc_{i}" for i in range(n_components)]
    names["old_aux"] = [c for b in OLD_AUX_PARTS for c in names.get(b, [])]

    artifacts = {
        "scalers": scalers,
        "pca": pca,
        "pca_input_scaler": raw_for_pca,
        "feature_names": names,
    }
    if scaled_train["old_aux"].shape[1] != 52:
        raise RuntimeError(f"old_aux must contain 52 features, got {scaled_train['old_aux'].shape[1]}")
    return scaled_train, scaled_valid, artifacts


def build_feature_audit_blocks(X_train, y_train, builder_seed, n_bins=8,
                               alpha=5.0, pca_components=128):
    """Build standardized train-only views for an unsupervised redundancy audit.

    This deliberately reuses :func:`make_feature_blocks`, so supervised train
    representations are generated by the same inner-OOF ``fit_transform`` path
    as the modeling cache.  The one-row transform probe comes from ``X_train``
    itself and is discarded; validation and test data are never accepted here.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    if X_train.ndim != 2 or len(X_train) != len(y_train) or len(X_train) < 2:
        raise ValueError("X_train/y_train shape mismatch or insufficient audit rows")
    scaled_train, _, artifacts = make_feature_blocks(
        X_train, y_train, X_train[:1], builder_seed,
        n_bins=n_bins, alpha=alpha, pca_components=pca_components)
    audit_names = ("geometry", "density", "aggregates", "knn",
                   "coord", "pca128", "old_aux")
    audit_blocks = {
        name: np.asarray(scaled_train[name], dtype=float)
        for name in audit_names
    }
    feature_names = {
        name: list(artifacts["feature_names"][name])
        for name in audit_names
    }
    for name in audit_names:
        block = audit_blocks[name]
        if block.ndim != 2 or block.shape[0] != len(X_train):
            raise RuntimeError(f"invalid audit block shape for {name}: {block.shape}")
        if len(feature_names[name]) != block.shape[1]:
            raise RuntimeError(
                f"feature-name mismatch for {name}: "
                f"{len(feature_names[name])} names vs {block.shape[1]} columns")
        if not np.isfinite(block).all():
            raise RuntimeError(f"non-finite values in audit block {name}")
    return audit_blocks, feature_names


def _median_gamma_from_distance(D, seed, n_pairs=100_000):
    rng, n = np.random.default_rng(seed), len(D)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = i != j
    values = D[i[keep], j[keep]]
    values = values[values > 0]
    median = float(np.median(values)) if len(values) else 1.0
    return float(np.log(2.0) / median), median


def _fold_key(seed, fold, train_idx, valid_idx, n_bins, alpha, pca_components):
    split_hash = sha256(np.asarray(train_idx, dtype=np.int32).tobytes()
                        + np.asarray(valid_idx, dtype=np.int32).tobytes()).hexdigest()[:16]
    spec = {"version": CACHE_VERSION, "seed": int(seed), "fold": int(fold),
            "split": split_hash, "builder": FEATURE_BUILDER_CONFIG,
            "inner_seed_offset": INNER_SEED_OFFSET, "n_bins": int(n_bins),
            "alpha": float(alpha), "pca_components": int(pca_components),
            "ppca_components": list(PPCA_COMPONENTS)}
    digest = sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    return digest, spec


def _paths(folder, kind, names):
    return {name: folder / f"{kind}_{name}.npy" for name in names}


def _load_array(value):
    return np.load(value, mmap_mode="r", allow_pickle=False) if isinstance(value, (str, Path)) else value


def _save_preprocessing(folder, artifacts):
    values = {}
    for name, scaler in artifacts["scalers"].items():
        values[f"scaler_{name}_mean"] = scaler.mean_
        values[f"scaler_{name}_scale"] = scaler.scale_
    pca, raw_scaler = artifacts["pca"], artifacts["pca_input_scaler"]
    values.update({"pca_components": pca.components_, "pca_mean": pca.mean_,
                   "pca_explained_variance": pca.explained_variance_,
                   "pca_input_mean": raw_scaler.mean_, "pca_input_scale": raw_scaler.scale_})
    np.savez(folder / "preprocessing.npz", **values)
    (folder / "feature_names.json").write_text(json.dumps(artifacts["feature_names"]), encoding="utf-8")


def _manifest_complete(folder, spec):
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("spec") == spec and all((folder / name).exists() for name in manifest.get("files", []))


def _fold_from_disk(folder, repeat, fold, seed, train_idx, valid_idx, y, key):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    distance_names = (*BLOCKS, "old_density", "old_aux")
    feature_names = (*BLOCKS, "old_aux")
    return DistanceFold(repeat, fold, seed, train_idx, valid_idx, y[train_idx], y[valid_idx],
                        _paths(folder, "d2_train", distance_names),
                        _paths(folder, "d2_valid", distance_names),
                        _paths(folder, "feature_train", feature_names),
                        _paths(folder, "feature_valid", feature_names),
                        {k: float(v) for k, v in manifest["gamma0"].items()}, key)


def _build_fold(X, y, repeat, seed, fold, train_idx, valid_idx, folder, key, spec,
                n_bins, alpha, pca_components):
    print(f"build cache: repeat={repeat + 1}, fold={fold + 1}")
    builder_seed = seed + INNER_SEED_OFFSET + fold
    train, valid, artifacts = make_feature_blocks(
        X[train_idx], y[train_idx], X[valid_idx], builder_seed, n_bins, alpha, pca_components)
    folder.mkdir(parents=True, exist_ok=True)
    distance_names = (*BLOCKS, "old_density", "old_aux")
    gamma0, medians = {}, {}
    for block in distance_names:
        Dtr = cdist(train[block], train[block], "sqeuclidean").astype(np.float32)
        Dva = cdist(valid[block], train[block], "sqeuclidean").astype(np.float32)
        gamma0[block], medians[block] = _median_gamma_from_distance(Dtr, seed + fold + len(gamma0))
        np.save(folder / f"d2_train_{block}.npy", Dtr, allow_pickle=False)
        np.save(folder / f"d2_valid_{block}.npy", Dva, allow_pickle=False)
        del Dtr, Dva
    for block in (*BLOCKS, "old_aux"):
        np.save(folder / f"feature_train_{block}.npy", np.asarray(train[block], np.float32), allow_pickle=False)
        np.save(folder / f"feature_valid_{block}.npy", np.asarray(valid[block], np.float32), allow_pickle=False)
    np.save(folder / "train_idx.npy", np.asarray(train_idx, np.int32), allow_pickle=False)
    np.save(folder / "valid_idx.npy", np.asarray(valid_idx, np.int32), allow_pickle=False)
    np.save(folder / "y_train.npy", y[train_idx], allow_pickle=False)
    np.save(folder / "y_valid.npy", y[valid_idx], allow_pickle=False)
    _save_preprocessing(folder, artifacts)
    files = [p.name for p in folder.iterdir() if p.name != "manifest.json"]
    manifest = {"spec": spec, "gamma0": gamma0, "median_d2": medians,
                "n_features": {b: int(train[b].shape[1]) for b in (*BLOCKS, "old_aux")},
                "files": sorted(files)}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return _fold_from_disk(folder, repeat, fold, seed, train_idx, valid_idx, y, key)


def build_or_load_distance_cache(X, y, seeds, n_splits=5, cache_dir=".svc_multikernel_cache_v2",
                                 rebuild=False, n_bins=8, alpha=5.0, pca_components=128,
                                 n_jobs=1):
    """Build/load feature and distance cache. Gamma, weights and C are not cache keys."""
    X, y = np.asarray(X, float), np.asarray(y)
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs, loaded, hits = [], [], 0
    for repeat, seed in enumerate(seeds):
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            if np.intersect1d(train_idx, valid_idx).size:
                raise RuntimeError("outer train/valid indices overlap")
            key, spec = _fold_key(seed, fold, train_idx, valid_idx, n_bins, alpha, pca_components)
            folder = root / key
            if not rebuild and _manifest_complete(folder, spec):
                print(f"cache hit: repeat={repeat + 1}, fold={fold + 1}")
                loaded.append(_fold_from_disk(folder, repeat, fold, seed, train_idx, valid_idx, y, key))
                hits += 1
            else:
                jobs.append((repeat, seed, fold, train_idx, valid_idx, folder, key, spec))

    built = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_build_fold)(X, y, *job, n_bins, alpha, pca_components) for job in jobs)
    folds = loaded + built
    folds.sort(key=lambda item: (item.repeat, item.fold))
    return DistanceCache(folds, tuple(seeds), len(X), root, hits, len(built))


def cache_summary(cache):
    return pd.DataFrame([{"repeat": f.repeat, "fold": f.fold, "seed": f.seed,
                          "cache_key": f.cache_key, **{f"gamma0_{b}": f.gamma0[b] for b in BLOCKS}}
                         for f in cache.folds])


def normalized_weights(weights):
    unknown = set(weights) - set(BLOCKS)
    if unknown:
        raise KeyError(f"unknown blocks: {sorted(unknown)}")
    values = np.asarray([float(weights.get(b, 0.0)) for b in BLOCKS])
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative and have a positive sum")
    return dict(zip(BLOCKS, values / values.sum()))


def _kernel_from_distances(fold, mode, weights, gammas, train=True):
    source = fold.d2_train if train else fold.d2_valid
    active = [b for b in BLOCKS if weights.get(b, 0.0) > 0]
    if mode == "additive":
        result = None
        for block in active:
            kernel = np.exp(-gammas[block] * _load_array(source[block])).astype(np.float32)
            if result is None:
                result = weights[block] * kernel
            else:
                result += weights[block] * kernel
        return np.asarray(result, np.float32)
    if mode == "product":
        exponent = None
        for block in active:
            term = np.float32(weights[block] * gammas[block]) * _load_array(source[block])
            if exponent is None:
                exponent = np.asarray(term, np.float32).copy()
            else:
                exponent += term
        return np.exp(-exponent).astype(np.float32)
    raise ValueError("mode must be 'additive' or 'product'")


def _evaluate_kernel_grid(cache, kernel_builder, c_grid, n_jobs=-1):
    c_grid = tuple(float(c) for c in c_grid)

    def one(fold):
        Ktr, Kva = kernel_builder(fold, True), kernel_builder(fold, False)
        if not np.isfinite(Ktr).all() or not np.isfinite(Kva).all():
            raise RuntimeError("non-finite kernel matrix")
        predictions = []
        for C in c_grid:
            model = SVC(C=C, kernel="precomputed").fit(Ktr, fold.y_train)
            score = model.decision_function(Kva)
            predictions.append((roc_auc_score(fold.y_valid, score), score))
        return fold.repeat, fold.valid_idx, predictions

    items = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(fold) for fold in cache.folds)
    output = {C: {"aucs": [], "oof": np.full((len(cache.seeds), cache.n_samples), np.nan)} for C in c_grid}
    for repeat, valid_idx, predictions in items:
        for C, (auc, score) in zip(c_grid, predictions):
            output[C]["aucs"].append(auc)
            output[C]["oof"][repeat, valid_idx] = score
    for C in c_grid:
        output[C]["aucs"] = np.asarray(output[C]["aucs"])
        if not np.isfinite(output[C]["oof"]).all():
            raise RuntimeError("incomplete OOF predictions")
    return output


def evaluate_config_grid(cache, config, gammas, c_grid, n_jobs=-1):
    mode = config["mode"]
    weights = normalized_weights(config["weights"])
    for block, weight in weights.items():
        if weight > 0 and block not in gammas:
            raise KeyError(f"missing gamma for active block {block}")
    builder = lambda fold, train: _kernel_from_distances(fold, mode, weights, gammas, train)
    grid = _evaluate_kernel_grid(cache, builder, c_grid, n_jobs)
    best_C = max(grid, key=lambda C: grid[C]["aucs"].mean())
    return {"mode": mode, "weights": weights, "gammas": dict(gammas), "best_C": best_C,
            "aucs": grid[best_C]["aucs"], "oof": grid[best_C]["oof"], "grid": grid}


def score_raw_direct(cache, C, gamma, n_jobs=-1):
    """Independent legacy-style sklearn RBF path used for Raw parity."""
    def one(fold):
        A, B = _load_array(fold.features_train["raw"]), _load_array(fold.features_valid["raw"])
        model = SVC(C=C, kernel="rbf", gamma=gamma).fit(A, fold.y_train)
        score = model.decision_function(B)
        return fold.repeat, fold.valid_idx, roc_auc_score(fold.y_valid, score), score
    items = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(f) for f in cache.folds)
    aucs, oof = [], np.full((len(cache.seeds), cache.n_samples), np.nan)
    for repeat, idx, auc, score in items:
        aucs.append(auc); oof[repeat, idx] = score
    return np.asarray(aucs), oof


def score_raw_precomputed(cache, C, gamma, n_jobs=-1):
    weights = normalized_weights({"raw": 1.0})
    gammas = {"raw": float(gamma)}
    builder = lambda fold, train: _kernel_from_distances(fold, "product", weights, gammas, train)
    result = _evaluate_kernel_grid(cache, builder, [C], n_jobs)[float(C)]
    return result["aucs"], result["oof"]


def _old_aux_reconstructed(fold, train=True):
    source = fold.d2_train if train else fold.d2_valid
    result = np.asarray(_load_array(source[OLD_AUX_PARTS[0]]), np.float32).copy()
    for part in OLD_AUX_PARTS[1:]:
        result += _load_array(source[part])
    return result


def score_current_best(cache, params, reconstructed=False, n_jobs=-1):
    """Score direct 52-D old_aux or its exact sum-of-block-distances reconstruction."""
    def builder(fold, train):
        source = fold.d2_train if train else fold.d2_valid
        Draw = _load_array(source["raw"])
        Daux = _old_aux_reconstructed(fold, train) if reconstructed else _load_array(source["old_aux"])
        return np.exp(-params["gamma_raw"] * Draw - params["gamma_geom"] * Daux).astype(np.float32)
    result = _evaluate_kernel_grid(cache, builder, [params["C"]], n_jobs)[float(params["C"])]
    return result["aucs"], result["oof"]


def select_standalone_gammas(cache, gamma_multipliers, c_grid, raw_gamma,
                             raw_oof, current_oof, n_jobs=-1):
    rows, selected, runs = [], {}, {}
    for block in BLOCKS:
        base = float(np.median([fold.gamma0[block] for fold in cache.folds]))
        candidates = [(1.0, float(raw_gamma))] if block == "raw" else [
            (float(multiplier), base * float(multiplier)) for multiplier in gamma_multipliers]
        best = None
        for multiplier, gamma in candidates:
            run = evaluate_config_grid(cache, {"mode": "product", "weights": {block: 1.0}},
                                       {block: gamma}, c_grid, n_jobs)
            candidate = (run["aucs"].mean(), multiplier, gamma, run)
            if best is None or candidate[0] > best[0]:
                best = candidate
        _, multiplier, gamma, run = best
        selected[block], runs[block] = gamma, run
        rows.append({"block": block, "best_gamma": gamma, "best_gamma_multiplier": multiplier,
                     "best_C": run["best_C"], "CV_mean": run["aucs"].mean(),
                     "CV_std": run["aucs"].std(),
                     "corr_with_raw": spearmanr(run["oof"].ravel(), raw_oof.ravel()).statistic,
                     "corr_with_current": spearmanr(run["oof"].ravel(), current_oof.ravel()).statistic})
    return pd.DataFrame(rows), selected, runs


def run_user_configs(cache, configs, fixed_gammas, c_grid, n_jobs=-1):
    runs = {}
    for name, config in configs.items():
        started = perf_counter()
        run = evaluate_config_grid(cache, config, fixed_gammas, c_grid, n_jobs)
        run["fit_time"] = perf_counter() - started
        runs[name] = run
    return runs


def comparison_row(name, mode, aucs, oof, raw_aucs, raw_oof, current_aucs, current_oof,
                   best_C, gammas=None, weights=None, fit_time=np.nan):
    gammas, weights = gammas or {}, weights or {}
    return {"config": name, "mode": mode, "CV_mean": aucs.mean(), "CV_std": aucs.std(),
            "CV_median": np.median(aucs), "delta_vs_raw": np.mean(aucs - raw_aucs),
            "delta_vs_current": np.mean(aucs - current_aucs),
            "wins_vs_raw": int((aucs > raw_aucs).sum()),
            "wins_vs_current": int((aucs > current_aucs).sum()), "best_C": best_C,
            **{f"gamma_{b}": gammas.get(b, np.nan) for b in BLOCKS},
            **{f"w_{b}": weights.get(b, 0.0) for b in BLOCKS},
            "corr_with_raw": spearmanr(oof.ravel(), raw_oof.ravel()).statistic,
            "corr_with_current": spearmanr(oof.ravel(), current_oof.ravel()).statistic,
            "fit_time": fit_time}


def make_external_fold(X_train, y_train, X_valid, y_valid, seed, n_bins=8, alpha=5.0,
                       pca_components=128):
    train, valid, _ = make_feature_blocks(X_train, y_train, X_valid,
                                          seed + INNER_SEED_OFFSET, n_bins, alpha, pca_components)
    dtr, dva, gamma0 = {}, {}, {}
    for block in (*BLOCKS, "old_density", "old_aux"):
        dtr[block] = cdist(train[block], train[block], "sqeuclidean").astype(np.float32)
        dva[block] = cdist(valid[block], train[block], "sqeuclidean").astype(np.float32)
        gamma0[block], _ = _median_gamma_from_distance(dtr[block], seed + len(gamma0))
    return DistanceFold(0, 0, seed, np.arange(len(X_train)), np.arange(len(X_valid)),
                        np.asarray(y_train), np.asarray(y_valid), dtr, dva, train, valid, gamma0, "external")


def score_external_config(fold, config, gammas, C):
    weights = normalized_weights(config["weights"])
    Ktr = _kernel_from_distances(fold, config["mode"], weights, gammas, True)
    Kva = _kernel_from_distances(fold, config["mode"], weights, gammas, False)
    score = SVC(C=C, kernel="precomputed").fit(Ktr, fold.y_train).decision_function(Kva)
    return roc_auc_score(fold.y_valid, score), score


def normalized_components(components):
    """Validate and normalize virtual/base kernel mixture weights."""
    unknown = set(components) - set(COMPOSITE_COMPONENTS)
    if unknown:
        raise KeyError(f"unknown composite components: {sorted(unknown)}")
    values = np.asarray([float(components.get(name, 0.0)) for name in COMPOSITE_COMPONENTS])
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("component weights must be finite, nonnegative and have a positive sum")
    return dict(zip(COMPOSITE_COMPONENTS, values / values.sum()))


def _current_best_kernel(fold, current_best_params, train=True):
    """Exact parity Current Best kernel; SVC C intentionally is not used here."""
    source = fold.d2_train if train else fold.d2_valid
    return np.exp(
        -float(current_best_params["gamma_raw"]) * _load_array(source["raw"])
        -float(current_best_params["gamma_geom"]) * _load_array(source["old_aux"])
    ).astype(np.float32)


def _fine6_kernel(fold, fixed_gammas, fine6_weights, train=True):
    weights = normalized_weights(fine6_weights)
    return _kernel_from_distances(fold, "additive", weights, fixed_gammas, train)


def combine_kernel_components(fold, components, fixed_gammas, current_best_params,
                              fine6_weights, train=True):
    """Combine kernel matrices before fitting one precomputed-kernel SVC.

    Supported components are current_best, fine6, and every member of BLOCKS.
    This is kernel composition, not decision-function blending.
    """
    weights = normalized_components(components)
    source = fold.d2_train if train else fold.d2_valid
    result = None
    for component, weight in weights.items():
        if weight <= 0:
            continue
        if component == "current_best":
            kernel = _current_best_kernel(fold, current_best_params, train)
        elif component == "fine6":
            kernel = _fine6_kernel(fold, fixed_gammas, fine6_weights, train)
        else:
            if component not in fixed_gammas:
                raise KeyError(f"missing fixed gamma for {component}")
            kernel = np.exp(-float(fixed_gammas[component]) * _load_array(source[component])).astype(np.float32)
        if result is None:
            result = np.float32(weight) * kernel
        else:
            result += np.float32(weight) * kernel
    result = np.asarray(result, dtype=np.float32)
    if train and not np.allclose(np.diag(result), 1.0, atol=2e-5):
        raise RuntimeError("composite kernel diagonal is not one")
    return result


def evaluate_composite_grid(cache, components, fixed_gammas, current_best_params,
                            fine6_weights, c_grid, n_jobs=-1):
    normalized = normalized_components(components)
    builder = lambda fold, train: combine_kernel_components(
        fold, normalized, fixed_gammas, current_best_params, fine6_weights, train)
    grid = _evaluate_kernel_grid(cache, builder, c_grid, n_jobs)
    best_C = max(grid, key=lambda C: grid[C]["aucs"].mean())
    return {"components": normalized, "best_C": best_C, "aucs": grid[best_C]["aucs"],
            "oof": grid[best_C]["oof"], "grid": grid}


def run_composite_configs(cache, configs, fixed_gammas, current_best_params,
                          fine6_weights, c_grid, n_jobs=-1):
    runs = {}
    for name, spec in configs.items():
        components = spec.get("components", spec)
        started = perf_counter()
        run = evaluate_composite_grid(cache, components, fixed_gammas, current_best_params,
                                      fine6_weights, c_grid, n_jobs)
        run["fit_time"] = perf_counter() - started
        run["type"] = spec.get("type", "composite") if "components" in spec else "composite"
        runs[name] = run
    return runs


def score_current_best_component(cache, current_best_params, fixed_gammas,
                                 fine6_weights, n_jobs=-1):
    """Sanity path through the generic composite implementation."""
    run = evaluate_composite_grid(
        cache, {"current_best": 1.0}, fixed_gammas, current_best_params,
        fine6_weights, [current_best_params["C"]], n_jobs)
    return run["aucs"], run["oof"]


def composite_comparison_row(name, kind, run, current_aucs, current_oof,
                             fine6_aucs, fine6_oof):
    aucs, oof = run["aucs"], run["oof"]
    delta = aucs - current_aucs
    return {"config": name, "type": kind, "CV_mean": aucs.mean(), "CV_std": aucs.std(),
            "CV_median": np.median(aucs), "delta_vs_current": delta.mean(),
            "wins_vs_current": int((delta > 0).sum()), "best_C": run["best_C"],
            "corr_with_current": spearmanr(oof.ravel(), current_oof.ravel()).statistic,
            "corr_with_fine6": spearmanr(oof.ravel(), fine6_oof.ravel()).statistic,
            "fit_time": run.get("fit_time", np.nan)}


def confirmed_comparison_row(name, kind, run, current_aucs, current_oof,
                             fine6_aucs, fine6_oof):
    """Paired 25-fold statistics for a preselected composite configuration."""
    row = composite_comparison_row(name, kind, run, current_aucs, current_oof,
                                   fine6_aucs, fine6_oof)
    delta = run["aucs"] - current_aucs
    row.update({"losses_vs_current": int((delta < 0).sum()),
                "wilcoxon_p_vs_current": wilcoxon(delta).pvalue if np.any(delta) else 1.0,
                "mean_fold_delta": delta.mean(), "median_fold_delta": np.median(delta),
                "min_fold_delta": delta.min(), "max_fold_delta": delta.max()})
    return row


def score_external_composite(fold, components, fixed_gammas, current_best_params,
                             fine6_weights, C):
    Ktr = combine_kernel_components(fold, components, fixed_gammas,
                                    current_best_params, fine6_weights, True)
    Kva = combine_kernel_components(fold, components, fixed_gammas,
                                    current_best_params, fine6_weights, False)
    score = SVC(C=C, kernel="precomputed").fit(Ktr, fold.y_train).decision_function(Kva)
    return roc_auc_score(fold.y_valid, score), score


def optimize_fine6_optuna(cache, fixed_gammas, fine6_weights, n_trials=50,
                          seed=42, n_jobs=-1, c_bounds=(2.0, 50.0),
                          temperature_bounds=(0.5, 2.0)):
    """Tune robust global fine_6 parameters on cached CV distances.

    Relative block weights and relative standalone gamma scales stay fixed.
    Optuna tunes SVC C and one shared kernel temperature, avoiding a fragile
    seven-dimensional per-block search on a small dataset.
    """
    import optuna

    config = {"mode": "additive", "weights": dict(fine6_weights)}
    active = [b for b, w in normalized_weights(fine6_weights).items() if w > 0]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        C = trial.suggest_float("C", float(c_bounds[0]), float(c_bounds[1]), log=True)
        temperature = trial.suggest_float(
            "kernel_temperature", float(temperature_bounds[0]),
            float(temperature_bounds[1]), log=True)
        gammas = {b: float(fixed_gammas[b]) * temperature for b in active}
        run = evaluate_config_grid(cache, config, gammas, [C], n_jobs)
        aucs = run["aucs"]
        trial.set_user_attr("cv_std", float(aucs.std()))
        trial.set_user_attr("cv_median", float(np.median(aucs)))
        return float(aucs.mean())

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=True, n_jobs=1)
    best_C = float(study.best_params["C"])
    temperature = float(study.best_params["kernel_temperature"])
    best_gammas = {b: float(fixed_gammas[b]) * temperature for b in active}
    best_run = evaluate_config_grid(cache, config, best_gammas, [best_C], n_jobs)
    params = {"C": best_C, "kernel_temperature": temperature,
              "gammas": best_gammas, "weights": normalized_weights(fine6_weights)}
    return study, params, best_run


def fit_fine6_submission_model(X_train, y_train, X_test, fixed_gammas,
                               fine6_weights, C, kernel_temperature=1.0,
                               seed=42, n_bins=8, alpha=5.0,
                               pca_components=128, probability=True):
    """Fit final fine_6 on train only and predict test probabilities.

    Validation is deliberately not accepted by this API. Heavy supervised
    features are built once; kernels are accumulated one block at a time.
    """
    X_train, y_train, X_test = np.asarray(X_train), np.asarray(y_train), np.asarray(X_test)
    train, test, _ = make_feature_blocks(
        X_train, y_train, X_test, seed + INNER_SEED_OFFSET,
        n_bins, alpha, pca_components)
    weights = normalized_weights(fine6_weights)
    K_train = K_test = None
    for block, weight in weights.items():
        if weight <= 0:
            continue
        gamma = float(fixed_gammas[block]) * float(kernel_temperature)
        D_train = cdist(train[block], train[block], "sqeuclidean")
        D_test = cdist(test[block], train[block], "sqeuclidean")
        block_train = np.exp(-gamma * D_train).astype(np.float32)
        block_test = np.exp(-gamma * D_test).astype(np.float32)
        if K_train is None:
            K_train = np.float32(weight) * block_train
            K_test = np.float32(weight) * block_test
        else:
            K_train += np.float32(weight) * block_train
            K_test += np.float32(weight) * block_test
        del D_train, D_test, block_train, block_test
    if not np.allclose(np.diag(K_train), 1.0, atol=2e-5):
        raise RuntimeError("final fine_6 kernel diagonal is not one")
    import warnings
    model = SVC(C=float(C), kernel="precomputed", probability=bool(probability),
                random_state=seed)
    with warnings.catch_warnings():
        # sklearn 1.9 deprecates this flag, but libsvm's internal calibration
        # remains the correct probability path for a precomputed Gram matrix.
        warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated", category=FutureWarning)
        model.fit(K_train, y_train)
    decision = model.decision_function(K_test)
    probabilities = model.predict_proba(K_test)[:, list(model.classes_).index(1)] if probability else None
    return model, decision, probabilities


def normalize_blend_weights(weights, available_models):
    """Validate prediction-blend weights without accepting silent model typos."""
    unknown = set(weights) - set(available_models)
    if unknown:
        raise KeyError(f"unknown prediction models: {sorted(unknown)}")
    values = {name: float(weight) for name, weight in weights.items()}
    if (not values or any(not np.isfinite(w) or w < 0 for w in values.values())
            or sum(values.values()) <= 0):
        raise ValueError("blend weights must be finite, nonnegative and have a positive sum")
    total = sum(values.values())
    return {name: weight / total for name, weight in values.items()}


def percentile_rank(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("rank input must be a finite non-empty vector")
    return rankdata(values, method="average") / len(values)


def foldwise_rank_oof(cache, oof):
    """Rank predictions independently inside each repeat/fold validation set."""
    oof = np.asarray(oof, dtype=float)
    expected = (len(cache.seeds), cache.n_samples)
    if oof.shape != expected or not np.isfinite(oof).all():
        raise ValueError(f"OOF must have finite shape {expected}, got {oof.shape}")
    ranked = np.full_like(oof, np.nan, dtype=float)
    for fold in cache.folds:
        ranked[fold.repeat, fold.valid_idx] = percentile_rank(oof[fold.repeat, fold.valid_idx])
    if not np.isfinite(ranked).all():
        raise RuntimeError("fold-wise ranked OOF is incomplete")
    return ranked


def fold_aucs_from_oof(cache, oof):
    """Recompute fold AUCs in cache order and thereby verify OOF alignment."""
    oof = np.asarray(oof, dtype=float)
    return np.asarray([
        roc_auc_score(fold.y_valid, oof[fold.repeat, fold.valid_idx])
        for fold in cache.folds
    ])


def evaluate_rank_blend(cache, model_oof, blend_weights):
    """Evaluate a prediction-level percentile-rank blend fold by fold."""
    if not model_oof:
        raise ValueError("model_oof is empty")
    expected = (len(cache.seeds), cache.n_samples)
    arrays = {name: np.asarray(values, dtype=float) for name, values in model_oof.items()}
    for name, values in arrays.items():
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"{name} OOF must have finite shape {expected}, got {values.shape}")
    weights = normalize_blend_weights(blend_weights, arrays)
    blended_oof = np.full(expected, np.nan, dtype=float)
    aucs = []
    for fold in cache.folds:
        score = np.zeros(len(fold.valid_idx), dtype=float)
        for name, weight in weights.items():
            fold_prediction = arrays[name][fold.repeat, fold.valid_idx]
            score += weight * percentile_rank(fold_prediction)
        blended_oof[fold.repeat, fold.valid_idx] = score
        aucs.append(roc_auc_score(fold.y_valid, score))
    if not np.isfinite(blended_oof).all():
        raise RuntimeError("rank-blend OOF is incomplete")
    return np.asarray(aucs), blended_oof, weights


def run_rank_blend_configs(cache, model_oof, configs):
    return {name: dict(zip(("aucs", "oof", "weights"),
                           evaluate_rank_blend(cache, model_oof, weights)))
            for name, weights in configs.items()}


def rank_blend_result_row(name, aucs, oof, weights, fine6_aucs,
                          fine6_rank_oof, model_order=None):
    aucs, oof = np.asarray(aucs), np.asarray(oof)
    delta = aucs - np.asarray(fine6_aucs)
    model_order = tuple(model_order or (
        "fine6", "geometry", "density", "knn", "coord",
        "pca128", "current_best", "aggregates"))
    row = {"config": name, "CV_mean": aucs.mean(), "CV_std": aucs.std(),
           "CV_median": np.median(aucs), "delta_vs_fine6": delta.mean(),
           "wins_vs_fine6": int((delta > 0).sum()),
           "losses_vs_fine6": int((delta < 0).sum()),
           "mean_fold_delta": delta.mean(), "median_fold_delta": np.median(delta),
           "min_fold_delta": delta.min(), "max_fold_delta": delta.max(),
           "wilcoxon_p_vs_fine6": wilcoxon(delta).pvalue if np.any(delta) else 1.0,
           "corr_with_fine6": spearmanr(oof.ravel(), fine6_rank_oof.ravel()).statistic}
    row.update({f"w_{model}": float(weights.get(model, 0.0)) for model in model_order})
    return row


def evaluate_rank_blend_validation(y_valid, model_scores, blend_weights):
    """Apply already-selected weights; ranks are computed over validation only."""
    y_valid = np.asarray(y_valid)
    arrays = {name: np.asarray(score, dtype=float) for name, score in model_scores.items()}
    if any(values.shape != (len(y_valid),) or not np.isfinite(values).all()
           for values in arrays.values()):
        raise ValueError("validation decision-score shape mismatch")
    weights = normalize_blend_weights(blend_weights, arrays)
    score = sum(weight * percentile_rank(arrays[name]) for name, weight in weights.items())
    return roc_auc_score(y_valid, score), score, weights


def get_standalone_C(standalone_diagnostics, block):
    rows = standalone_diagnostics.loc[standalone_diagnostics["block"].eq(block), "best_C"]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one standalone C for {block}, got {len(rows)}")
    return float(rows.iloc[0])


def prepare_blend_model_runs(cache, model_specs, fixed_gammas, standalone_diagnostics,
                             kernel_configs, config_runs, current_best_params,
                             n_jobs=-1):
    """Create aligned repeated-CV OOF runs from a declarative model registry."""
    runs = {}
    for name, spec in model_specs.items():
        kind = spec.get("kind")
        if kind == "kernel_config":
            config_name = spec["config"]
            if config_name not in kernel_configs or config_name not in config_runs:
                raise KeyError(f"unknown cached kernel config {config_name}")
            C = float(config_runs[config_name]["best_C"])
            run = evaluate_config_grid(
                cache, kernel_configs[config_name], fixed_gammas, [C], n_jobs)
            run.update({"C": C, "fixed_gamma": np.nan,
                        "quick_CV_mean": float(config_runs[config_name]["aucs"].mean()),
                        "kind": kind, "config": config_name})
        elif kind == "current_best":
            aucs, oof = score_current_best(cache, current_best_params, False, n_jobs)
            run = {"aucs": aucs, "oof": oof, "C": float(current_best_params["C"]),
                   "fixed_gamma": np.nan, "quick_CV_mean": np.nan,
                   "kind": kind}
        elif kind == "block":
            block = spec["block"]
            if block not in BLOCKS:
                raise KeyError(f"unknown cached block {block}")
            C = get_standalone_C(standalone_diagnostics, block)
            gamma = float(fixed_gammas[block])
            run = evaluate_config_grid(
                cache, {"mode": "product", "weights": {block: 1.0}},
                {block: gamma}, [C], n_jobs)
            quick_rows = standalone_diagnostics.loc[
                standalone_diagnostics["block"].eq(block), "CV_mean"]
            run.update({"C": C, "fixed_gamma": gamma,
                        "quick_CV_mean": float(quick_rows.iloc[0]),
                        "kind": kind, "block": block})
        else:
            raise ValueError(f"unsupported blend model kind for {name}: {kind}")
        if len(run["aucs"]) != len(cache.folds):
            raise RuntimeError(f"{name} fold count mismatch")
        recomputed = fold_aucs_from_oof(cache, run["oof"])
        if not np.allclose(recomputed, run["aucs"], atol=1e-12):
            raise RuntimeError(f"{name} OOF/fold ordering mismatch")
        runs[name] = run
    return runs


def predict_external_blend_models(fold, required_models, model_specs, model_runs,
                                  fixed_gammas, kernel_configs,
                                  current_best_params, fine6_weights):
    """Lazily fit only registered models needed by selected validation blends."""
    scores, aucs = {}, {}
    for name in sorted(set(required_models)):
        if name not in model_specs or name not in model_runs:
            raise KeyError(f"unregistered validation model {name}")
        spec, run = model_specs[name], model_runs[name]
        kind = spec["kind"]
        if kind == "kernel_config":
            auc, score = score_external_config(
                fold, kernel_configs[spec["config"]], fixed_gammas, run["C"])
        elif kind == "current_best":
            auc, score = score_external_composite(
                fold, {"current_best": 1.0}, fixed_gammas,
                current_best_params, fine6_weights, run["C"])
        elif kind == "block":
            block = spec["block"]
            auc, score = score_external_config(
                fold, {"mode": "product", "weights": {block: 1.0}},
                {block: fixed_gammas[block]}, run["C"])
        else:
            raise ValueError(f"unsupported validation model kind {kind}")
        aucs[name], scores[name] = auc, score
    return aucs, scores


ENGINEERED_AUX_BLOCKS = ("geometry", "aggregates", "density", "knn", "coord")
AUXILIARY_BLOCKS = ENGINEERED_AUX_BLOCKS + ("pca128",)


def auxiliary_kernel_config(editable_weights):
    """Return a normalized additive auxiliary config with raw hard-disabled."""
    unknown = set(editable_weights) - set(AUXILIARY_BLOCKS)
    if unknown:
        raise KeyError(
            f"auxiliary weights may contain only {AUXILIARY_BLOCKS}; "
            f"raw is forbidden and unknown keys are invalid: {sorted(unknown)}")
    values = np.asarray([float(editable_weights.get(block, 0.0))
                         for block in AUXILIARY_BLOCKS])
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError(
            "auxiliary weights must be finite, nonnegative and have a positive sum")
    values /= values.sum()
    weights = {block: 0.0 for block in BLOCKS}
    weights.update(dict(zip(AUXILIARY_BLOCKS, values)))
    assert weights["raw"] == 0.0
    return {"mode": "additive", "weights": weights}


def run_auxiliary_quick(cache, configs, fixed_gammas, c_grid, n_jobs=-1):
    """Select a small-grid C for every explicit raw-free auxiliary model."""
    runs = {}
    for name, spec in configs.items():
        config = auxiliary_kernel_config(spec["weights"])
        run = evaluate_config_grid(cache, config, fixed_gammas, c_grid, n_jobs)
        run["kernel_config"] = config
        runs[name] = run
    return runs


def run_auxiliary_fixed(cache, quick_runs, fixed_gammas, n_jobs=-1):
    """Evaluate every auxiliary model on repeated CV at its QUICK-selected C."""
    runs = {}
    for name, quick in quick_runs.items():
        run = evaluate_config_grid(
            cache, quick["kernel_config"], fixed_gammas,
            [quick["best_C"]], n_jobs)
        run["kernel_config"] = quick["kernel_config"]
        run["quick_CV_mean"] = float(quick["aucs"].mean())
        runs[name] = run
    return runs


def engineered_aux_kernel_config(editable_weights):
    """Return a normalized additive config with raw and PCA hard-disabled."""
    unknown = set(editable_weights) - set(ENGINEERED_AUX_BLOCKS)
    if unknown:
        raise KeyError(
            f"engineered-only weights may contain only {ENGINEERED_AUX_BLOCKS}; "
            f"forbidden/unknown: {sorted(unknown)}")
    values = np.asarray([float(editable_weights.get(block, 0.0))
                         for block in ENGINEERED_AUX_BLOCKS])
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("engineered-only weights must be finite, nonnegative and have a positive sum")
    values /= values.sum()
    weights = {block: 0.0 for block in BLOCKS}
    weights.update(dict(zip(ENGINEERED_AUX_BLOCKS, values)))
    assert weights["raw"] == 0.0 and weights["pca128"] == 0.0
    return {"mode": "additive", "weights": weights}


def run_engineered_aux_quick(cache, configs, fixed_gammas, c_grid, n_jobs=-1):
    """Select a small-grid C for each explicit engineered-only configuration."""
    runs = {}
    for name, spec in configs.items():
        config = engineered_aux_kernel_config(spec["weights"])
        run = evaluate_config_grid(cache, config, fixed_gammas, c_grid, n_jobs)
        run["kernel_config"] = config
        runs[name] = run
    return runs


def run_engineered_aux_fixed(cache, quick_runs, fixed_gammas, n_jobs=-1):
    """Evaluate each auxiliary model on repeated CV at its QUICK-selected C."""
    runs = {}
    for name, quick in quick_runs.items():
        run = evaluate_config_grid(
            cache, quick["kernel_config"], fixed_gammas,
            [quick["best_C"]], n_jobs)
        run["kernel_config"] = quick["kernel_config"]
        run["quick_CV_mean"] = float(quick["aucs"].mean())
        runs[name] = run
    return runs


def auxiliary_blend_result_row(name, aux_model, run, fine6_aucs,
                               fine6_rank_oof):
    aucs, oof = np.asarray(run["aucs"]), np.asarray(run["oof"])
    delta = aucs - np.asarray(fine6_aucs)
    return {"config": name, "aux_model": aux_model,
            "CV_mean": aucs.mean(), "CV_std": aucs.std(),
            "CV_median": np.median(aucs), "delta_vs_fine6": delta.mean(),
            "wins_vs_fine6": int((delta > 0).sum()),
            "losses_vs_fine6": int((delta < 0).sum()),
            "mean_fold_delta": delta.mean(), "median_fold_delta": np.median(delta),
            "min_fold_delta": delta.min(), "max_fold_delta": delta.max(),
            "wilcoxon_p_vs_fine6": wilcoxon(delta).pvalue if np.any(delta) else 1.0,
            "corr_with_fine6": spearmanr(oof.ravel(), fine6_rank_oof.ravel()).statistic,
            "w_fine6": float(run["weights"].get("fine6", 0.0)),
            "w_aux": float(run["weights"].get(aux_model, 0.0))}


def engineered_blend_result_row(name, aux_model, run, fine6_aucs,
                                fine6_rank_oof):
    """Backward-compatible alias for notebooks created before PCA unification."""
    return auxiliary_blend_result_row(
        name, aux_model, run, fine6_aucs, fine6_rank_oof)


# ---------------------------------------------------------------------------
# Feature-block quality experiments (independent from the legacy v2 cache)
# ---------------------------------------------------------------------------

FEATURE_QUALITY_CACHE_VERSION = 1
FEATURE_QUALITY_REFERENCE_SPECS = {
    "raw_reference": {"source": "raw", "transform": "identity"},
    "pca128_reference": {"source": "pca128", "transform": "identity"},
}


@dataclass
class FeatureQualityFold:
    repeat: int
    fold: int
    seed: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    d2_train: dict[str, Path | np.ndarray]
    d2_valid: dict[str, Path | np.ndarray]
    features_train: dict[str, Path | np.ndarray]
    features_valid: dict[str, Path | np.ndarray]
    gamma0: dict[str, float]
    metadata: dict[str, dict]
    cache_key: str


@dataclass
class FeatureQualityCache:
    folds: list[FeatureQualityFold]
    seeds: tuple[int, ...]
    n_samples: int
    cache_dir: Path
    cache_hits: int
    cache_builds: int
    variant_specs: dict[str, dict]


def _validate_feature_quality_specs(variant_specs):
    allowed_sources = set(BLOCKS)
    allowed_transforms = {"identity", "pca_whiten", "corr_prune"}
    clean = {}
    for name, raw_spec in variant_specs.items():
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(f"unsafe feature-quality config name: {name!r}")
        spec = dict(raw_spec)
        source, transform = spec.get("source"), spec.get("transform")
        if source not in allowed_sources:
            raise KeyError(f"unknown source block for {name}: {source}")
        if transform not in allowed_transforms:
            raise ValueError(f"unknown transform for {name}: {transform}")
        expected = {"source", "transform"}
        if transform == "pca_whiten":
            expected.add("n_components")
            if int(spec.get("n_components", 0)) <= 0:
                raise ValueError(f"n_components must be positive for {name}")
            spec["n_components"] = int(spec["n_components"])
        elif transform == "corr_prune":
            expected.add("threshold")
            threshold = float(spec.get("threshold", np.nan))
            if not 0 < threshold <= 1:
                raise ValueError(f"threshold must be in (0, 1] for {name}")
            spec["threshold"] = threshold
        unknown = set(spec) - expected
        missing = expected - set(spec)
        if unknown or missing:
            raise ValueError(
                f"invalid keys for {name}; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}")
        clean[name] = spec
    return clean


def _quality_spearman_abs(A):
    A = np.asarray(A, dtype=float)
    n_features = A.shape[1]
    if n_features == 1:
        return np.ones((1, 1), dtype=float)
    ranked = rankdata(A, axis=0, method="average")
    corr = np.asarray(np.corrcoef(ranked, rowvar=False), dtype=float)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _quality_structure(A):
    A = np.asarray(A, dtype=float)
    covariance = np.atleast_2d(np.cov(A, rowvar=False, ddof=1))
    eigvals = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = eigvals.sum()
    effective_rank = (total ** 2 / np.square(eigvals).sum()
                      if total > 0 and np.square(eigvals).sum() > 0 else 0.0)
    corr = _quality_spearman_abs(A)
    values = corr[np.triu_indices(A.shape[1], 1)]
    return {
        "effective_rank": float(effective_rank),
        "effective_rank_ratio": float(effective_rank / A.shape[1]),
        "max_abs_spearman": float(values.max()) if len(values) else 0.0,
        "pairs_spearman_gt_095": int((values > 0.95).sum()),
    }


def _quality_transform_pair(A, B, feature_names, spec, builder_seed):
    """Fit an unsupervised variant on outer-train A and transform outer-valid B."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    names = list(feature_names)
    transform = spec["transform"]
    metadata = {
        "original_dim": int(A.shape[1]),
        "transform": transform,
        "source_block": spec["source"],
        "explained_variance_ratio_sum": None,
        "min_kept_explained_variance": None,
        "threshold": None,
        "kept_feature_names": names,
        "dropped_feature_names": [],
    }
    if transform == "identity":
        A_out, B_out = A, B
    elif transform == "pca_whiten":
        n_components = int(spec["n_components"])
        maximum = min(A.shape[1], len(A) - 1)
        if n_components > maximum:
            raise ValueError(
                f"n_components={n_components} exceeds fold limit {maximum}")
        pca = PCA(n_components=n_components, random_state=builder_seed)
        A_pca, B_pca = pca.fit_transform(A), pca.transform(B)
        if np.min(pca.explained_variance_) < 1e-10:
            raise ValueError(
                "retained PCA component has explained_variance_ < 1e-10")
        scaler = StandardScaler()
        A_out, B_out = scaler.fit_transform(A_pca), scaler.transform(B_pca)
        metadata.update({
            "n_components": n_components,
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()),
            "min_kept_explained_variance": float(
                pca.explained_variance_.min()),
            "kept_feature_names": [f"pc_{i}" for i in range(n_components)],
        })
    elif transform == "corr_prune":
        threshold = float(spec["threshold"])
        abs_corr = _quality_spearman_abs(A)
        keep = []
        for index in range(A.shape[1]):
            if not keep or np.all(abs_corr[index, keep] < threshold):
                keep.append(index)
        if not keep:
            raise RuntimeError("correlation pruning removed every feature")
        dropped = [index for index in range(A.shape[1]) if index not in keep]
        scaler = StandardScaler()
        A_out = scaler.fit_transform(A[:, keep])
        B_out = scaler.transform(B[:, keep])
        metadata.update({
            "threshold": threshold,
            "kept_feature_names": [names[index] for index in keep],
            "dropped_feature_names": [names[index] for index in dropped],
        })
    else:  # guarded by _validate_feature_quality_specs
        raise ValueError(f"unsupported transform {transform}")

    if not np.isfinite(A_out).all() or not np.isfinite(B_out).all():
        raise RuntimeError("non-finite transformed feature-quality block")
    means, stds = A_out.mean(axis=0), A_out.std(axis=0, ddof=0)
    nonconstant = stds >= 1e-8
    if nonconstant.any():
        if not np.allclose(means[nonconstant], 0.0, atol=1e-6):
            raise RuntimeError("transformed outer-train means are not approximately zero")
        if not np.allclose(stds[nonconstant], 1.0, atol=1e-5):
            raise RuntimeError("transformed outer-train stds are not approximately one")
    metadata.update({
        "output_dim": int(A_out.shape[1]),
        "dropped_dim": int(A.shape[1] - A_out.shape[1]),
        **_quality_structure(A_out),
    })
    return np.asarray(A_out, np.float32), np.asarray(B_out, np.float32), metadata


def _quality_fold_key(seed, fold, train_idx, valid_idx, specs, n_bins,
                      alpha, pca_components):
    split_hash = sha256(
        np.asarray(train_idx, dtype=np.int32).tobytes()
        + np.asarray(valid_idx, dtype=np.int32).tobytes()).hexdigest()[:16]
    spec = {
        "version": FEATURE_QUALITY_CACHE_VERSION,
        "seed": int(seed), "fold": int(fold), "split": split_hash,
        "builder": FEATURE_BUILDER_CONFIG,
        "inner_seed_offset": INNER_SEED_OFFSET,
        "n_bins": int(n_bins), "alpha": float(alpha),
        "pca_components": int(pca_components),
        "ppca_components": list(PPCA_COMPONENTS),
        "variants": specs,
    }
    digest = sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    return digest, spec


def _quality_manifest_complete(folder, spec):
    path = folder / "manifest.json"
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (manifest.get("spec") == spec
            and all((folder / name).exists() for name in manifest.get("files", [])))


def _quality_fold_from_disk(folder, repeat, fold, seed, train_idx, valid_idx,
                            y, key, names):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return FeatureQualityFold(
        repeat, fold, seed, train_idx, valid_idx,
        y[train_idx], y[valid_idx],
        _paths(folder, "d2_train", names),
        _paths(folder, "d2_valid", names),
        _paths(folder, "feature_train", names),
        _paths(folder, "feature_valid", names),
        {name: float(value) for name, value in manifest["gamma0"].items()},
        manifest["metadata"], key)


def _build_feature_quality_fold(X, y, repeat, seed, fold, train_idx, valid_idx,
                                folder, key, cache_spec, specs, n_bins, alpha,
                                pca_components):
    print(f"build feature-quality cache: repeat={repeat + 1}, fold={fold + 1}")
    builder_seed = seed + INNER_SEED_OFFSET + fold
    train_blocks, valid_blocks, artifacts = make_feature_blocks(
        X[train_idx], y[train_idx], X[valid_idx], builder_seed,
        n_bins, alpha, pca_components)
    folder.mkdir(parents=True, exist_ok=True)
    gamma0, metadata = {}, {}
    for variant_index, (name, variant_spec) in enumerate(specs.items()):
        source = variant_spec["source"]
        A, B, variant_metadata = _quality_transform_pair(
            train_blocks[source], valid_blocks[source],
            artifacts["feature_names"][source], variant_spec,
            builder_seed + variant_index)
        Dtr = cdist(A, A, metric="sqeuclidean").astype(np.float32)
        Dva = cdist(B, A, metric="sqeuclidean").astype(np.float32)
        gamma0[name], _ = _median_gamma_from_distance(
            Dtr, builder_seed + variant_index)
        np.save(folder / f"feature_train_{name}.npy", A, allow_pickle=False)
        np.save(folder / f"feature_valid_{name}.npy", B, allow_pickle=False)
        np.save(folder / f"d2_train_{name}.npy", Dtr, allow_pickle=False)
        np.save(folder / f"d2_valid_{name}.npy", Dva, allow_pickle=False)
        metadata[name] = variant_metadata
        del A, B, Dtr, Dva
    np.save(folder / "train_idx.npy", np.asarray(train_idx, np.int32), allow_pickle=False)
    np.save(folder / "valid_idx.npy", np.asarray(valid_idx, np.int32), allow_pickle=False)
    files = [item.name for item in folder.iterdir() if item.name != "manifest.json"]
    manifest = {
        "spec": cache_spec,
        "gamma0": gamma0,
        "metadata": metadata,
        "files": sorted(files),
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return _quality_fold_from_disk(
        folder, repeat, fold, seed, train_idx, valid_idx, y, key, tuple(specs))


def build_or_load_feature_quality_cache(
        X, y, variant_specs, seeds, n_splits=5,
        cache_dir=".svc_feature_quality_cache_v1", rebuild=False,
        n_bins=8, alpha=5.0, pca_components=128, n_jobs=1,
        include_references=True):
    """Build each outer fold once, then derive and cache all requested variants."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    specs = _validate_feature_quality_specs(variant_specs)
    if include_references:
        overlap = set(specs) & set(FEATURE_QUALITY_REFERENCE_SPECS)
        if overlap:
            raise KeyError(f"reserved reference config names: {sorted(overlap)}")
        specs = {**specs, **FEATURE_QUALITY_REFERENCE_SPECS}
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs, folds, hits = [], [], 0
    for repeat, seed in enumerate(seeds):
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
            key, cache_spec = _quality_fold_key(
                seed, fold, train_idx, valid_idx, specs,
                n_bins, alpha, pca_components)
            folder = root / key
            if not rebuild and _quality_manifest_complete(folder, cache_spec):
                print(f"feature-quality cache hit: repeat={repeat + 1}, fold={fold + 1}")
                folds.append(_quality_fold_from_disk(
                    folder, repeat, fold, seed, train_idx, valid_idx,
                    y, key, tuple(specs)))
                hits += 1
            else:
                jobs.append((repeat, seed, fold, train_idx, valid_idx,
                             folder, key, cache_spec))
    built = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_build_feature_quality_fold)(
            X, y, *job, specs, n_bins, alpha, pca_components)
        for job in jobs)
    folds.extend(built)
    folds.sort(key=lambda item: (item.repeat, item.fold))
    return FeatureQualityCache(
        folds, tuple(seeds), len(X), root, hits, len(built), specs)


def _evaluate_feature_quality_grid(cache, config, gamma_values, c_grid,
                                   n_jobs=-1):
    gamma_values = tuple(float(value) for value in gamma_values)
    c_grid = tuple(float(value) for value in c_grid)

    def one(fold):
        Dtr = _load_array(fold.d2_train[config])
        Dva = _load_array(fold.d2_valid[config])
        predictions = {}
        for gamma in gamma_values:
            Ktr = np.exp(-gamma * Dtr).astype(np.float32)
            Kva = np.exp(-gamma * Dva).astype(np.float32)
            for C in c_grid:
                model = SVC(C=C, kernel="precomputed").fit(Ktr, fold.y_train)
                score = model.decision_function(Kva)
                predictions[(gamma, C)] = (
                    roc_auc_score(fold.y_valid, score), score)
        return fold.repeat, fold.valid_idx, predictions

    items = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one)(fold) for fold in cache.folds)
    output = {
        key: {"aucs": [], "oof": np.full(
            (len(cache.seeds), cache.n_samples), np.nan)}
        for key in ((gamma, C) for gamma in gamma_values for C in c_grid)
    }
    for repeat, valid_idx, predictions in items:
        for key, (auc, score) in predictions.items():
            output[key]["aucs"].append(auc)
            output[key]["oof"][repeat, valid_idx] = score
    for result in output.values():
        result["aucs"] = np.asarray(result["aucs"])
        if not np.isfinite(result["oof"]).all():
            raise RuntimeError("incomplete feature-quality OOF predictions")
    return output


def run_feature_quality_screening(cache, gamma_multipliers, c_grid, n_jobs=-1):
    """Tune a coarse gamma multiplier/C grid fairly for every cached variant."""
    runs = {}
    for config in cache.variant_specs:
        started = perf_counter()
        fold_gamma0 = np.asarray([fold.gamma0[config] for fold in cache.folds])
        base_gamma = float(np.median(fold_gamma0))
        multipliers = tuple(float(value) for value in gamma_multipliers)
        gamma_values = tuple(base_gamma * value for value in multipliers)
        grid = _evaluate_feature_quality_grid(
            cache, config, gamma_values, c_grid, n_jobs)
        best_gamma, best_C = max(
            grid, key=lambda key: grid[key]["aucs"].mean())
        best = grid[(best_gamma, best_C)]
        runs[config] = {
            "config": config,
            "source_block": cache.variant_specs[config]["source"],
            "transform": cache.variant_specs[config]["transform"],
            "base_gamma": base_gamma,
            "best_gamma_multiplier": best_gamma / base_gamma,
            "best_gamma": best_gamma,
            "best_C": best_C,
            "aucs": best["aucs"],
            "oof": best["oof"],
            "grid": grid,
            "fit_time": perf_counter() - started,
        }
    return runs


def feature_quality_result_table(cache, runs):
    rows = []
    reference_names = set(FEATURE_QUALITY_REFERENCE_SPECS)
    for config, run in runs.items():
        source = run["source_block"]
        original = f"{source}_orig"
        dimensions = np.asarray([
            fold.metadata[config]["output_dim"] for fold in cache.folds])
        is_reference = config in reference_names
        if is_reference:
            delta = np.nan
            wins = np.nan
            corr = np.nan
        else:
            if original not in runs:
                raise KeyError(f"missing fair original baseline {original} for {config}")
            original_run = runs[original]
            fold_delta = run["aucs"] - original_run["aucs"]
            delta = float(fold_delta.mean())
            wins = int((fold_delta > 0).sum())
            corr = float(spearmanr(
                run["oof"].ravel(), original_run["oof"].ravel()).statistic)
        rows.append({
            "config": config,
            "source_block": source,
            "transform": run["transform"],
            "output_dim": (int(dimensions[0])
                           if np.all(dimensions == dimensions[0]) else np.nan),
            "median_output_dim": float(np.median(dimensions)),
            "best_gamma_multiplier": run["best_gamma_multiplier"],
            "best_gamma": run["best_gamma"],
            "best_C": run["best_C"],
            "CV_mean": run["aucs"].mean(),
            "CV_std": run["aucs"].std(),
            "CV_median": np.median(run["aucs"]),
            "delta_vs_source_original": delta,
            "wins_vs_source_original": wins,
            "corr_with_source_original_oof": corr,
            "fit_time": run["fit_time"],
        })
    return pd.DataFrame(rows)


def feature_quality_structural_table(cache):
    rows = []
    for config, spec in cache.variant_specs.items():
        metadata = [fold.metadata[config] for fold in cache.folds]
        values = lambda key: np.asarray([
            np.nan if item.get(key) is None else item[key] for item in metadata],
            dtype=float)
        explained = values("explained_variance_ratio_sum")
        rows.append({
            "config": config,
            "source_block": spec["source"],
            "transform": spec["transform"],
            "original_dim": int(np.median(values("original_dim"))),
            "median_output_dim": float(np.median(values("output_dim"))),
            "median_effective_rank": float(np.median(values("effective_rank"))),
            "median_effective_rank_ratio": float(np.median(
                values("effective_rank_ratio"))),
            "median_max_abs_spearman": float(np.median(
                values("max_abs_spearman"))),
            "median_pairs_spearman_gt_095": float(np.median(
                values("pairs_spearman_gt_095"))),
            "explained_variance_sum": (float(np.nanmedian(explained))
                                       if np.isfinite(explained).any() else np.nan),
            "median_kept_dim": float(np.median(values("output_dim"))),
            "min_kept_dim": int(np.min(values("output_dim"))),
            "max_kept_dim": int(np.max(values("output_dim"))),
        })
    return pd.DataFrame(rows)


def feature_quality_source_ranking(result_table):
    references = set(FEATURE_QUALITY_REFERENCE_SPECS)
    rows = []
    for source in ("geometry", "density", "coord", "aggregates"):
        group = result_table[
            result_table["source_block"].eq(source)
            & ~result_table["config"].isin(references)]
        if group.empty:
            continue
        best = group.loc[group["CV_mean"].idxmax()]
        original = group.loc[group["config"].eq(f"{source}_orig")].iloc[0]
        rows.append({
            "source_block": source,
            "best_variant": best["config"],
            "original_CV": original["CV_mean"],
            "best_variant_CV": best["CV_mean"],
            "delta": best["CV_mean"] - original["CV_mean"],
            "best_C": best["best_C"],
            "best_gamma": best["best_gamma"],
        })
    return pd.DataFrame(rows)


def check_feature_quality_identity_parity(cache, config, gamma, C,
                                          tolerance=1e-5, n_jobs=-1):
    """Rebuild identity distances and compare two precomputed-RBF paths.

    The legacy standalone block logic uses cached squared distances followed by
    a float32 precomputed kernel.  Comparing that path with ``kernel='rbf'`` is
    not an exact parity test because libsvm computes its internal kernel with a
    different floating-point path.  Here distances are independently rebuilt
    from the stored identity features, then evaluated with the same numerical
    convention as the legacy precomputed-kernel pipeline.
    """
    if cache.variant_specs[config]["transform"] != "identity":
        raise ValueError("parity check requires an identity config")
    cached = _evaluate_feature_quality_grid(cache, config, [gamma], [C], n_jobs)[
        (float(gamma), float(C))]

    def reconstructed(fold):
        A = _load_array(fold.features_train[config])
        B = _load_array(fold.features_valid[config])
        rebuilt_train = cdist(A, A, metric="sqeuclidean").astype(np.float32)
        rebuilt_valid = cdist(B, A, metric="sqeuclidean").astype(np.float32)
        cached_train = _load_array(fold.d2_train[config])
        cached_valid = _load_array(fold.d2_valid[config])
        distance_delta = max(
            float(np.max(np.abs(rebuilt_train - cached_train))),
            float(np.max(np.abs(rebuilt_valid - cached_valid))))
        Ktr = np.exp(-float(gamma) * rebuilt_train).astype(np.float32)
        Kva = np.exp(-float(gamma) * rebuilt_valid).astype(np.float32)
        model = SVC(C=C, kernel="precomputed").fit(Ktr, fold.y_train)
        score = model.decision_function(Kva)
        return roc_auc_score(fold.y_valid, score), distance_delta

    reconstructed_items = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(reconstructed)(fold) for fold in cache.folds)
    reconstructed_aucs = np.asarray([item[0] for item in reconstructed_items])
    max_distance_delta = max(item[1] for item in reconstructed_items)
    max_delta = float(np.max(np.abs(cached["aucs"] - reconstructed_aucs)))
    if max_distance_delta > 1e-6:
        raise RuntimeError(
            f"feature-quality identity distance parity failed for {config}: "
            f"max distance difference={max_distance_delta:.3g}")
    if max_delta >= tolerance:
        raise RuntimeError(
            f"feature-quality identity parity failed for {config}: "
            f"max AUC difference={max_delta:.3g}")
    return {"config": config, "C": float(C), "gamma": float(gamma),
            "max_distance_difference": max_distance_delta,
            "max_auc_difference": max_delta}


def evaluate_feature_quality_fixed(cache, config, gamma, C, n_jobs=-1):
    result = _evaluate_feature_quality_grid(
        cache, config, [gamma], [C], n_jobs)[(float(gamma), float(C))]
    return {"aucs": result["aucs"], "oof": result["oof"],
            "best_gamma": float(gamma), "best_C": float(C)}


def feature_quality_confirmation_table(cache, selected_configs, fixed_runs,
                                       quick_runs):
    rows = []
    for config in selected_configs:
        source = cache.variant_specs[config]["source"]
        original = f"{source}_orig"
        run, base = fixed_runs[config], fixed_runs[original]
        delta = run["aucs"] - base["aucs"]
        rows.append({
            "config": config,
            "CV_mean_25": run["aucs"].mean(),
            "CV_std_25": run["aucs"].std(),
            "CV_median_25": np.median(run["aucs"]),
            "delta_vs_original": delta.mean(),
            "wins_vs_original": int((delta > 0).sum()),
            "losses_vs_original": int((delta < 0).sum()),
            "mean_fold_delta": delta.mean(),
            "median_fold_delta": np.median(delta),
            "min_fold_delta": delta.min(),
            "max_fold_delta": delta.max(),
            "corr_with_original": spearmanr(
                run["oof"].ravel(), base["oof"].ravel()).statistic,
            "wilcoxon_p_vs_original": (wilcoxon(delta).pvalue
                                        if np.any(delta) else 1.0),
            "fixed_C": quick_runs[config]["best_C"],
            "fixed_gamma": quick_runs[config]["best_gamma"],
        })
    return pd.DataFrame(rows)

def load_existing_distance_cache(
        y, seeds, cache_dir=".svc_multikernel_cache_v2", n_splits=5,
        n_bins=8, alpha=5.0, pca_components=128):
    """Load an already-built legacy DistanceCache without creating CV splits.

    The function scans cache manifests and reuses the saved train/valid indices.
    It never calls ``StratifiedKFold`` and never calls ``make_feature_blocks``.
    Missing or ambiguous folds are an error instead of triggering a rebuild.
    """
    y = np.asarray(y)
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("existing-cache seed list contains duplicates")
    repeat_by_seed = {seed: repeat for repeat, seed in enumerate(seeds)}
    root = Path(cache_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"legacy cache directory does not exist: {root}")

    expected_common = {
        "version": CACHE_VERSION,
        "builder": FEATURE_BUILDER_CONFIG,
        "inner_seed_offset": INNER_SEED_OFFSET,
        "n_bins": int(n_bins),
        "alpha": float(alpha),
        "pca_components": int(pca_components),
        "ppca_components": list(PPCA_COMPONENTS),
    }
    candidates = {}
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        spec = manifest.get("spec", {})
        seed = spec.get("seed")
        fold_number = spec.get("fold")
        if seed not in repeat_by_seed:
            continue
        if not isinstance(fold_number, int) or not 0 <= fold_number < int(n_splits):
            continue
        if any(spec.get(key) != value for key, value in expected_common.items()):
            continue
        if not _manifest_complete(folder, spec):
            continue
        train_path, valid_path = folder / "train_idx.npy", folder / "valid_idx.npy"
        if not train_path.exists() or not valid_path.exists():
            continue
        train_idx = np.load(train_path, allow_pickle=False)
        valid_idx = np.load(valid_path, allow_pickle=False)
        if len(train_idx) + len(valid_idx) != len(y):
            continue
        if np.intersect1d(train_idx, valid_idx).size:
            continue
        key = (int(seed), int(fold_number))
        candidates.setdefault(key, []).append(
            (folder, str(folder.name), train_idx, valid_idx))

    missing = [
        (seed, fold)
        for seed in seeds
        for fold in range(int(n_splits))
        if (seed, fold) not in candidates
    ]
    ambiguous = {
        key: [str(item[0]) for item in values]
        for key, values in candidates.items()
        if len(values) != 1
    }
    if missing or ambiguous:
        raise RuntimeError(
            "required existing legacy cache is incomplete/ambiguous; "
            f"missing={missing}, ambiguous={ambiguous}. "
            "Clean experiments refuse to build new folds.")

    folds = []
    for seed in seeds:
        repeat = repeat_by_seed[seed]
        for fold_number in range(int(n_splits)):
            folder, key, train_idx, valid_idx = candidates[(seed, fold_number)][0]
            folds.append(_fold_from_disk(
                folder, repeat, fold_number, seed, train_idx, valid_idx, y, key))
    folds.sort(key=lambda item: (item.repeat, item.fold))
    return DistanceCache(
        folds=folds,
        seeds=seeds,
        n_samples=len(y),
        cache_dir=root,
        cache_hits=len(folds),
        cache_builds=0,
    )

# ---------------------------------------------------------------------------
# Clean multi-kernel experiments: derived whitening from the legacy cache only
# ---------------------------------------------------------------------------

CLEAN_DERIVED_CACHE_VERSION = 2
CLEAN_DERIVED_SPECS = {
    "coord_white4": {"source": "coord", "n_components": 4},
    "aggregates_white4": {"source": "aggregates", "n_components": 4},
}
CLEAN_BLOCKS = (
    "raw", "geometry", "density", "knn",
    "coord_white4", "aggregates_white4", "pca128",
)
CLEAN_BASE_BLOCKS = tuple(
    block for block in CLEAN_BLOCKS if block not in CLEAN_DERIVED_SPECS)


@dataclass
class CleanDerivedFold:
    """Sidecar distances derived strictly from an already-built DistanceFold."""

    repeat: int
    fold: int
    seed: int
    base_cache_key: str
    train_idx: np.ndarray
    valid_idx: np.ndarray
    d2_train: dict[str, Path | np.ndarray]
    d2_valid: dict[str, Path | np.ndarray]
    cache_key: str


@dataclass
class CleanDerivedCache:
    folds: list[CleanDerivedFold]
    cache_dir: Path
    cache_hits: int
    cache_builds: int
    block_names: tuple[str, ...]


def _clean_sidecar_spec(base_fold, specs):
    spec = {
        "version": CLEAN_DERIVED_CACHE_VERSION,
        "base_cache_key": str(base_fold.cache_key),
        "repeat": int(base_fold.repeat),
        "fold": int(base_fold.fold),
        "seed": int(base_fold.seed),
        "transforms": {
            name: {
                "source": str(cfg["source"]),
                "n_components": int(cfg["n_components"]),
                "pipeline": "cached_standardized_block->PCA->StandardScaler",
            }
            for name, cfg in specs.items()
        },
    }
    digest = sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    return digest, spec


def _clean_sidecar_complete(folder, spec):
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("spec") == spec
        and all((folder / name).exists() for name in manifest.get("files", []))
    )


def _clean_sidecar_from_disk(folder, base_fold, key, specs):
    names = tuple(specs)
    return CleanDerivedFold(
        repeat=base_fold.repeat,
        fold=base_fold.fold,
        seed=base_fold.seed,
        base_cache_key=base_fold.cache_key,
        train_idx=base_fold.train_idx,
        valid_idx=base_fold.valid_idx,
        d2_train=_paths(folder, "d2_train", names),
        d2_valid=_paths(folder, "d2_valid", names),
        cache_key=key,
    )


def _build_clean_sidecar_fold(base_fold, folder, key, cache_spec, specs):
    """Build only cheap unsupervised transforms from legacy cached features.

    This function intentionally never receives X/y and therefore cannot call
    ``make_feature_blocks`` or create a new outer split.
    """
    print(
        f"build clean derived cache: repeat={base_fold.repeat + 1}, "
        f"fold={base_fold.fold + 1}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    produced = []

    for name, cfg in specs.items():
        source = cfg["source"]
        if source not in base_fold.features_train or source not in base_fold.features_valid:
            raise KeyError(
                f"legacy cache does not contain cached features for source {source!r}"
            )
        A = np.asarray(_load_array(base_fold.features_train[source]), dtype=float)
        B = np.asarray(_load_array(base_fold.features_valid[source]), dtype=float)
        n_components = int(cfg["n_components"])
        maximum = min(A.shape[1], len(A) - 1)
        if n_components > maximum:
            raise ValueError(
                f"{name}: n_components={n_components} exceeds fold limit {maximum}"
            )

        # Use exactly the same transform implementation and dtype path as the
        # feature-quality experiment.  In particular, _quality_transform_pair
        # returns float32 arrays before cdist.  The previous clean implementation
        # computed cdist from float64 PCA/scaled arrays, which produced tiny
        # (~1e-6..1e-5) distance differences and could flip one ROC-AUC pair.
        builder_seed = int(base_fold.seed) + INNER_SEED_OFFSET + int(base_fold.fold)
        feature_names = [f"{source}_{i}" for i in range(A.shape[1])]
        A_out, B_out, _ = _quality_transform_pair(
            A, B, feature_names,
            {
                "source": source,
                "transform": "pca_whiten",
                "n_components": n_components,
            },
            builder_seed,
        )

        Dtr = cdist(A_out, A_out, metric="sqeuclidean").astype(np.float32)
        Dva = cdist(B_out, A_out, metric="sqeuclidean").astype(np.float32)
        np.save(folder / f"d2_train_{name}.npy", Dtr, allow_pickle=False)
        np.save(folder / f"d2_valid_{name}.npy", Dva, allow_pickle=False)
        produced.extend([f"d2_train_{name}.npy", f"d2_valid_{name}.npy"])

    manifest = {"spec": cache_spec, "files": sorted(produced)}
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return _clean_sidecar_from_disk(folder, base_fold, key, specs)


def build_or_load_clean_derived_cache(
        base_cache, cache_dir=".svc_clean_derived_cache_v1",
        rebuild=False, n_jobs=1, specs=None):
    """Derive whitening distances from an existing DistanceCache only.

    No ``StratifiedKFold`` and no ``make_feature_blocks`` call appears in this
    path.  Fold identities and seeds are inherited verbatim from ``base_cache``.
    """
    specs = dict(CLEAN_DERIVED_SPECS if specs is None else specs)
    unknown_sources = {
        cfg["source"] for cfg in specs.values()
        if cfg["source"] not in BLOCKS
    }
    if unknown_sources:
        raise KeyError(f"unknown clean derived sources: {sorted(unknown_sources)}")

    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    loaded, jobs, hits = [], [], 0

    for base_fold in base_cache.folds:
        key, cache_spec = _clean_sidecar_spec(base_fold, specs)
        folder = root / key
        if not rebuild and _clean_sidecar_complete(folder, cache_spec):
            print(
                f"clean derived cache hit: repeat={base_fold.repeat + 1}, "
                f"fold={base_fold.fold + 1}"
            )
            loaded.append(
                _clean_sidecar_from_disk(folder, base_fold, key, specs))
            hits += 1
        else:
            jobs.append((base_fold, folder, key, cache_spec))

    built = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_build_clean_sidecar_fold)(
            base_fold, folder, key, cache_spec, specs)
        for base_fold, folder, key, cache_spec in jobs
    )
    folds = loaded + built
    folds.sort(key=lambda item: (item.repeat, item.fold))

    if len(folds) != len(base_cache.folds):
        raise RuntimeError("clean sidecar fold count mismatch")
    for base_fold, clean_fold in zip(base_cache.folds, folds):
        if (
            base_fold.repeat != clean_fold.repeat
            or base_fold.fold != clean_fold.fold
            or base_fold.seed != clean_fold.seed
            or base_fold.cache_key != clean_fold.base_cache_key
            or not np.array_equal(base_fold.train_idx, clean_fold.train_idx)
            or not np.array_equal(base_fold.valid_idx, clean_fold.valid_idx)
        ):
            raise RuntimeError("clean sidecar is not aligned with the legacy cache")

    return CleanDerivedCache(
        folds=folds,
        cache_dir=root,
        cache_hits=hits,
        cache_builds=len(built),
        block_names=tuple(specs),
    )


def _clean_derived_by_base_key(clean_cache):
    mapping = {fold.base_cache_key: fold for fold in clean_cache.folds}
    if len(mapping) != len(clean_cache.folds):
        raise RuntimeError("duplicate base cache keys in clean sidecar")
    return mapping


def get_clean_distance(base_fold, clean_cache, block, train=True):
    """Return a cached squared-distance matrix for a clean block."""
    if block not in CLEAN_BLOCKS:
        raise KeyError(f"unknown clean block {block!r}")
    if block in CLEAN_DERIVED_SPECS:
        derived = _clean_derived_by_base_key(clean_cache).get(base_fold.cache_key)
        if derived is None:
            raise KeyError(
                f"missing clean sidecar distances for base fold {base_fold.cache_key}"
            )
        source = derived.d2_train if train else derived.d2_valid
    else:
        source = base_fold.d2_train if train else base_fold.d2_valid
    return _load_array(source[block])


def normalized_clean_weights(weights):
    unknown = set(weights) - set(CLEAN_BLOCKS)
    if unknown:
        raise KeyError(f"unknown clean blocks: {sorted(unknown)}")
    values = np.asarray(
        [float(weights.get(block, 0.0)) for block in CLEAN_BLOCKS], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError(
            "clean weights must be finite, nonnegative and have a positive sum")
    values /= values.sum()
    return dict(zip(CLEAN_BLOCKS, values))


def _clean_additive_kernel(base_fold, clean_cache, weights, gammas, train=True):
    result = None
    for block, weight in weights.items():
        if weight <= 0:
            continue
        if block not in gammas:
            raise KeyError(f"missing clean gamma for active block {block}")
        D = get_clean_distance(base_fold, clean_cache, block, train=train)
        kernel = np.exp(-float(gammas[block]) * D).astype(np.float32)
        if result is None:
            result = np.float32(weight) * kernel
        else:
            result += np.float32(weight) * kernel
    if result is None:
        raise RuntimeError("clean additive kernel has no active blocks")
    result = np.asarray(result, dtype=np.float32)
    if train and not np.allclose(np.diag(result), 1.0, atol=2e-5):
        raise RuntimeError("clean additive kernel diagonal is not one")
    return result


def evaluate_clean_additive_config(
        cache, clean_cache, weights, gammas, c_grid, n_jobs=-1):
    """Evaluate one clean additive kernel on exactly the folds in ``cache``."""
    normalized = normalized_clean_weights(weights)
    c_grid = tuple(float(value) for value in c_grid)

    # Alignment guard: clean stage must not invent or reorder folds.
    clean_keys = [
        (fold.repeat, fold.fold, fold.seed, fold.base_cache_key)
        for fold in clean_cache.folds
    ]
    base_keys = [
        (fold.repeat, fold.fold, fold.seed, fold.cache_key)
        for fold in cache.folds
    ]
    if clean_keys != base_keys:
        raise RuntimeError("clean cache fold keys differ from legacy cache fold keys")

    def one(base_fold):
        Ktr = _clean_additive_kernel(
            base_fold, clean_cache, normalized, gammas, train=True)
        Kva = _clean_additive_kernel(
            base_fold, clean_cache, normalized, gammas, train=False)
        predictions = []
        for C in c_grid:
            model = SVC(C=C, kernel="precomputed").fit(Ktr, base_fold.y_train)
            score = model.decision_function(Kva)
            predictions.append((roc_auc_score(base_fold.y_valid, score), score))
        return base_fold.repeat, base_fold.valid_idx, predictions

    items = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one)(fold) for fold in cache.folds)

    grid = {
        C: {
            "aucs": [],
            "oof": np.full((len(cache.seeds), cache.n_samples), np.nan),
        }
        for C in c_grid
    }
    for repeat, valid_idx, predictions in items:
        for C, (auc, score) in zip(c_grid, predictions):
            grid[C]["aucs"].append(auc)
            grid[C]["oof"][repeat, valid_idx] = score
    for C in c_grid:
        grid[C]["aucs"] = np.asarray(grid[C]["aucs"], dtype=float)
        if not np.isfinite(grid[C]["oof"]).all():
            raise RuntimeError("incomplete clean OOF predictions")

    best_C = max(c_grid, key=lambda C: grid[C]["aucs"].mean())
    best = grid[best_C]
    return {
        "weights": normalized,
        "gammas": {k: float(v) for k, v in gammas.items()},
        "best_C": float(best_C),
        "aucs": best["aucs"],
        "oof": best["oof"],
        "grid": grid,
    }


def screen_cached_block_gamma(
        cache, block, gamma_multipliers, c_grid, n_jobs=-1):
    """Cheap gamma/C screen using only an already-cached legacy distance block."""
    if block not in BLOCKS:
        raise KeyError(f"unknown cached legacy block {block!r}")
    base_gamma = float(np.median([fold.gamma0[block] for fold in cache.folds]))
    rows, runs = [], {}
    best = None
    for multiplier in gamma_multipliers:
        multiplier = float(multiplier)
        gamma = base_gamma * multiplier
        run = evaluate_config_grid(
            cache,
            {"mode": "product", "weights": {block: 1.0}},
            {block: gamma},
            c_grid,
            n_jobs,
        )
        runs[multiplier] = run
        row = {
            "block": block,
            "gamma_multiplier": multiplier,
            "gamma": gamma,
            "best_C": float(run["best_C"]),
            "CV_mean": float(run["aucs"].mean()),
            "CV_std": float(run["aucs"].std()),
            "CV_median": float(np.median(run["aucs"])),
        }
        rows.append(row)
        candidate = (row["CV_mean"], -multiplier, gamma, run)
        if best is None or candidate[0] > best[0]:
            best = candidate
    best_mean, _, best_gamma, best_run = best
    return (
        pd.DataFrame(rows).sort_values(
            ["CV_mean", "gamma_multiplier"], ascending=[False, True]
        ).reset_index(drop=True),
        {
            "block": block,
            "base_gamma": base_gamma,
            "best_gamma": float(best_gamma),
            "best_C": float(best_run["best_C"]),
            "CV_mean": float(best_mean),
            "run": best_run,
        },
        runs,
    )


def _paired_clean_stats(aucs, baseline_aucs):
    delta = np.asarray(aucs, dtype=float) - np.asarray(baseline_aucs, dtype=float)
    return {
        "delta": float(delta.mean()),
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
        "mean_fold_delta": float(delta.mean()),
        "median_fold_delta": float(np.median(delta)),
        "min_fold_delta": float(delta.min()),
        "max_fold_delta": float(delta.max()),
    }


def run_clean_pair_screening(
        cache, clean_cache, blocks, pair_weights, gammas, c_grid,
        raw_run, fine6_run, n_jobs=-1):
    """Screen raw + one clean block, without changing gamma values."""
    blocks = tuple(blocks)
    forbidden = set(blocks) - (set(CLEAN_BLOCKS) - {"raw", "pca128"})
    if forbidden:
        raise KeyError(f"invalid clean pair blocks: {sorted(forbidden)}")

    rows, runs = [], {}
    for block in blocks:
        for block_weight in pair_weights:
            block_weight = float(block_weight)
            if not 0 < block_weight < 1:
                raise ValueError("pair block weights must be strictly between 0 and 1")
            weights = {"raw": 1.0 - block_weight, block: block_weight}
            active_gammas = {
                "raw": float(gammas["raw"]),
                block: float(gammas[block]),
            }
            started = perf_counter()
            run = evaluate_clean_additive_config(
                cache, clean_cache, weights, active_gammas, c_grid, n_jobs)
            run["fit_time"] = perf_counter() - started
            config = f"raw_plus_{block}_w{int(round(block_weight * 100)):02d}"
            runs[config] = run

            raw_stats = _paired_clean_stats(run["aucs"], raw_run["aucs"])
            fine_stats = _paired_clean_stats(run["aucs"], fine6_run["aucs"])
            rows.append({
                "config": config,
                "block": block,
                "w_raw": 1.0 - block_weight,
                "w_block": block_weight,
                "best_C": run["best_C"],
                "CV_mean": float(run["aucs"].mean()),
                "CV_std": float(run["aucs"].std()),
                "CV_median": float(np.median(run["aucs"])),
                "delta_vs_raw": raw_stats["delta"],
                "wins_vs_raw": raw_stats["wins"],
                "losses_vs_raw": raw_stats["losses"],
                "delta_vs_fine6": fine_stats["delta"],
                "wins_vs_fine6": fine_stats["wins"],
                "losses_vs_fine6": fine_stats["losses"],
                "mean_fold_delta_vs_fine6": fine_stats["mean_fold_delta"],
                "median_fold_delta_vs_fine6": fine_stats["median_fold_delta"],
                "min_fold_delta_vs_fine6": fine_stats["min_fold_delta"],
                "max_fold_delta_vs_fine6": fine_stats["max_fold_delta"],
                "corr_with_raw": float(spearmanr(
                    run["oof"].ravel(), raw_run["oof"].ravel()).statistic),
                "corr_with_fine6": float(spearmanr(
                    run["oof"].ravel(), fine6_run["oof"].ravel()).statistic),
                "fit_time": float(run["fit_time"]),
            })
    table = pd.DataFrame(rows)
    if table.empty:
        best_per_block = pd.DataFrame()
    else:
        best_indices = table.groupby("block")["CV_mean"].idxmax()
        best_per_block = table.loc[
            best_indices,
            ["block", "w_block", "best_C", "CV_mean", "delta_vs_raw",
             "delta_vs_fine6", "wins_vs_fine6", "losses_vs_fine6",
             "corr_with_fine6"]
        ].sort_values("CV_mean", ascending=False).reset_index(drop=True)
    return table, best_per_block, runs


def check_clean_derived_parity(
        base_cache, clean_cache, quality_cache, fixed_points,
        distance_tolerance=1e-5, auc_tolerance=1e-8, n_jobs=-1):
    """Compare clean sidecar distances with the earlier feature-quality path.

    Only folds whose seed and exact train/valid indices already exist in both
    caches are compared.  This normally reuses the original QUICK 5 folds and
    does not create a split.
    """
    clean_by_key = _clean_derived_by_base_key(clean_cache)
    quality_by_identity = {
        (
            int(fold.seed),
            int(fold.fold),
            np.asarray(fold.train_idx, dtype=np.int32).tobytes(),
            np.asarray(fold.valid_idx, dtype=np.int32).tobytes(),
        ): fold
        for fold in quality_cache.folds
    }

    tasks = []
    for base_fold in base_cache.folds:
        identity = (
            int(base_fold.seed),
            int(base_fold.fold),
            np.asarray(base_fold.train_idx, dtype=np.int32).tobytes(),
            np.asarray(base_fold.valid_idx, dtype=np.int32).tobytes(),
        )
        quality_fold = quality_by_identity.get(identity)
        if quality_fold is None:
            continue
        clean_fold = clean_by_key[base_fold.cache_key]
        tasks.append((base_fold, clean_fold, quality_fold))

    if not tasks:
        raise RuntimeError(
            "no existing folds overlap between clean and feature-quality caches")

    rows = []
    for config, point in fixed_points.items():
        if config not in CLEAN_DERIVED_SPECS:
            raise KeyError(f"parity requested for non-derived clean block {config}")
        if config not in quality_cache.variant_specs:
            raise KeyError(f"feature-quality cache lacks parity config {config}")
        gamma, C = float(point["gamma"]), float(point["C"])
        distance_deltas, auc_deltas = [], []

        for base_fold, clean_fold, quality_fold in tasks:
            clean_Dtr = np.asarray(_load_array(clean_fold.d2_train[config]))
            clean_Dva = np.asarray(_load_array(clean_fold.d2_valid[config]))
            quality_Dtr = np.asarray(_load_array(quality_fold.d2_train[config]))
            quality_Dva = np.asarray(_load_array(quality_fold.d2_valid[config]))
            distance_deltas.append(max(
                float(np.max(np.abs(clean_Dtr - quality_Dtr))),
                float(np.max(np.abs(clean_Dva - quality_Dva))),
            ))

            def score(Dtr, Dva):
                Ktr = np.exp(-gamma * Dtr).astype(np.float32)
                Kva = np.exp(-gamma * Dva).astype(np.float32)
                model = SVC(C=C, kernel="precomputed").fit(
                    Ktr, base_fold.y_train)
                pred = model.decision_function(Kva)
                return roc_auc_score(base_fold.y_valid, pred)

            auc_deltas.append(abs(
                score(clean_Dtr, clean_Dva) - score(quality_Dtr, quality_Dva)))

        max_distance = float(max(distance_deltas))
        max_auc = float(max(auc_deltas))
        if max_distance > distance_tolerance:
            raise RuntimeError(
                f"{config} clean/feature-quality distance parity failed: "
                f"{max_distance:.3g} > {distance_tolerance:.3g}")
        if max_auc > auc_tolerance:
            raise RuntimeError(
                f"{config} clean/feature-quality AUC parity failed: "
                f"{max_auc:.3g} > {auc_tolerance:.3g}")
        rows.append({
            "config": config,
            "matched_folds": len(tasks),
            "gamma": gamma,
            "C": C,
            "max_distance_difference": max_distance,
            "max_auc_difference": max_auc,
        })
    return pd.DataFrame(rows)

def check_clean_derived_transform_parity(
        base_cache, clean_cache, fixed_points, max_folds=5,
        distance_tolerance=1e-6, auc_tolerance=1e-10):
    """Recompute clean transforms from the same cached source features.

    This exercises the earlier feature-quality ``_quality_transform_pair`` path
    on the exact float32 legacy source arrays, so parity can be bit-close without
    rebuilding supervised features or creating CV splits.
    """
    clean_by_key = _clean_derived_by_base_key(clean_cache)
    selected_folds = list(base_cache.folds[:int(max_folds)])
    if not selected_folds:
        raise RuntimeError("base cache is empty")

    rows = []
    for config, point in fixed_points.items():
        if config not in CLEAN_DERIVED_SPECS:
            raise KeyError(f"parity requested for non-derived clean block {config}")
        cfg = CLEAN_DERIVED_SPECS[config]
        gamma, C = float(point["gamma"]), float(point["C"])
        distance_deltas, auc_deltas = [], []

        for base_fold in selected_folds:
            clean_fold = clean_by_key[base_fold.cache_key]
            source = cfg["source"]
            A = np.asarray(_load_array(base_fold.features_train[source]), dtype=float)
            B = np.asarray(_load_array(base_fold.features_valid[source]), dtype=float)
            feature_names = [f"{source}_{i}" for i in range(A.shape[1])]
            builder_seed = (
                int(base_fold.seed) + INNER_SEED_OFFSET + int(base_fold.fold))
            A_q, B_q, _ = _quality_transform_pair(
                A, B, feature_names,
                {"source": source, "transform": "pca_whiten",
                 "n_components": int(cfg["n_components"])},
                builder_seed,
            )
            Dtr_q = cdist(A_q, A_q, metric="sqeuclidean").astype(np.float32)
            Dva_q = cdist(B_q, A_q, metric="sqeuclidean").astype(np.float32)
            Dtr_c = np.asarray(_load_array(clean_fold.d2_train[config]))
            Dva_c = np.asarray(_load_array(clean_fold.d2_valid[config]))
            distance_deltas.append(max(
                float(np.max(np.abs(Dtr_q - Dtr_c))),
                float(np.max(np.abs(Dva_q - Dva_c))),
            ))

            def auc_from_distance(Dtr, Dva):
                Ktr = np.exp(-gamma * Dtr).astype(np.float32)
                Kva = np.exp(-gamma * Dva).astype(np.float32)
                model = SVC(C=C, kernel="precomputed").fit(
                    Ktr, base_fold.y_train)
                pred = model.decision_function(Kva)
                return roc_auc_score(base_fold.y_valid, pred)

            auc_deltas.append(abs(
                auc_from_distance(Dtr_q, Dva_q)
                - auc_from_distance(Dtr_c, Dva_c)))

        max_distance = float(max(distance_deltas))
        max_auc = float(max(auc_deltas))
        if max_distance > float(distance_tolerance):
            raise RuntimeError(
                f"{config} derived-transform distance parity failed: "
                f"{max_distance:.3g} > {distance_tolerance:.3g}")

        # ROC-AUC changes in discrete positive/negative-pair quanta.  Once the
        # distance matrices pass the strict parity check, allow at most one such
        # quantum per checked fold as numerical rank-tie noise.
        auc_quantum = 0.0
        for base_fold in selected_folds:
            yv = np.asarray(base_fold.y_valid)
            n_pos = int(np.sum(yv == 1))
            n_neg = int(np.sum(yv == 0))
            if n_pos and n_neg:
                auc_quantum = max(auc_quantum, 1.0 / (n_pos * n_neg))
        effective_auc_tolerance = max(float(auc_tolerance), auc_quantum + 1e-12)
        if max_auc > effective_auc_tolerance:
            raise RuntimeError(
                f"{config} derived-transform AUC parity failed: "
                f"{max_auc:.3g} > {effective_auc_tolerance:.3g} "
                f"(requested={float(auc_tolerance):.3g}, one-pair quantum={auc_quantum:.3g})")
        rows.append({
            "config": config,
            "checked_folds": len(selected_folds),
            "gamma": gamma,
            "C": C,
            "max_distance_difference": max_distance,
            "max_auc_difference": max_auc,
            "auc_pair_quantum": auc_quantum,
            "effective_auc_tolerance": effective_auc_tolerance,
        })
    return pd.DataFrame(rows)


def load_existing_clean_derived_cache(
        base_cache, cache_dir=".svc_clean_derived_cache_v1", specs=None):
    """Load an already-complete clean sidecar without building any file."""
    specs = dict(CLEAN_DERIVED_SPECS if specs is None else specs)
    root = Path(cache_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"required existing clean cache does not exist: {root}")
    folds = []
    missing = []
    for base_fold in base_cache.folds:
        key, cache_spec = _clean_sidecar_spec(base_fold, specs)
        folder = root / key
        if not _clean_sidecar_complete(folder, cache_spec):
            missing.append({
                "repeat": int(base_fold.repeat),
                "fold": int(base_fold.fold),
                "seed": int(base_fold.seed),
                "path": str(folder),
            })
            continue
        folds.append(_clean_sidecar_from_disk(
            folder, base_fold, key, specs))
    if missing:
        raise RuntimeError(
            "required clean sidecar is incomplete; refusing to build new "
            f"distances. Missing folds: {missing}")
    folds.sort(key=lambda item: (item.repeat, item.fold))
    base_keys = [
        (fold.repeat, fold.fold, fold.seed, fold.cache_key,
         fold.train_idx.tobytes(), fold.valid_idx.tobytes())
        for fold in base_cache.folds
    ]
    clean_keys = [
        (fold.repeat, fold.fold, fold.seed, fold.base_cache_key,
         fold.train_idx.tobytes(), fold.valid_idx.tobytes())
        for fold in folds
    ]
    if clean_keys != base_keys:
        raise RuntimeError("existing clean sidecar fold keys are not exact")
    return CleanDerivedCache(
        folds=folds,
        cache_dir=root,
        cache_hits=len(folds),
        cache_builds=0,
        block_names=tuple(specs),
    )


def evaluate_clean_config_batch(
        cache, clean_cache, configs, gammas, c_grid, n_jobs=-1):
    """Evaluate many additive weights while exponentiating each block once/fold.

    No split, feature, distance, or cache construction occurs here.  For each
    existing fold, fixed component kernels are materialized once and reused by
    every weight configuration and every C value in the batch.
    """
    if not configs:
        raise ValueError("clean config batch is empty")
    c_grid = tuple(float(value) for value in c_grid)
    if not c_grid or not np.isfinite(c_grid).all() or min(c_grid) <= 0:
        raise ValueError("C grid must contain finite positive values")

    normalized = {}
    for name, weights in configs.items():
        total = float(sum(float(value) for value in weights.values()))
        if not np.isclose(total, 1.0, atol=1e-12):
            raise ValueError(f"{name} weights must sum to one, got {total}")
        normalized[name] = normalized_clean_weights(weights)
    active_blocks = tuple(
        block for block in CLEAN_BLOCKS
        if any(weights[block] > 0 for weights in normalized.values()))
    missing_gammas = set(active_blocks) - set(gammas)
    if missing_gammas:
        raise KeyError(f"missing fixed clean gammas: {sorted(missing_gammas)}")

    base_keys = [
        (fold.repeat, fold.fold, fold.seed, fold.cache_key,
         fold.train_idx.tobytes(), fold.valid_idx.tobytes())
        for fold in cache.folds
    ]
    clean_keys = [
        (fold.repeat, fold.fold, fold.seed, fold.base_cache_key,
         fold.train_idx.tobytes(), fold.valid_idx.tobytes())
        for fold in clean_cache.folds
    ]
    if clean_keys != base_keys:
        raise RuntimeError("clean cache does not use exact legacy fold keys")

    def one(base_fold):
        component_train, component_valid = {}, {}
        for block in active_blocks:
            Dtr = get_clean_distance(base_fold, clean_cache, block, train=True)
            Dva = get_clean_distance(base_fold, clean_cache, block, train=False)
            component_train[block] = np.exp(
                -float(gammas[block]) * Dtr).astype(np.float32)
            component_valid[block] = np.exp(
                -float(gammas[block]) * Dva).astype(np.float32)

        predictions = {}
        for name, weights in normalized.items():
            Ktr = None
            Kva = None
            for block in active_blocks:
                weight = weights[block]
                if weight <= 0:
                    continue
                if Ktr is None:
                    Ktr = np.float32(weight) * component_train[block]
                    Kva = np.float32(weight) * component_valid[block]
                else:
                    Ktr += np.float32(weight) * component_train[block]
                    Kva += np.float32(weight) * component_valid[block]
            if Ktr is None or Kva is None:
                raise RuntimeError(f"{name} has no active clean kernel")
            if not np.allclose(np.diag(Ktr), 1.0, atol=2e-5):
                raise RuntimeError(f"{name} clean kernel diagonal is not one")
            for C in c_grid:
                model = SVC(C=C, kernel="precomputed").fit(
                    Ktr, base_fold.y_train)
                score = model.decision_function(Kva)
                predictions[(name, C)] = (
                    roc_auc_score(base_fold.y_valid, score), score)
            del Ktr, Kva
        return base_fold.repeat, base_fold.valid_idx, predictions

    items = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one)(fold) for fold in cache.folds)
    output = {
        name: {
            C: {
                "aucs": [],
                "oof": np.full(
                    (len(cache.seeds), cache.n_samples), np.nan),
            }
            for C in c_grid
        }
        for name in normalized
    }
    for repeat, valid_idx, predictions in items:
        for (name, C), (auc, score) in predictions.items():
            output[name][C]["aucs"].append(auc)
            output[name][C]["oof"][repeat, valid_idx] = score

    runs = {}
    for name, weights in normalized.items():
        for C in c_grid:
            output[name][C]["aucs"] = np.asarray(
                output[name][C]["aucs"], dtype=float)
            if not np.isfinite(output[name][C]["oof"]).all():
                raise RuntimeError(f"incomplete clean OOF predictions for {name}")
        best_C = max(c_grid, key=lambda C: output[name][C]["aucs"].mean())
        runs[name] = {
            "weights": weights,
            "gammas": {
                block: float(gammas[block])
                for block in active_blocks if weights[block] > 0},
            "best_C": float(best_C),
            "aucs": output[name][best_C]["aucs"],
            "oof": output[name][best_C]["oof"],
            "grid": output[name],
        }
    return runs


def clean_additive_result_table(configs, runs, raw_run, fine6_run):
    """Build paired 25-fold diagnostics for additive clean configurations."""
    rows = []
    for name, requested_weights in configs.items():
        run = runs[name]
        raw_stats = _paired_clean_stats(run["aucs"], raw_run["aucs"])
        fine_stats = _paired_clean_stats(run["aucs"], fine6_run["aucs"])
        weights = normalized_clean_weights(requested_weights)
        rows.append({
            "config": name,
            **{f"w_{block}": float(weights[block]) for block in CLEAN_BLOCKS},
            "best_C": float(run["best_C"]),
            "CV_mean": float(run["aucs"].mean()),
            "CV_std": float(run["aucs"].std()),
            "CV_median": float(np.median(run["aucs"])),
            "delta_vs_raw": raw_stats["delta"],
            "wins_vs_raw": raw_stats["wins"],
            "losses_vs_raw": raw_stats["losses"],
            "delta_vs_fine6": fine_stats["delta"],
            "wins_vs_fine6": fine_stats["wins"],
            "losses_vs_fine6": fine_stats["losses"],
            "median_fold_delta_vs_fine6": fine_stats["median_fold_delta"],
            "min_fold_delta_vs_fine6": fine_stats["min_fold_delta"],
            "max_fold_delta_vs_fine6": fine_stats["max_fold_delta"],
            "corr_with_fine6": float(spearmanr(
                run["oof"].ravel(), fine6_run["oof"].ravel()).statistic),
        })
    return pd.DataFrame(rows).sort_values(
        "CV_mean", ascending=False).reset_index(drop=True)
