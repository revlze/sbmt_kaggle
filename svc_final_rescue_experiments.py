"""Final rescue pipeline built strictly on the existing 25-fold SVC cache."""
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
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import svc_multikernel_experiments as legacy


RESCUE_CACHE_VERSION = 1
RESCUE_DERIVED_SPECS = {
    "geometry_white16": {
        "source": "geometry", "transform": "pca_white", "n_components": 16},
    "geometry_family": {
        "source": "geometry", "transform": "geometry_family"},
    "density_white5": {
        "source": "density", "transform": "pca_white", "n_components": 5},
    "density_family": {
        "source": "density", "transform": "density_family"},
    "coord_white4": {
        "source": "coord", "transform": "pca_white", "n_components": 4},
    "aggregates_white4": {
        "source": "aggregates", "transform": "pca_white", "n_components": 4},
}
RESCUE_ORIGINAL_SOURCES = {
    "geometry_orig": "geometry",
    "density_orig": "density",
    "coord_orig": "coord",
    "aggregates_orig": "aggregates",
    "knn_orig": "knn",
    "pca128_orig": "pca128",
    "raw_orig": "raw",
}
RESCUE_REPRESENTATIONS = (
    "geometry_orig", "geometry_white16", "geometry_family",
    "density_orig", "density_white5", "density_family",
    "coord_orig", "coord_white4",
    "aggregates_orig", "aggregates_white4",
    "knn_orig", "pca128_orig", "raw_orig",
)


@dataclass
class RescueDerivedFold:
    repeat: int
    fold: int
    seed: int
    base_cache_key: str
    train_idx: np.ndarray
    valid_idx: np.ndarray
    d2_train: dict[str, Path]
    d2_valid: dict[str, Path]
    gamma0: dict[str, float]
    metadata: dict[str, dict]
    cache_key: str


@dataclass
class RescueDerivedCache:
    folds: list[RescueDerivedFold]
    cache_dir: Path
    cache_hits: int
    cache_builds: int


def load_existing_cv_cache(y, seeds, cache_dir, n_splits, n_bins, alpha,
                           pca_components):
    try:
        cache = legacy.load_existing_distance_cache(
            y, seeds=seeds, cache_dir=cache_dir, n_splits=n_splits,
            n_bins=n_bins, alpha=alpha, pca_components=pca_components)
    except Exception as exc:
        raise RuntimeError(
            "Existing CV cache incomplete. Do not build new folds in rescue notebook."
        ) from exc
    if cache.cache_builds != 0 or len(cache.folds) != len(tuple(seeds)) * n_splits:
        raise RuntimeError(
            "Existing CV cache incomplete. Do not build new folds in rescue notebook.")
    return cache


def _feature_names_for_fold(base_fold, source):
    path = Path(base_fold.features_train[source]).parent / "feature_names.json"
    names = json.loads(path.read_text(encoding="utf-8"))[source]
    return list(names)


def _pca_scaled(A, B, n_components):
    maximum = min(A.shape[1], len(A) - 1)
    k = min(int(n_components), maximum)
    if k <= 0:
        raise ValueError("PCA representation has no valid component")
    pca = PCA(n_components=k, svd_solver="full")
    A_pc = pca.fit_transform(A)
    B_pc = pca.transform(B)
    if np.min(pca.explained_variance_) < 1e-10:
        raise ValueError("retained rescue PCA eigenvalue is below 1e-10")
    scaler = StandardScaler()
    return (
        scaler.fit_transform(A_pc),
        scaler.transform(B_pc),
        {
            "n_components": k,
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()),
            "min_explained_variance": float(pca.explained_variance_.min()),
        },
    )


def _geometry_families(names):
    base_names = {"d_euclid_diff", "proj_w", "cos_w", "lda_score"}
    groups = {name: [] for name in ("base", "mahal", "rda", "mid", "fisher")}
    for index, name in enumerate(names):
        if name in base_names:
            family = "base"
        elif name.startswith(("mahal_", "d_mahal_")):
            family = "mahal"
        elif name.startswith("rda_"):
            family = "rda"
        elif name.startswith("mid_"):
            family = "mid"
        elif name.startswith("fisher_"):
            family = "fisher"
        else:
            raise ValueError(f"unclassified geometry feature: {name}")
        groups[family].append(index)
    if any(not values for values in groups.values()):
        raise ValueError(f"empty geometry family: {groups}")
    return groups


def _family_piece(A, B, indices, pca_components=None):
    A_part, B_part = A[:, indices], B[:, indices]
    if pca_components is None:
        k = A_part.shape[1]
        return A_part / np.sqrt(k), B_part / np.sqrt(k), {"n_components": k}
    A_pc, B_pc, metadata = _pca_scaled(
        A_part, B_part, min(int(pca_components), A_part.shape[1]))
    k = A_pc.shape[1]
    return A_pc / np.sqrt(k), B_pc / np.sqrt(k), metadata


def transform_rescue_representation(A, B, feature_names, spec):
    """Fit a deterministic unsupervised rescue transform on A only."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    transform = spec["transform"]
    metadata = {"source": spec["source"], "transform": transform,
                "input_dim": int(A.shape[1])}
    if transform == "pca_white":
        A_out, B_out, details = _pca_scaled(A, B, spec["n_components"])
        metadata.update(details)
    elif transform == "geometry_family":
        groups = _geometry_families(feature_names)
        pieces = []
        family_meta = {}
        for family, k in (("base", None), ("mahal", 3), ("rda", 3),
                          ("mid", 2), ("fisher", 2)):
            A_piece, B_piece, details = _family_piece(
                A, B, groups[family], k)
            pieces.append((A_piece, B_piece))
            family_meta[family] = {
                "input_dim": len(groups[family]), **details}
        A_out = np.column_stack([item[0] for item in pieces])
        B_out = np.column_stack([item[1] for item in pieces])
        metadata["families"] = family_meta
    elif transform == "density_family":
        base = [i for i, name in enumerate(feature_names)
                if name in {"qda_score", "pca_subspace_diff"}]
        ppca = [i for i, name in enumerate(feature_names)
                if name.startswith("ppca_")]
        classified = set(base) | set(ppca)
        if len(classified) != len(feature_names) or not base or not ppca:
            unknown = [name for i, name in enumerate(feature_names)
                       if i not in classified]
            raise ValueError(f"invalid density families; unknown={unknown}")
        A_base, B_base, base_meta = _family_piece(A, B, base, None)
        A_ppca, B_ppca, ppca_meta = _family_piece(A, B, ppca, 4)
        A_out = np.column_stack([A_base, A_ppca])
        B_out = np.column_stack([B_base, B_ppca])
        metadata["families"] = {
            "density_base": {"input_dim": len(base), **base_meta},
            "ppca": {"input_dim": len(ppca), **ppca_meta},
        }
    else:
        raise ValueError(f"unsupported rescue transform: {transform}")
    if not np.isfinite(A_out).all() or not np.isfinite(B_out).all():
        raise RuntimeError("non-finite rescue representation")
    metadata["output_dim"] = int(A_out.shape[1])
    return np.asarray(A_out, np.float32), np.asarray(B_out, np.float32), metadata


def _sidecar_spec(base_fold):
    spec = {
        "version": RESCUE_CACHE_VERSION,
        "base_cache_key": str(base_fold.cache_key),
        "repeat": int(base_fold.repeat), "fold": int(base_fold.fold),
        "seed": int(base_fold.seed), "variants": RESCUE_DERIVED_SPECS,
        "pca_solver": "full",
    }
    key = sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    return key, spec


def _sidecar_complete(folder, spec):
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (manifest.get("spec") == spec
            and all((folder / name).exists()
                    for name in manifest.get("files", [])))


def _sidecar_from_disk(folder, base_fold, key):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    names = tuple(RESCUE_DERIVED_SPECS)
    return RescueDerivedFold(
        base_fold.repeat, base_fold.fold, base_fold.seed, base_fold.cache_key,
        base_fold.train_idx, base_fold.valid_idx,
        {name: folder / f"d2_train_{name}.npy" for name in names},
        {name: folder / f"d2_valid_{name}.npy" for name in names},
        {name: float(value) for name, value in manifest["gamma0"].items()},
        manifest["metadata"], key)


def _build_sidecar_fold(base_fold, folder, key, cache_spec):
    print(f"build rescue derived: repeat={base_fold.repeat + 1}, "
          f"fold={base_fold.fold + 1}")
    folder.mkdir(parents=True, exist_ok=True)
    gamma0, metadata, files = {}, {}, []
    for name, spec in RESCUE_DERIVED_SPECS.items():
        source = spec["source"]
        A = np.asarray(legacy._load_array(base_fold.features_train[source]), float)
        B = np.asarray(legacy._load_array(base_fold.features_valid[source]), float)
        names = _feature_names_for_fold(base_fold, source)
        A_out, B_out, details = transform_rescue_representation(A, B, names, spec)
        Dtr = cdist(A_out, A_out, "sqeuclidean").astype(np.float32)
        Dva = cdist(B_out, A_out, "sqeuclidean").astype(np.float32)
        gamma0[name], _ = legacy._median_gamma_from_distance(Dtr, base_fold.seed)
        train_name, valid_name = f"d2_train_{name}.npy", f"d2_valid_{name}.npy"
        np.save(folder / train_name, Dtr, allow_pickle=False)
        np.save(folder / valid_name, Dva, allow_pickle=False)
        files.extend([train_name, valid_name])
        metadata[name] = details
        del A, B, A_out, B_out, Dtr, Dva
    manifest = {"spec": cache_spec, "gamma0": gamma0,
                "metadata": metadata, "files": sorted(files)}
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return _sidecar_from_disk(folder, base_fold, key)


def build_or_load_rescue_cache(base_cache, cache_dir, rebuild=False, n_jobs=1):
    """Build only deterministic derived distances from cached fold features."""
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    loaded, jobs = [], []
    for base_fold in base_cache.folds:
        key, spec = _sidecar_spec(base_fold)
        folder = root / key
        if not rebuild and _sidecar_complete(folder, spec):
            loaded.append(_sidecar_from_disk(folder, base_fold, key))
        else:
            jobs.append((base_fold, folder, key, spec))
    built = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_build_sidecar_fold)(*job) for job in jobs)
    folds = sorted(loaded + built, key=lambda fold: (fold.repeat, fold.fold))
    if len(folds) != len(base_cache.folds):
        raise RuntimeError("rescue sidecar fold count mismatch")
    for base, derived in zip(base_cache.folds, folds):
        if (base.repeat, base.fold, base.seed, base.cache_key,
                base.train_idx.tobytes(), base.valid_idx.tobytes()) != (
                derived.repeat, derived.fold, derived.seed,
                derived.base_cache_key, derived.train_idx.tobytes(),
                derived.valid_idx.tobytes()):
            raise RuntimeError("rescue cache does not match exact legacy folds")
    return RescueDerivedCache(folds, root, len(loaded), len(built))


def _derived_by_key(rescue_cache):
    return {fold.base_cache_key: fold for fold in rescue_cache.folds}


def get_representation_distance(base_fold, rescue_cache, representation, train=True):
    if representation in RESCUE_ORIGINAL_SOURCES:
        source = base_fold.d2_train if train else base_fold.d2_valid
        return legacy._load_array(source[RESCUE_ORIGINAL_SOURCES[representation]])
    if representation not in RESCUE_DERIVED_SPECS:
        raise KeyError(f"unknown rescue representation: {representation}")
    derived = _derived_by_key(rescue_cache)[base_fold.cache_key]
    source = derived.d2_train if train else derived.d2_valid
    return legacy._load_array(source[representation])


def representation_gamma0(base_fold, rescue_cache, representation):
    if representation in RESCUE_ORIGINAL_SOURCES:
        return float(base_fold.gamma0[RESCUE_ORIGINAL_SOURCES[representation]])
    return float(_derived_by_key(rescue_cache)[base_fold.cache_key].gamma0[representation])


def representation_dim(base_fold, rescue_cache, representation):
    if representation in RESCUE_ORIGINAL_SOURCES:
        source = RESCUE_ORIGINAL_SOURCES[representation]
        return int(np.asarray(legacy._load_array(
            base_fold.features_train[source])).shape[1])
    return int(_derived_by_key(rescue_cache)[base_fold.cache_key].metadata[
        representation]["output_dim"])


def _evaluate_single_on_folds(folds, representation, rescue_cache, gamma, c_grid,
                              n_jobs=-1):
    c_grid = tuple(float(C) for C in c_grid)

    def one(fold):
        Dtr = get_representation_distance(fold, rescue_cache, representation, True)
        Dva = get_representation_distance(fold, rescue_cache, representation, False)
        Ktr = np.exp(-float(gamma) * Dtr).astype(np.float32)
        Kva = np.exp(-float(gamma) * Dva).astype(np.float32)
        result = []
        for C in c_grid:
            model = SVC(C=C, kernel="precomputed").fit(Ktr, fold.y_train)
            score = model.decision_function(Kva)
            result.append((roc_auc_score(fold.y_valid, score), score))
        return fold.repeat, fold.valid_idx, result

    items = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one)(fold) for fold in folds)
    output = {C: {"aucs": [], "items": []} for C in c_grid}
    for repeat, valid_idx, predictions in items:
        for C, (auc, score) in zip(c_grid, predictions):
            output[C]["aucs"].append(auc)
            output[C]["items"].append((repeat, valid_idx, score))
    for C in c_grid:
        output[C]["aucs"] = np.asarray(output[C]["aucs"], float)
    return output


def _oof_from_items(cache, items):
    oof = np.full((len(cache.seeds), cache.n_samples), np.nan)
    for repeat, valid_idx, score in items:
        oof[repeat, valid_idx] = score
    if not np.isfinite(oof).all():
        raise RuntimeError("incomplete rescue OOF")
    return oof


def evaluate_fixed_representation(cache, rescue_cache, representation, gamma, C,
                                  n_jobs=-1):
    """Evaluate one already chosen representation on the existing cached folds."""
    C = float(C)
    result = _evaluate_single_on_folds(
        cache.folds, representation, rescue_cache, float(gamma), [C], n_jobs)[C]
    return {
        "representation": representation,
        "gamma": float(gamma),
        "C": C,
        "aucs": result["aucs"],
        "oof": _oof_from_items(cache, result["items"]),
    }


def calibrate_representations(cache, rescue_cache, gamma_multipliers, c_grid,
                              n_jobs=-1):
    quick_folds = [fold for fold in cache.folds if fold.repeat == 0]
    if len(quick_folds) != 5 or quick_folds != cache.folds[:5]:
        raise RuntimeError("first existing repeat must contain exactly five folds")
    runs = {}
    for representation in RESCUE_REPRESENTATIONS:
        base_gamma = float(np.median([
            representation_gamma0(fold, rescue_cache, representation)
            for fold in quick_folds]))
        best = None
        for multiplier in gamma_multipliers:
            gamma = base_gamma * float(multiplier)
            grid = _evaluate_single_on_folds(
                quick_folds, representation, rescue_cache, gamma, c_grid, n_jobs)
            for C, result in grid.items():
                candidate = (result["aucs"].mean(), -float(multiplier), -C,
                             gamma, C)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        quick_mean, _, _, gamma, C = best
        full = _evaluate_single_on_folds(
            cache.folds, representation, rescue_cache, gamma, [C], n_jobs)[float(C)]
        runs[representation] = {
            "representation": representation,
            "source_block": (RESCUE_ORIGINAL_SOURCES.get(representation)
                             or RESCUE_DERIVED_SPECS[representation]["source"]),
            "dim": representation_dim(cache.folds[0], rescue_cache, representation),
            "gamma": float(gamma), "C": float(C),
            "quick_CV": float(quick_mean), "aucs": full["aucs"],
            "oof": _oof_from_items(cache, full["items"]),
        }

    original_for = {
        "geometry_orig": "geometry_orig", "geometry_white16": "geometry_orig",
        "geometry_family": "geometry_orig", "density_orig": "density_orig",
        "density_white5": "density_orig", "density_family": "density_orig",
        "coord_orig": "coord_orig", "coord_white4": "coord_orig",
        "aggregates_orig": "aggregates_orig",
        "aggregates_white4": "aggregates_orig", "knn_orig": "knn_orig",
        "pca128_orig": "pca128_orig", "raw_orig": "raw_orig",
    }
    rows = []
    for representation in RESCUE_REPRESENTATIONS:
        run = runs[representation]
        original = runs[original_for[representation]]
        delta = run["aucs"] - original["aucs"]
        rows.append({
            "representation": representation,
            "source_block": run["source_block"], "dim": run["dim"],
            "gamma": run["gamma"], "C": run["C"],
            "quick_CV": run["quick_CV"],
            "full25_CV": float(run["aucs"].mean()),
            "full25_std": float(run["aucs"].std()),
            "delta_vs_orig": float(delta.mean()),
            "wins_vs_orig": int((delta > 0).sum()),
            "losses_vs_orig": int((delta < 0).sum()),
        })
    return pd.DataFrame(rows), runs


def evaluate_additive_candidates(cache, rescue_cache, block_representations,
                                 gammas, configs, c_grid, n_jobs=-1):
    c_grid = tuple(float(C) for C in c_grid)
    normalized = {}
    for name, weights in configs.items():
        unknown = set(weights) - set(block_representations)
        if unknown:
            raise KeyError(f"{name} unknown blocks: {sorted(unknown)}")
        values = {block: float(weights.get(block, 0.0))
                  for block in block_representations}
        if any(value < 0 for value in values.values()) or not np.isclose(
                sum(values.values()), 1.0, atol=1e-12):
            raise ValueError(f"{name} weights must be nonnegative and sum to one")
        normalized[name] = values
    active_blocks = tuple(block for block in block_representations
                          if any(w[block] > 0 for w in normalized.values()))

    def one(fold):
        Ktr_parts, Kva_parts = {}, {}
        for block in active_blocks:
            representation = block_representations[block]
            Dtr = get_representation_distance(fold, rescue_cache, representation, True)
            Dva = get_representation_distance(fold, rescue_cache, representation, False)
            Ktr_parts[block] = np.exp(-gammas[block] * Dtr).astype(np.float32)
            Kva_parts[block] = np.exp(-gammas[block] * Dva).astype(np.float32)
        predictions = {}
        for name, weights in normalized.items():
            Ktr = sum(np.float32(weights[b]) * Ktr_parts[b]
                      for b in active_blocks if weights[b] > 0)
            Kva = sum(np.float32(weights[b]) * Kva_parts[b]
                      for b in active_blocks if weights[b] > 0)
            for C in c_grid:
                model = SVC(C=C, kernel="precomputed").fit(Ktr, fold.y_train)
                score = model.decision_function(Kva)
                predictions[(name, C)] = (
                    roc_auc_score(fold.y_valid, score), score)
        return fold.repeat, fold.valid_idx, predictions

    items = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one)(fold) for fold in cache.folds)
    grids = {name: {C: {"aucs": [], "items": []} for C in c_grid}
             for name in normalized}
    for repeat, valid_idx, predictions in items:
        for (name, C), (auc, score) in predictions.items():
            grids[name][C]["aucs"].append(auc)
            grids[name][C]["items"].append((repeat, valid_idx, score))
    runs = {}
    for name in normalized:
        for C in c_grid:
            grids[name][C]["aucs"] = np.asarray(grids[name][C]["aucs"], float)
        best_C = max(c_grid, key=lambda C: grids[name][C]["aucs"].mean())
        best = grids[name][best_C]
        runs[name] = {
            "weights": normalized[name], "best_C": best_C,
            "aucs": best["aucs"], "oof": _oof_from_items(cache, best["items"]),
            "grid": grids[name],
        }
    return runs


def candidate_result_table(configs, runs, fine6_run):
    rows = []
    for name in configs:
        run = runs[name]
        delta = run["aucs"] - fine6_run["aucs"]
        rows.append({
            "config": name, "weights": dict(configs[name]),
            "best_C": run["best_C"], "CV_mean": run["aucs"].mean(),
            "CV_std": run["aucs"].std(), "CV_median": np.median(run["aucs"]),
            "delta_vs_legacy_fine6": delta.mean(),
            "wins_vs_legacy_fine6": int((delta > 0).sum()),
            "losses_vs_legacy_fine6": int((delta < 0).sum()),
            "median_fold_delta": np.median(delta), "min_fold_delta": delta.min(),
            "max_fold_delta": delta.max(),
            "corr_with_fine6": spearmanr(
                run["oof"].ravel(), fine6_run["oof"].ravel()).statistic,
        })
    return pd.DataFrame(rows).sort_values("CV_mean", ascending=False).reset_index(drop=True)


def select_best_kernel(table, configs, tolerance=0.00015):
    best_mean = float(table["CV_mean"].max())
    eligible = table[table["CV_mean"] >= best_mean - tolerance].copy()
    eligible["raw_weight"] = eligible["config"].map(
        lambda name: float(configs[name].get("raw", 0.0)))
    eligible["active_blocks"] = eligible["config"].map(
        lambda name: sum(value > 0 for value in configs[name].values()))
    eligible = eligible.sort_values(
        ["raw_weight", "active_blocks", "CV_mean"],
        ascending=[False, True, False])
    return str(eligible.iloc[0]["config"]), eligible.reset_index(drop=True)


def blend_result_table(cache, blend_runs, fine6_run):
    rows = []
    for name, run in blend_runs.items():
        delta = run["aucs"] - fine6_run["aucs"]
        rows.append({
            "blend": name, "CV_mean": run["aucs"].mean(),
            "CV_std": run["aucs"].std(), "CV_median": np.median(run["aucs"]),
            "delta_vs_fine6": delta.mean(),
            "wins_vs_fine6": int((delta > 0).sum()),
            "losses_vs_fine6": int((delta < 0).sum()),
            "corr_with_fine6": spearmanr(
                run["oof"].ravel(), fine6_run["oof"].ravel()).statistic,
            "median_fold_delta": np.median(delta),
            "min_fold_delta": delta.min(), "max_fold_delta": delta.max(),
        })
    return pd.DataFrame(rows).sort_values("CV_mean", ascending=False).reset_index(drop=True)


def select_diverse_blends(blend_table, blend_configs, minimum_l1=0.08):
    ordered = blend_table["blend"].tolist()
    first = ordered[0]
    names = sorted({key for cfg in blend_configs.values() for key in cfg})
    first_vector = np.asarray([blend_configs[first].get(name, 0.0) for name in names])
    second = None
    for candidate in ordered[1:]:
        vector = np.asarray([blend_configs[candidate].get(name, 0.0) for name in names])
        if np.abs(vector - first_vector).sum() >= minimum_l1:
            second = candidate
            break
    return first, second or ordered[1]


def build_external_views(X_train, y_train, X_external, builder_seed, n_bins,
                         alpha, pca_components):
    """One leakage-safe full-train feature build plus all frozen rescue views."""
    base_train, base_external, artifacts = legacy.make_feature_blocks(
        X_train, y_train, X_external, builder_seed,
        n_bins=n_bins, alpha=alpha, pca_components=pca_components)
    train_views = {name: np.asarray(base_train[source], np.float32)
                   for name, source in RESCUE_ORIGINAL_SOURCES.items()}
    external_views = {name: np.asarray(base_external[source], np.float32)
                      for name, source in RESCUE_ORIGINAL_SOURCES.items()}
    for name, spec in RESCUE_DERIVED_SPECS.items():
        source = spec["source"]
        A, B, _ = transform_rescue_representation(
            base_train[source], base_external[source],
            artifacts["feature_names"][source], spec)
        train_views[name], external_views[name] = A, B
    return {
        "base_train": base_train, "base_external": base_external,
        "train_views": train_views, "external_views": external_views,
    }


def _additive_kernel_from_views(train_views, external_views, representations,
                                gammas, weights):
    Ktr = None
    Kext = None
    for block, weight in weights.items():
        if weight <= 0:
            continue
        representation = representations[block]
        A, B = train_views[representation], external_views[representation]
        Dtr = cdist(A, A, "sqeuclidean")
        Dext = cdist(B, A, "sqeuclidean")
        part_train = np.exp(-gammas[block] * Dtr).astype(np.float32)
        part_external = np.exp(-gammas[block] * Dext).astype(np.float32)
        if Ktr is None:
            Ktr = np.float32(weight) * part_train
            Kext = np.float32(weight) * part_external
        else:
            Ktr += np.float32(weight) * part_train
            Kext += np.float32(weight) * part_external
    return Ktr, Kext


def fit_external_component_scores(
        bundle, y_train, final_representations, final_gammas,
        clean_weights, clean_C, aux_weights, aux_C,
        legacy_weights, legacy_gammas, legacy_C):
    scores = {}
    # Historical Fine6 uses original legacy feature views.
    legacy_representations = {
        block: f"{block}_orig" for block in
        ("raw", "geometry", "density", "aggregates", "knn", "coord", "pca128")}
    Ktr, Kext = _additive_kernel_from_views(
        bundle["train_views"], bundle["external_views"],
        legacy_representations, legacy_gammas, legacy_weights)
    scores["legacy_fine6"] = SVC(
        C=legacy_C, kernel="precomputed").fit(Ktr, y_train).decision_function(Kext)

    Ktr, Kext = _additive_kernel_from_views(
        bundle["train_views"], bundle["external_views"],
        final_representations, final_gammas, clean_weights)
    scores["best_clean_kernel"] = SVC(
        C=clean_C, kernel="precomputed").fit(Ktr, y_train).decision_function(Kext)

    Ktr, Kext = _additive_kernel_from_views(
        bundle["train_views"], bundle["external_views"],
        final_representations, final_gammas, aux_weights)
    scores["clean_aux"] = SVC(
        C=aux_C, kernel="precomputed").fit(Ktr, y_train).decision_function(Kext)
    return scores


def external_rank_blend(component_scores, weights):
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0, atol=1e-12):
        raise ValueError("external blend weights must sum to one")
    return sum(float(weight) * legacy.percentile_rank(component_scores[name])
               for name, weight in weights.items())
