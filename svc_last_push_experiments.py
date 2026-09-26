"""Last-push experiments for week-03-hidden-pairs.

This module deliberately reuses ONLY the already-existing outer CV folds/cache.
It never creates new CV seeds/folds and never touches validation/test labels.

Experiments:
1) sharp-gamma search for the active clean kernel blocks;
2) leakage-safe sqrt(X) raw view on the exact existing folds;
3) additive log-raw + sqrt-raw multi-kernel candidates.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import svc_multikernel_experiments as legacy
import svc_final_rescue_experiments as rescue


SQRT_CACHE_VERSION = 1


def _load(path_or_array):
    return legacy._load_array(path_or_array)


def _oof_from_items(cache, items):
    oof = np.full((len(cache.seeds), cache.n_samples), np.nan, dtype=float)
    for repeat, valid_idx, score in items:
        oof[int(repeat), np.asarray(valid_idx)] = np.asarray(score, dtype=float)
    if not np.isfinite(oof).all():
        raise RuntimeError("Incomplete OOF predictions.")
    return oof


def sharp_gamma_search(
    cache,
    rescue_cache,
    block_representations,
    gamma_multipliers=(1.0, 2.0, 4.0, 8.0),
    c_grid=(3.0, 6.0, 10.0),
    n_jobs=-1,
    old_full25=None,
):
    """Search sharper RBF gammas using only repeat-0 for selection, then 25-fold check."""
    quick_folds = [fold for fold in cache.folds if fold.repeat == 0]
    if len(quick_folds) != 5 or quick_folds != cache.folds[:5]:
        raise RuntimeError("Expected the first existing repeat to contain exactly five folds.")

    rows = []
    runs = {}

    for block, representation in block_representations.items():
        base_gamma = float(np.median([
            rescue.representation_gamma0(fold, rescue_cache, representation)
            for fold in quick_folds
        ]))

        best = None
        for multiplier in gamma_multipliers:
            gamma = base_gamma * float(multiplier)
            grid = rescue._evaluate_single_on_folds(
                quick_folds,
                representation,
                rescue_cache,
                gamma,
                c_grid,
                n_jobs,
            )
            for C, result in grid.items():
                quick_cv = float(np.mean(result["aucs"]))
                # Deterministic tie break: prefer multiplier near 4, then smaller C.
                candidate = (
                    quick_cv,
                    -abs(np.log2(float(multiplier) / 4.0)),
                    -float(C),
                    float(gamma),
                    float(C),
                    float(multiplier),
                )
                if best is None or candidate[:3] > best[:3]:
                    best = candidate

        quick_cv, _, _, gamma, C, multiplier = best
        full = rescue.evaluate_fixed_representation(
            cache, rescue_cache, representation, gamma, C, n_jobs
        )
        full_cv = float(np.mean(full["aucs"]))
        run = {
            **full,
            "block": block,
            "base_gamma": base_gamma,
            "multiplier": multiplier,
            "quick5_CV": quick_cv,
        }
        runs[block] = run

        row = {
            "block": block,
            "representation": representation,
            "base_gamma": base_gamma,
            "chosen_multiplier": multiplier,
            "gamma": gamma,
            "C": C,
            "quick5_CV": quick_cv,
            "sharp_full25_CV": full_cv,
            "sharp_full25_std": float(np.std(full["aucs"])),
        }
        if old_full25 is not None and block in old_full25:
            row["old_full25_CV"] = float(old_full25[block])
            row["delta_vs_old"] = full_cv - float(old_full25[block])
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("sharp_full25_CV", ascending=False)
    return table.reset_index(drop=True), runs


def _sqrt_sidecar_spec(base_fold):
    spec = {
        "version": SQRT_CACHE_VERSION,
        "base_cache_key": str(base_fold.cache_key),
        "repeat": int(base_fold.repeat),
        "fold": int(base_fold.fold),
        "seed": int(base_fold.seed),
        "transform": "sqrt_then_foldwise_standard_scaler",
    }
    key = sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    return key, spec


def _sqrt_sidecar_complete(folder, spec):
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("spec") == spec
        and (folder / "d2_train.npy").exists()
        and (folder / "d2_valid.npy").exists()
        and np.isfinite(float(manifest.get("gamma0", np.nan)))
    )


def _sqrt_record_from_disk(folder, base_fold, key):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return {
        "repeat": int(base_fold.repeat),
        "fold": int(base_fold.fold),
        "seed": int(base_fold.seed),
        "base_cache_key": str(base_fold.cache_key),
        "cache_key": key,
        "d2_train": folder / "d2_train.npy",
        "d2_valid": folder / "d2_valid.npy",
        "gamma0": float(manifest["gamma0"]),
    }


def _build_sqrt_fold(X, base_fold, folder, key, spec):
    folder.mkdir(parents=True, exist_ok=True)

    train_idx = np.asarray(base_fold.train_idx)
    valid_idx = np.asarray(base_fold.valid_idx)

    A0 = np.asarray(X[train_idx], dtype=np.float64)
    B0 = np.asarray(X[valid_idx], dtype=np.float64)
    if np.min(A0) < 0 or np.min(B0) < 0:
        raise ValueError("sqrt_raw requires non-negative original features.")

    scaler = StandardScaler()
    A = scaler.fit_transform(np.sqrt(A0))
    B = scaler.transform(np.sqrt(B0))

    Dtr = cdist(A, A, metric="sqeuclidean").astype(np.float32)
    Dva = cdist(B, A, metric="sqeuclidean").astype(np.float32)
    gamma0, median_distance = legacy._median_gamma_from_distance(Dtr, base_fold.seed)

    np.save(folder / "d2_train.npy", Dtr, allow_pickle=False)
    np.save(folder / "d2_valid.npy", Dva, allow_pickle=False)
    manifest = {
        "spec": spec,
        "gamma0": float(gamma0),
        "median_sq_distance": float(median_distance),
        "train_shape": list(A.shape),
        "valid_shape": list(B.shape),
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return _sqrt_record_from_disk(folder, base_fold, key)


def build_or_load_sqrt_cache(X, base_cache, cache_dir, rebuild=False, n_jobs=1):
    """Build sqrt_raw distances on the exact existing folds; no split generation."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) != base_cache.n_samples:
        raise ValueError("X shape does not match the existing CV cache.")
    if np.min(X) < 0:
        raise ValueError("sqrt_raw requires non-negative original features.")

    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    loaded = []
    jobs = []
    for base_fold in base_cache.folds:
        key, spec = _sqrt_sidecar_spec(base_fold)
        folder = root / key
        if not rebuild and _sqrt_sidecar_complete(folder, spec):
            loaded.append(_sqrt_record_from_disk(folder, base_fold, key))
        else:
            jobs.append((base_fold, folder, key, spec))

    built = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_build_sqrt_fold)(X, *job) for job in jobs
    )

    records = sorted(
        loaded + built, key=lambda r: (int(r["repeat"]), int(r["fold"]))
    )
    if len(records) != len(base_cache.folds):
        raise RuntimeError("sqrt cache fold count mismatch.")

    by_base_key = {r["base_cache_key"]: r for r in records}
    if len(by_base_key) != len(records):
        raise RuntimeError("sqrt cache has duplicate base cache keys.")

    for base_fold in base_cache.folds:
        if str(base_fold.cache_key) not in by_base_key:
            raise RuntimeError("sqrt cache is not aligned with the exact legacy folds.")

    return {
        "records": records,
        "by_base_key": by_base_key,
        "cache_dir": root,
        "cache_hits": len(loaded),
        "cache_builds": len(built),
    }


def _sqrt_record(sqrt_cache, fold):
    return sqrt_cache["by_base_key"][str(fold.cache_key)]


def _evaluate_sqrt_on_folds(folds, sqrt_cache, gamma, c_grid, n_jobs=-1):
    c_grid = tuple(float(C) for C in c_grid)

    def one(fold):
        record = _sqrt_record(sqrt_cache, fold)
        Dtr = _load(record["d2_train"])
        Dva = _load(record["d2_valid"])
        Ktr = np.exp(-float(gamma) * Dtr).astype(np.float32)
        Kva = np.exp(-float(gamma) * Dva).astype(np.float32)
        out = []
        for C in c_grid:
            model = SVC(C=C, kernel="precomputed").fit(Ktr, fold.y_train)
            score = model.decision_function(Kva)
            out.append((roc_auc_score(fold.y_valid, score), score))
        return fold.repeat, fold.valid_idx, out

    items = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one)(fold) for fold in folds
    )
    result = {C: {"aucs": [], "items": []} for C in c_grid}
    for repeat, valid_idx, predictions in items:
        for C, (auc, score) in zip(c_grid, predictions):
            result[C]["aucs"].append(float(auc))
            result[C]["items"].append((repeat, valid_idx, score))
    for C in c_grid:
        result[C]["aucs"] = np.asarray(result[C]["aucs"], dtype=float)
    return result


def calibrate_sqrt_raw(
    cache,
    sqrt_cache,
    gamma_multipliers=(1.0, 2.0, 4.0, 8.0),
    c_grid=(1.0, 3.0, 6.0, 10.0),
    n_jobs=-1,
):
    """Select sqrt_raw gamma/C on existing repeat-0, then report full 25-fold CV."""
    quick_folds = [fold for fold in cache.folds if fold.repeat == 0]
    if len(quick_folds) != 5 or quick_folds != cache.folds[:5]:
        raise RuntimeError("Expected the first existing repeat to contain exactly five folds.")

    base_gamma = float(np.median([
        _sqrt_record(sqrt_cache, fold)["gamma0"] for fold in quick_folds
    ]))

    best = None
    for multiplier in gamma_multipliers:
        gamma = base_gamma * float(multiplier)
        grid = _evaluate_sqrt_on_folds(
            quick_folds, sqrt_cache, gamma, c_grid, n_jobs
        )
        for C, result in grid.items():
            quick_cv = float(np.mean(result["aucs"]))
            candidate = (
                quick_cv,
                -abs(np.log2(float(multiplier) / 4.0)),
                -float(C),
                float(gamma),
                float(C),
                float(multiplier),
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate

    quick_cv, _, _, gamma, C, multiplier = best
    full = _evaluate_sqrt_on_folds(cache.folds, sqrt_cache, gamma, [C], n_jobs)[C]
    run = {
        "representation": "sqrt_raw",
        "base_gamma": base_gamma,
        "multiplier": multiplier,
        "gamma": gamma,
        "C": C,
        "quick5_CV": quick_cv,
        "aucs": full["aucs"],
        "oof": _oof_from_items(cache, full["items"]),
    }
    return run


def evaluate_log_sqrt_configs(
    cache,
    rescue_cache,
    sqrt_cache,
    block_representations,
    gammas,
    sqrt_gamma,
    configs,
    c_grid=(3.0, 6.0),
    n_jobs=-1,
):
    """Evaluate additive kernels containing both log-raw and sqrt-raw views."""
    c_grid = tuple(float(C) for C in c_grid)
    allowed = {"raw_log", "raw_sqrt", *block_representations.keys()}

    normalized = {}
    for name, cfg in configs.items():
        unknown = set(cfg) - allowed
        if unknown:
            raise KeyError(f"{name}: unknown blocks {sorted(unknown)}")
        values = {key: float(cfg.get(key, 0.0)) for key in allowed}
        if any(v < 0 for v in values.values()):
            raise ValueError(f"{name}: negative kernel weight.")
        if not np.isclose(sum(values.values()), 1.0, atol=1e-12):
            raise ValueError(f"{name}: weights must sum to one.")
        normalized[name] = values

    active_regular = [
        block for block in block_representations
        if any(cfg.get(block, 0.0) > 0 for cfg in normalized.values())
    ]
    need_log = any(cfg.get("raw_log", 0.0) > 0 for cfg in normalized.values())
    need_sqrt = any(cfg.get("raw_sqrt", 0.0) > 0 for cfg in normalized.values())

    def one(fold):
        Ktr_parts = {}
        Kva_parts = {}

        if need_log:
            Dtr = rescue.get_representation_distance(
                fold, rescue_cache, block_representations["raw"], True
            )
            Dva = rescue.get_representation_distance(
                fold, rescue_cache, block_representations["raw"], False
            )
            Ktr_parts["raw_log"] = np.exp(-float(gammas["raw"]) * Dtr).astype(np.float32)
            Kva_parts["raw_log"] = np.exp(-float(gammas["raw"]) * Dva).astype(np.float32)

        if need_sqrt:
            record = _sqrt_record(sqrt_cache, fold)
            Dtr = _load(record["d2_train"])
            Dva = _load(record["d2_valid"])
            Ktr_parts["raw_sqrt"] = np.exp(-float(sqrt_gamma) * Dtr).astype(np.float32)
            Kva_parts["raw_sqrt"] = np.exp(-float(sqrt_gamma) * Dva).astype(np.float32)

        for block in active_regular:
            if block == "raw":
                continue
            representation = block_representations[block]
            Dtr = rescue.get_representation_distance(
                fold, rescue_cache, representation, True
            )
            Dva = rescue.get_representation_distance(
                fold, rescue_cache, representation, False
            )
            Ktr_parts[block] = np.exp(-float(gammas[block]) * Dtr).astype(np.float32)
            Kva_parts[block] = np.exp(-float(gammas[block]) * Dva).astype(np.float32)

        predictions = {}
        for name, cfg in normalized.items():
            Ktr = None
            Kva = None
            for block, weight in cfg.items():
                if weight <= 0:
                    continue
                # "raw" is only a representation-map key; log raw is exposed as raw_log.
                if block == "raw":
                    continue
                part_tr = Ktr_parts[block]
                part_va = Kva_parts[block]
                if Ktr is None:
                    Ktr = np.float32(weight) * part_tr
                    Kva = np.float32(weight) * part_va
                else:
                    Ktr += np.float32(weight) * part_tr
                    Kva += np.float32(weight) * part_va

            if Ktr is None:
                raise RuntimeError(f"{name}: empty kernel.")

            for C in c_grid:
                model = SVC(C=C, kernel="precomputed").fit(Ktr, fold.y_train)
                score = model.decision_function(Kva)
                predictions[(name, C)] = (
                    roc_auc_score(fold.y_valid, score), score
                )

        return fold.repeat, fold.valid_idx, predictions

    items = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one)(fold) for fold in cache.folds
    )

    grids = {
        name: {C: {"aucs": [], "items": []} for C in c_grid}
        for name in normalized
    }
    for repeat, valid_idx, predictions in items:
        for (name, C), (auc, score) in predictions.items():
            grids[name][C]["aucs"].append(float(auc))
            grids[name][C]["items"].append((repeat, valid_idx, score))

    runs = {}
    for name in normalized:
        for C in c_grid:
            grids[name][C]["aucs"] = np.asarray(grids[name][C]["aucs"], dtype=float)
        best_C = max(c_grid, key=lambda C: float(np.mean(grids[name][C]["aucs"])))
        best = grids[name][best_C]
        runs[name] = {
            "weights": {k: v for k, v in normalized[name].items() if v > 0},
            "best_C": float(best_C),
            "aucs": best["aucs"],
            "oof": _oof_from_items(cache, best["items"]),
            "grid": grids[name],
        }
    return runs


def model_table(runs, references=None):
    references = references or {}
    rows = []
    for name, run in runs.items():
        row = {
            "model": name,
            "best_C": float(run.get("best_C", run.get("C", np.nan))),
            "CV_mean": float(np.mean(run["aucs"])),
            "CV_std": float(np.std(run["aucs"])),
            "CV_median": float(np.median(run["aucs"])),
        }
        for ref_name, ref_run in references.items():
            delta = np.asarray(run["aucs"]) - np.asarray(ref_run["aucs"])
            row[f"delta_vs_{ref_name}"] = float(np.mean(delta))
            row[f"wins_vs_{ref_name}"] = int(np.sum(delta > 0))
            row[f"losses_vs_{ref_name}"] = int(np.sum(delta < 0))
            if "oof" in run and "oof" in ref_run:
                row[f"corr_{ref_name}"] = float(
                    spearmanr(run["oof"].ravel(), ref_run["oof"].ravel()).statistic
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("CV_mean", ascending=False).reset_index(drop=True)
