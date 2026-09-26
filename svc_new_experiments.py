"""Leakage-safe repeated-CV experiments for ``svc_new.ipynb``.

The module keeps supervised 512-D evidence train representations inner-OOF,
caches pairwise matrices before Optuna, and never touches external validation
except in explicit post-selection evaluation functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wilcoxon
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


EPS = 1e-10


def _array(X):
    return np.asarray(X, dtype=np.float64)


def _safe_auc(y, score):
    score = np.asarray(score)
    return roc_auc_score(y, score) if np.ptp(score) > 0 else 0.5


def robust_positive_scale(X):
    X = _array(X)
    scale = np.ones(X.shape[1])
    for j in range(X.shape[1]):
        positive = X[:, j][X[:, j] > 0]
        if positive.size >= 5:
            med = float(np.median(positive))
            if np.isfinite(med) and med > EPS:
                scale[j] = med
    return scale


class CoordinateWiseEvidenceTransformer(BaseEstimator, TransformerMixin):
    """Map every coordinate to train-only evidence without reducing dimension."""

    def __init__(self, mode="binned", n_bins=12, alpha=5.0, clip=1e-4):
        self.mode, self.n_bins, self.alpha, self.clip = mode, n_bins, alpha, clip

    def fit(self, X, y):
        X, y = _array(X), np.asarray(y)
        if self.mode not in {"binned", "isotonic"}:
            raise ValueError(self.mode)
        self.models_ = []
        n0, n1 = (y == 0).sum(), (y == 1).sum()
        for j in range(X.shape[1]):
            x = X[:, j]
            if self.mode == "binned":
                positive = x[x > 0]
                edges = np.unique(np.quantile(
                    positive, np.linspace(0, 1, self.n_bins + 1)[1:-1]
                )) if positive.size else np.array([])
                ids = np.where(x == 0, 0, 1 + np.searchsorted(edges, x, side="right"))
                B = len(edges) + 2
                c0 = np.bincount(ids[y == 0], minlength=B)
                c1 = np.bincount(ids[y == 1], minlength=B)
                p0 = (c0 + self.alpha) / (n0 + self.alpha * B)
                p1 = (c1 + self.alpha) / (n1 + self.alpha * B)
                self.models_.append((edges, np.log(p1 / p0)))
            else:
                direction = _safe_auc(y, x) >= 0.5
                iso = IsotonicRegression(increasing=direction, out_of_bounds="clip", y_min=self.clip, y_max=1-self.clip)
                iso.fit(x, y.astype(float))
                self.models_.append(iso)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = _array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("feature count changed")
        E = np.empty_like(X)
        for j, model in enumerate(self.models_):
            if self.mode == "binned":
                edges, score = model
                ids = np.where(X[:, j] == 0, 0, 1 + np.searchsorted(edges, X[:, j], side="right"))
                E[:, j] = score[ids]
            else:
                p = np.clip(model.predict(X[:, j]), self.clip, 1 - self.clip)
                E[:, j] = np.log(p / (1 - p))
        return np.nan_to_num(E)


def inner_oof_evidence(X, y, mode, n_splits=5, seed=42, n_bins=12, alpha=5.0):
    """Return inner-OOF train evidence plus a full-train fitted transformer."""
    X, y = _array(X), np.asarray(y)
    oof = np.empty_like(X)
    coverage = np.zeros(len(X), dtype=np.int8)
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y):
        transformer = CoordinateWiseEvidenceTransformer(mode, n_bins, alpha).fit(X[tr], y[tr])
        oof[va] = transformer.transform(X[va])
        coverage[va] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("inner OOF coverage is not exactly one")
    full = CoordinateWiseEvidenceTransformer(mode, n_bins, alpha).fit(X, y)
    return oof, full


def _scaled_pair(A_train, A_valid):
    scaler = StandardScaler()
    return scaler.fit_transform(A_train), scaler.transform(A_valid)


def _sqeuclidean(A, B):
    return cdist(A, B, metric="sqeuclidean").astype(np.float32)


def _chi2(A, B, chunk=16):
    out = np.empty((len(A), len(B)), dtype=np.float32)
    for start in range(0, len(A), chunk):
        a = A[start:start + chunk, None, :]
        out[start:start + chunk] = np.sum((a - B[None, :, :]) ** 2 / (a + B[None, :, :] + EPS), axis=2)
    return out


def _l1(A, B):
    return cdist(A, B, metric="cityblock").astype(np.float32)


def _hist_intersection(A, B, chunk=16):
    out = np.empty((len(A), len(B)), dtype=np.float32)
    for start in range(0, len(A), chunk):
        out[start:start + chunk] = np.minimum(A[start:start + chunk, None, :], B[None, :, :]).sum(2)
    da = np.maximum(A.sum(1), EPS)
    db = np.maximum(B.sum(1), EPS)
    return out / np.sqrt(da[:, None] * db[None, :])


@dataclass(frozen=True)
class KernelSpec:
    name: str
    family: str
    spaces: tuple[str, ...]
    combination: str = "single"  # single, additive, product
    direct_spaces: tuple[str, ...] = ()


@dataclass
class CachedFold:
    repeat: int
    fold: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    train_matrices: dict[str, np.ndarray]
    valid_matrices: dict[str, np.ndarray]


@dataclass
class RepeatedCache:
    folds: list[CachedFold]
    seeds: tuple[int, ...]
    n_samples: int
    family: str


def _raw_blocks(Xtr, Xva):
    return _scaled_pair(np.log1p(Xtr), np.log1p(Xva))


def _direction_blocks(Xtr, Xva):
    ztr, zva = np.log1p(Xtr), np.log1p(Xva)
    l2tr, l2va = np.linalg.norm(ztr, axis=1, keepdims=True), np.linalg.norm(zva, axis=1, keepdims=True)
    l1tr, l1va = ztr.sum(1, keepdims=True), zva.sum(1, keepdims=True)
    magnitude_tr = np.column_stack([np.log1p(l1tr[:, 0]), np.log1p(l2tr[:, 0])])
    magnitude_va = np.column_stack([np.log1p(l1va[:, 0]), np.log1p(l2va[:, 0])])
    return {
        "raw": _scaled_pair(ztr, zva),
        "direction_l2": _scaled_pair(ztr / (l2tr + EPS), zva / (l2va + EPS)),
        "composition_l1": _scaled_pair(ztr / (l1tr + EPS), zva / (l1va + EPS)),
        "magnitude": _scaled_pair(magnitude_tr, magnitude_va),
    }


def _evidence_blocks(Xtr, ytr, Xva, seed, n_bins, alpha, modes=("binned", "isotonic")):
    raw_tr, raw_va = _raw_blocks(Xtr, Xva)
    result = {"raw": (raw_tr, raw_va)}
    for mode in modes:
        inner, transformer = inner_oof_evidence(Xtr, ytr, mode, 5, seed, n_bins, alpha)
        outer = transformer.transform(Xva)
        ev_tr, ev_va = _scaled_pair(inner, outer)
        result[f"evidence_{mode}"] = (ev_tr, ev_va)
        result[f"concat_{mode}"] = _scaled_pair(
            np.column_stack([np.log1p(Xtr), inner]),
            np.column_stack([np.log1p(Xva), outer]),
        )
    return result


def _custom_blocks(Xtr, Xva, names=None):
    names = set(names or {"raw", "robust_euclidean", "chi2", "laplacian", "hellinger", "histogram"})
    raw_tr, raw_va = _raw_blocks(Xtr, Xva)
    scale = robust_positive_scale(Xtr)
    rtr, rva = np.log1p(Xtr / scale), np.log1p(Xva / scale)
    ptr, pva = rtr / (rtr.sum(1, keepdims=True) + EPS), rva / (rva.sum(1, keepdims=True) + EPS)
    out = {}
    if "raw" in names: out["raw"] = ("distance", _sqeuclidean(raw_tr, raw_tr), _sqeuclidean(raw_va, raw_tr))
    if "robust_euclidean" in names: out["robust_euclidean"] = ("distance", _sqeuclidean(rtr, rtr), _sqeuclidean(rva, rtr))
    if "chi2" in names: out["chi2"] = ("distance", _chi2(rtr, rtr), _chi2(rva, rtr))
    if "laplacian" in names: out["laplacian"] = ("distance", _l1(rtr, rtr), _l1(rva, rtr))
    if "hellinger" in names: out["hellinger"] = ("distance", _sqeuclidean(np.sqrt(ptr), np.sqrt(ptr)), _sqeuclidean(np.sqrt(pva), np.sqrt(ptr)))
    if "histogram" in names: out["histogram"] = ("direct", _hist_intersection(ptr, ptr), _hist_intersection(pva, ptr))
    return out


def build_repeated_cache(X, y, seeds, family, n_splits=5, n_bins=12, alpha=5.0, n_jobs=1):
    """Build all matrices once for a family; validation data is not accepted."""
    X, y = _array(X), np.asarray(y)
    jobs = []
    for repeat, seed in enumerate(seeds):
        outer = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(outer.split(X, y)):
            jobs.append((repeat, seed, fold, tr, va))

    def make_fold(repeat, seed, fold, tr, va):
        print(f"cache {family}: repeat {repeat+1}/{len(seeds)}, fold {fold+1}/{n_splits}")
        Xtr, Xva, ytr = X[tr], X[va], y[tr]
        train_matrices, valid_matrices = {}, {}
        if family == "raw":
            A, B = _raw_blocks(Xtr, Xva)
            train_matrices["raw"], valid_matrices["raw"] = _sqeuclidean(A, A), _sqeuclidean(B, A)
        elif family in {"evidence", "evidence_binned", "evidence_isotonic"}:
            modes = ("binned", "isotonic") if family == "evidence" else (family.removeprefix("evidence_"),)
            blocks = _evidence_blocks(Xtr, ytr, Xva, seed + fold, n_bins, alpha, modes)
            for name, (A, B) in blocks.items():
                train_matrices[name], valid_matrices[name] = _sqeuclidean(A, A), _sqeuclidean(B, A)
        elif family == "direction":
            for name, (A, B) in _direction_blocks(Xtr, Xva).items():
                train_matrices[name], valid_matrices[name] = _sqeuclidean(A, A), _sqeuclidean(B, A)
        elif family.startswith("custom"):
            requested = {
                "custom": None, "custom_robust": {"robust_euclidean"},
                "custom_chi2": {"raw", "chi2"}, "custom_laplacian": {"laplacian"},
                "custom_hellinger": {"hellinger"}, "custom_histogram": {"histogram"},
            }.get(family)
            if family not in {"custom", "custom_robust", "custom_chi2", "custom_laplacian", "custom_hellinger", "custom_histogram"}:
                raise ValueError(family)
            for name, (_, A, B) in _custom_blocks(Xtr, Xva, requested).items():
                train_matrices[name], valid_matrices[name] = A, B
        else:
            raise ValueError(family)
        return CachedFold(repeat, fold, tr, va, ytr, y[va], train_matrices, valid_matrices)

    folds = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(make_fold)(*job) for job in jobs
    )
    return RepeatedCache(folds, tuple(seeds), len(X), family)


def _combined_matrices(Xtr, ytr, Xva, spaces, seed, n_bins=12, alpha=5.0):
    """Build only requested successful blocks for the gated Phase-D experiment."""
    train, valid = {}, {}
    remaining = set(spaces)
    if "raw" in remaining:
        A, B = _raw_blocks(Xtr, Xva)
        train["raw"], valid["raw"] = _sqeuclidean(A, A), _sqeuclidean(B, A)
        remaining.remove("raw")
    ev_names = remaining & {"evidence_binned", "evidence_isotonic"}
    for name in ev_names:
        mode = name.removeprefix("evidence_")
        inner, transformer = inner_oof_evidence(Xtr, ytr, mode, 5, seed, n_bins, alpha)
        A, B = _scaled_pair(inner, transformer.transform(Xva))
        train[name], valid[name] = _sqeuclidean(A, A), _sqeuclidean(B, A)
    remaining -= ev_names
    direction_names = remaining & {"direction_l2", "composition_l1", "magnitude"}
    if direction_names:
        blocks = _direction_blocks(Xtr, Xva)
        for name in direction_names:
            A, B = blocks[name]
            train[name], valid[name] = _sqeuclidean(A, A), _sqeuclidean(B, A)
    remaining -= direction_names
    custom_names = remaining & {"robust_euclidean", "chi2", "laplacian", "hellinger", "histogram"}
    if custom_names:
        blocks = _custom_blocks(Xtr, Xva, custom_names)
        for name in custom_names:
            _, train[name], valid[name] = blocks[name]
    remaining -= custom_names
    if remaining:
        raise KeyError(f"unsupported combined spaces: {remaining}")
    return train, valid


def build_combined_cache(X, y, seeds, spaces, n_splits=5, n_bins=12, alpha=5.0, n_jobs=1):
    X, y = _array(X), np.asarray(y)
    jobs = []
    for repeat, seed in enumerate(seeds):
        outer = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(outer.split(X, y)):
            jobs.append((repeat, seed, fold, tr, va))

    def make_fold(repeat, seed, fold, tr, va):
        print(f"cache combined: repeat {repeat+1}/{len(seeds)}, fold {fold+1}/{n_splits}")
        train, valid = _combined_matrices(X[tr], y[tr], X[va], spaces, seed + fold, n_bins, alpha)
        return CachedFold(repeat, fold, tr, va, y[tr], y[va], train, valid)

    folds = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(make_fold)(*job) for job in jobs
    )
    return RepeatedCache(folds, tuple(seeds), len(X), "combined")


def external_validation_combined(X_train, y_train, X_valid, y_valid, spaces, spec, params,
                                 seed=42, n_bins=12, alpha=5.0):
    train, valid = _combined_matrices(_array(X_train), np.asarray(y_train), _array(X_valid),
                                      spaces, seed, n_bins, alpha)
    Ktr, Kva = build_kernel(train, spec, params), build_kernel(valid, spec, params)
    validate_kernel_matrix(Ktr, Kva, len(X_train), len(X_valid))
    model = SVC(C=params["C"], kernel="precomputed").fit(Ktr, y_train)
    score = model.decision_function(Kva)
    return roc_auc_score(y_valid, score), score


def _space_kernel(matrix, space, spec, params):
    if space in spec.direct_spaces:
        return matrix
    return np.exp(-params[f"gamma_{space}"] * matrix)


def build_kernel(matrices, spec, params):
    kernels = [_space_kernel(matrices[s], s, spec, params) for s in spec.spaces]
    if spec.combination == "single":
        return kernels[0]
    if spec.combination == "product":
        out = kernels[0]
        for K in kernels[1:]:
            out = out * K
        return out
    if spec.combination == "additive":
        if len(kernels) == 2:
            weights = (params["beta"], 1 - params["beta"])
        else:
            logits = [0.0] + [params[f"logit_{i}"] for i in range(1, len(kernels))]
            weights = softmax(logits)
        return sum(w * K for w, K in zip(weights, kernels))
    raise ValueError(spec.combination)


def validate_kernel_matrix(K_train, K_valid, n_train, n_valid):
    if K_train.shape != (n_train, n_train) or K_valid.shape != (n_valid, n_train):
        raise AssertionError("precomputed kernel shape mismatch")
    if not np.isfinite(K_train).all() or not np.isfinite(K_valid).all():
        raise AssertionError("kernel contains non-finite values")
    if not np.allclose(K_train, K_train.T, atol=2e-5):
        raise AssertionError("train kernel is not symmetric")
    if np.any(np.diag(K_train) <= 0):
        raise AssertionError("kernel diagonal must be positive")


def evaluate_cached(cache, spec, params, return_oof=False, n_jobs=1):
    aucs, oof = [], np.full((len(cache.seeds), cache.n_samples), np.nan)

    def score_fold(item):
        Ktr = build_kernel(item.train_matrices, spec, params)
        Kva = build_kernel(item.valid_matrices, spec, params)
        validate_kernel_matrix(Ktr, Kva, len(item.y_train), len(item.y_valid))
        model = SVC(C=params["C"], kernel="precomputed").fit(Ktr, item.y_train)
        score = model.decision_function(Kva)
        return item.repeat, item.valid_idx, roc_auc_score(item.y_valid, score), score

    fold_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(score_fold)(item) for item in cache.folds
    )
    for repeat, valid_idx, auc, score in fold_results:
        aucs.append(auc)
        oof[repeat, valid_idx] = score
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF predictions incomplete")
    aucs = np.asarray(aucs)
    return (aucs, oof) if return_oof else aucs


def optimize_cached(cache, spec, n_trials=40, seed=42, n_jobs=1):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {"C": trial.suggest_float("C", 0.3, 30, log=True)}
        for space in spec.spaces:
            if space not in spec.direct_spaces:
                high = 3e-2 if space == "raw" else 1e-1
                params[f"gamma_{space}"] = trial.suggest_float(f"gamma_{space}", 1e-4, high, log=True)
        if spec.combination == "additive":
            if len(spec.spaces) == 2:
                params["beta"] = trial.suggest_float("beta", 0.05, 0.95)
            else:
                for i in range(1, len(spec.spaces)):
                    params[f"logit_{i}"] = trial.suggest_float(f"logit_{i}", -4, 4)
        return float(evaluate_cached(cache, spec, params, n_jobs=n_jobs).mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    params = dict(study.best_params)
    aucs, oof = evaluate_cached(cache, spec, params, True, n_jobs=n_jobs)
    return study, params, aucs, oof


def _validation_matrices(X_train, y_train, X_valid, family, seed=42, n_bins=12, alpha=5.0):
    Xtr, Xva, y = _array(X_train), _array(X_valid), np.asarray(y_train)
    train, valid = {}, {}
    if family == "raw":
        A, B = _raw_blocks(Xtr, Xva); train["raw"], valid["raw"] = _sqeuclidean(A, A), _sqeuclidean(B, A)
    elif family in {"evidence", "evidence_binned", "evidence_isotonic"}:
        modes = ("binned", "isotonic") if family == "evidence" else (family.removeprefix("evidence_"),)
        for name, (A, B) in _evidence_blocks(Xtr, y, Xva, seed, n_bins, alpha, modes).items():
            train[name], valid[name] = _sqeuclidean(A, A), _sqeuclidean(B, A)
    elif family == "direction":
        for name, (A, B) in _direction_blocks(Xtr, Xva).items():
            train[name], valid[name] = _sqeuclidean(A, A), _sqeuclidean(B, A)
    elif family.startswith("custom"):
        requested = {
            "custom": None, "custom_robust": {"robust_euclidean"},
            "custom_chi2": {"raw", "chi2"}, "custom_laplacian": {"laplacian"},
            "custom_hellinger": {"hellinger"}, "custom_histogram": {"histogram"},
        }.get(family)
        if family not in {"custom", "custom_robust", "custom_chi2", "custom_laplacian", "custom_hellinger", "custom_histogram"}:
            raise ValueError(family)
        for name, (_, A, B) in _custom_blocks(Xtr, Xva, requested).items(): train[name], valid[name] = A, B
    else:
        raise ValueError(family)
    return train, valid


def external_validation(X_train, y_train, X_valid, y_valid, family, spec, params, seed=42, n_bins=12, alpha=5.0):
    train, valid = _validation_matrices(X_train, y_train, X_valid, family, seed, n_bins, alpha)
    Ktr, Kva = build_kernel(train, spec, params), build_kernel(valid, spec, params)
    validate_kernel_matrix(Ktr, Kva, len(X_train), len(X_valid))
    model = SVC(C=params["C"], kernel="precomputed").fit(Ktr, y_train)
    score = model.decision_function(Kva)
    return roc_auc_score(y_valid, score), score


def evidence_diagnostics(X_train, y_train, X_valid=None, y_valid=None, seed=42, n_bins=12, alpha=5.0):
    X, y = _array(X_train), np.asarray(y_train)
    rows, outputs = [], {}
    for mode in ("binned", "isotonic"):
        E, transformer = inner_oof_evidence(X, y, mode, 5, seed, n_bins, alpha)
        outputs[mode] = (E, transformer)
        for j in range(X.shape[1]):
            rows.append({
                "representation": f"evidence_{mode}", "feature": j,
                "raw_single_auc": _safe_auc(y, X[:, j]),
                "evidence_single_auc": _safe_auc(y, E[:, j]),
                "raw_spearman": spearmanr(X[:, j], y).statistic,
                "raw_evidence_corr": spearmanr(X[:, j], E[:, j]).statistic,
            })
    detail = pd.DataFrame(rows)
    summary = detail.groupby("representation").evidence_single_auc.agg(
        median_auc="median", max_auc="max",
        n_auc_gt_060=lambda s: int((s > .60).sum()),
        n_auc_gt_065=lambda s: int((s > .65).sum()),
        n_auc_gt_070=lambda s: int((s > .70).sum()),
    ).reset_index()
    return detail, summary, outputs


def paired_stats(aucs, baseline):
    delta = np.asarray(aucs) - np.asarray(baseline)
    try: p = wilcoxon(delta).pvalue if np.any(delta) else 1.0
    except ValueError: p = np.nan
    return {"mean_delta": delta.mean(), "median_delta": np.median(delta),
            "wins": int((delta > 1e-12).sum()), "losses": int((delta < -1e-12).sum()),
            "wilcoxon_p": p, "delta_per_fold": delta.tolist()}


def result_row(name, family, representation, kernel, aucs, validation_auc, oof,
               raw_aucs, raw_oof, current_aucs=None, current_oof=None,
               fit_time=np.nan, params=None):
    aucs = np.asarray(aucs)
    raw = paired_stats(aucs, raw_aucs)
    current = paired_stats(aucs, current_aucs) if current_aucs is not None else None
    return {
        "experiment": name, "family": family, "representation": representation, "kernel": kernel,
        "CV_mean": aucs.mean(), "CV_std": aucs.std(), "CV_median": np.median(aucs),
        "CV_min": aucs.min(), "CV_max": aucs.max(),
        "delta_vs_raw": raw["mean_delta"], "wins_vs_raw": raw["wins"], "losses_vs_raw": raw["losses"],
        "delta_vs_current": np.nan if current is None else current["mean_delta"],
        "wins_vs_current": np.nan if current is None else current["wins"],
        "losses_vs_current": np.nan if current is None else current["losses"],
        "wilcoxon_vs_raw": raw["wilcoxon_p"],
        "wilcoxon_vs_current": np.nan if current is None else current["wilcoxon_p"],
        "validation_auc": validation_auc,
        "corr_with_raw": spearmanr(np.asarray(oof).ravel(), np.asarray(raw_oof).ravel()).statistic,
        "corr_with_current": np.nan if current_oof is None else spearmanr(np.asarray(oof).ravel(), np.asarray(current_oof).ravel()).statistic,
        "fit_time": fit_time, "params": params, "auc_per_fold": aucs.tolist(), "oof_predictions": oof,
    }


def disagreement_diagnostic(new_oof, current_oof, y):
    new, current = np.asarray(new_oof), np.asarray(current_oof)
    y_rep = np.tile(np.asarray(y), new.shape[0])
    nr = pd.Series(new.ravel()).rank(pct=True).to_numpy()
    cr = pd.Series(current.ravel()).rank(pct=True).to_numpy()
    diff = np.abs(nr - cr); mask = diff >= np.quantile(diff, .75)
    return {"subset_size": int(mask.sum()), "new_auc": roc_auc_score(y_rep[mask], new.ravel()[mask]),
            "current_auc": roc_auc_score(y_rep[mask], current.ravel()[mask]),
            "mean_rank_diff": diff[mask].mean()}


def current_best_repeated(X, y, seeds, config, params, n_splits=5):
    from structural_experiments import score_current_best_repeated
    return score_current_best_repeated(X, y, seeds, config, params, n_splits)


def current_best_validation(X_train, y_train, X_valid, y_valid, config, params, seed=42):
    from svc_experiments import fit_validation_blocks, validation_auc
    _, blocks, _, _ = fit_validation_blocks(X_train, y_train, X_valid, config, seed)
    return validation_auc(blocks, y_train, y_valid, params)
