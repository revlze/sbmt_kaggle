"""Repeated-CV structural representations and monotone metric models.

Used by the final sections of ``svc.ipynb``.  External validation is never used
for hyperparameter selection or early stopping, and this module never writes a
submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.spatial.distance import cdist
from scipy.stats import norm, pearsonr, spearmanr, wilcoxon
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


EPS = 1e-10


def _array(X):
    return np.asarray(X, dtype=np.float64)


def robust_positive_scale(X):
    """Fold-local median of positive values, with a safe 1.0 fallback."""
    X = _array(X)
    scale = np.ones(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        positive = X[:, j][X[:, j] > 0]
        if positive.size >= 5:
            value = float(np.median(positive))
            if np.isfinite(value) and value > EPS:
                scale[j] = value
    return scale


def block_sort(Z, n_blocks=1):
    if Z.shape[1] % n_blocks:
        raise ValueError(f"{Z.shape[1]} features cannot be split into {n_blocks} equal blocks")
    return np.concatenate([np.sort(part, axis=1) for part in np.split(Z, n_blocks, axis=1)], axis=1)


def feature_descriptors(X):
    X = _array(X)
    qs = np.quantile(X, [0.10, 0.25, 0.50, 0.75, 0.90, 0.99], axis=0).T
    return np.column_stack([X.mean(0), X.std(0), (X == 0).mean(0), qs])


def fit_unsupervised_groups(X_train, n_clusters, seed):
    desc = StandardScaler().fit_transform(feature_descriptors(X_train))
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(desc)
    return [np.flatnonzero(labels == k) for k in range(n_clusters) if np.any(labels == k)]


def grouped_sort(Z, groups):
    return np.concatenate([np.sort(Z[:, idx], axis=1) for idx in groups], axis=1)


def _base_representations(X_train, X_valid):
    raw_train, raw_valid = np.log1p(_array(X_train)), np.log1p(_array(X_valid))
    sorted_train, sorted_valid = np.sort(raw_train, axis=1), np.sort(raw_valid, axis=1)
    return {
        "raw": (raw_train, raw_valid),
        "sorted": (sorted_train, sorted_valid),
        "shape": (sorted_train - sorted_train.mean(1, keepdims=True),
                  sorted_valid - sorted_valid.mean(1, keepdims=True)),
        "shape_median": (sorted_train - np.median(sorted_train, axis=1, keepdims=True),
                         sorted_valid - np.median(sorted_valid, axis=1, keepdims=True)),
    }


def make_representations(X_train, X_valid, names, seed=42):
    """Fit every learned normalization/grouping using X_train only."""
    X_train, X_valid = _array(X_train), _array(X_valid)
    reps = _base_representations(X_train, X_valid)
    if "robust_sorted" in names:
        scale = robust_positive_scale(X_train)
        reps["robust_sorted"] = (np.sort(np.log1p(X_train / scale), axis=1),
                                 np.sort(np.log1p(X_valid / scale), axis=1))
    ztr, zva = np.log1p(X_train), np.log1p(X_valid)
    for blocks in (2, 4, 8):
        name = f"block{blocks}"
        if name in names:
            reps[name] = (block_sort(ztr, blocks), block_sort(zva, blocks))
    for clusters in (4, 8, 16):
        name = f"cluster{clusters}"
        if name in names:
            groups = fit_unsupervised_groups(X_train, clusters, seed)
            reps[name] = (grouped_sort(ztr, groups), grouped_sort(zva, groups))
    missing = set(names) - set(reps)
    if missing:
        raise KeyError(f"Unknown representations: {sorted(missing)}")
    scaled = {}
    for name in names:
        scaler = StandardScaler()
        tr, va = reps[name]
        scaled[name] = (scaler.fit_transform(tr), scaler.transform(va))
    return scaled


@dataclass
class StructuralFold:
    repeat: int
    fold: int
    valid_idx: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    distances_train: dict[str, np.ndarray]
    distances_valid: dict[str, np.ndarray]


@dataclass
class StructuralCache:
    folds: list[StructuralFold]
    seeds: tuple[int, ...]
    n_samples: int
    representations: tuple[str, ...]


def build_repeated_distance_cache(
    X, y, seeds, n_splits=5,
    representations=("raw", "sorted", "shape", "robust_sorted"),
    dtype=np.float32,
):
    """Cache scaled squared distances once; Optuna trials only exponentiate them."""
    X, y = _array(X), np.asarray(y)
    folds = []
    for repeat, seed in enumerate(seeds):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            print(f"distance cache repeat {repeat + 1}/{len(seeds)}, fold {fold + 1}/{n_splits}")
            reps = make_representations(X[tr], X[va], representations, seed)
            dtr, dva = {}, {}
            for name, (A, B) in reps.items():
                dtr[name] = cdist(A, A, metric="sqeuclidean").astype(dtype)
                dva[name] = cdist(B, A, metric="sqeuclidean").astype(dtype)
            folds.append(StructuralFold(repeat, fold, va, y[tr], y[va], dtr, dva))
    return StructuralCache(folds, tuple(seeds), len(X), tuple(representations))


def kernel_from_distances(distances, params):
    kind, spaces = params["kernel"], params["spaces"]
    kernels = [np.exp(-params[f"gamma_{s}"] * distances[s]) for s in spaces]
    if kind == "single":
        return kernels[0]
    if kind == "product":
        result = kernels[0]
        for kernel in kernels[1:]:
            result = result * kernel
        return result
    if kind == "additive":
        if len(kernels) == 2:
            weights = np.array([params["beta"], 1.0 - params["beta"]])
        else:
            logits = np.array([0.0] + [params[f"logit_{i}"] for i in range(1, len(kernels))])
            weights = softmax(logits)
        return sum(w * k for w, k in zip(weights, kernels))
    raise ValueError(kind)


def evaluate_kernel(cache, params, return_oof=False):
    aucs = []
    oof = np.full((len(cache.seeds), cache.n_samples), np.nan, dtype=np.float64)
    for item in cache.folds:
        model = SVC(C=params["C"], kernel="precomputed")
        model.fit(kernel_from_distances(item.distances_train, params), item.y_train)
        pred = model.decision_function(kernel_from_distances(item.distances_valid, params))
        aucs.append(roc_auc_score(item.y_valid, pred))
        oof[item.repeat, item.valid_idx] = pred
    aucs = np.asarray(aucs)
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF matrix is incomplete")
    return (aucs, oof) if return_oof else aucs


def optimize_structural_kernel(cache, spaces, kernel="single", n_trials=50, seed=42):
    import optuna
    spaces = tuple(spaces)
    if kernel == "single" and len(spaces) != 1:
        raise ValueError("single kernel requires exactly one representation")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {"kernel": kernel, "spaces": spaces,
                  "C": trial.suggest_float("C", 0.3, 30.0, log=True)}
        for space in spaces:
            high = 1e-1 if space != "raw" else 3e-2
            params[f"gamma_{space}"] = trial.suggest_float(f"gamma_{space}", 1e-4, high, log=True)
        if kernel == "additive" and len(spaces) == 2:
            params["beta"] = trial.suggest_float("beta", 0.05, 0.95)
        elif kernel == "additive" and len(spaces) > 2:
            for i in range(1, len(spaces)):
                params[f"logit_{i}"] = trial.suggest_float(f"logit_{i}", -4.0, 4.0)
        return float(evaluate_kernel(cache, params).mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    params = {"kernel": kernel, "spaces": spaces, **study.best_params}
    aucs, oof = evaluate_kernel(cache, params, return_oof=True)
    return study, params, aucs, oof


def external_validation_kernel(X_train, y_train, X_valid, y_valid, params):
    spaces = params["spaces"]
    reps = make_representations(X_train, X_valid, spaces, seed=0)
    dtr, dva = {}, {}
    for name, (A, B) in reps.items():
        dtr[name] = cdist(A, A, metric="sqeuclidean")
        dva[name] = cdist(B, A, metric="sqeuclidean")
    model = SVC(C=params["C"], kernel="precomputed")
    model.fit(kernel_from_distances(dtr, params), y_train)
    score = model.decision_function(kernel_from_distances(dva, params))
    return roc_auc_score(y_valid, score), score


def fold_summary(name, representation, kernel, aucs, validation_auc, params,
                 baseline_raw=None, baseline_current=None, fit_time=np.nan):
    aucs = np.asarray(aucs)
    row = {
        "experiment": name, "representation": representation, "kernel": kernel,
        "CV_mean": aucs.mean(), "CV_std": aucs.std(), "CV_median": np.median(aucs),
        "CV_min": aucs.min(), "CV_max": aucs.max(), "validation_auc": validation_auc,
        "params": params, "fit_time": fit_time,
    }
    for label, baseline in (("raw", baseline_raw), ("current_best", baseline_current)):
        if baseline is None:
            row[f"delta_vs_{label}"] = np.nan
            row[f"wins_vs_{label}"] = np.nan
            continue
        delta = aucs - np.asarray(baseline)
        row[f"delta_vs_{label}"] = delta.mean()
        row[f"wins_vs_{label}"] = int((delta > 1e-12).sum())
    return row


def paired_diagnostics(aucs, baseline):
    delta = np.asarray(aucs) - np.asarray(baseline)
    try:
        pvalue = wilcoxon(delta).pvalue if np.any(delta) else 1.0
    except ValueError:
        pvalue = np.nan
    return {
        "mean_delta": delta.mean(), "std_delta": delta.std(), "median_delta": np.median(delta),
        "wins": int((delta > 1e-12).sum()), "losses": int((delta < -1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()), "wilcoxon_p": pvalue,
        "delta_per_fold": delta.tolist(),
    }


def prediction_complementarity(oof_scores, y, reference="raw"):
    """Correlations use all repeated OOF predictions aligned by repeat/sample."""
    names = list(oof_scores)
    rows = []
    ref = np.asarray(oof_scores[reference]).ravel()
    y_rep = np.tile(np.asarray(y), np.asarray(oof_scores[reference]).shape[0])
    ref_rank = pd.Series(ref).rank(pct=True).to_numpy()
    for name in names:
        score = np.asarray(oof_scores[name]).ravel()
        disagreement = np.abs(ref_rank - pd.Series(score).rank(pct=True).to_numpy())
        mask = disagreement >= np.quantile(disagreement, 0.75)
        rows.append({
            "score": name,
            "pearson_with_raw": pearsonr(ref, score).statistic,
            "spearman_with_raw": spearmanr(ref, score).statistic,
            "top_quartile_disagreement_auc": roc_auc_score(y_rep[mask], score[mask]),
            "mean_abs_rank_disagreement": disagreement.mean(),
        })
    return pd.DataFrame(rows)


def permutation_invariance_diagnostic(X, seed=42):
    X = _array(X)
    rng = np.random.default_rng(seed)
    permuted = np.vstack([row[rng.permutation(X.shape[1])] for row in X])
    sorted_original = np.sort(np.log1p(X), axis=1)
    sorted_permuted = np.sort(np.log1p(permuted), axis=1)
    return permuted, float(np.max(np.abs(sorted_original - sorted_permuted)))


def score_current_best_repeated(X, y, seeds, config, params, n_splits=5, seed_offset=10_000):
    """Evaluate the previous-stage winner on exactly the same repeated folds."""
    from svc_experiments import LeakageSafeFeatureBuilder, _scale_blocks, validation_auc

    X, y = _array(X), np.asarray(y)
    aucs, oof = [], np.full((len(seeds), len(X)), np.nan)
    for repeat, seed in enumerate(seeds):
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            print(f"current best repeat={repeat + 1}, fold={fold + 1}")
            builder = LeakageSafeFeatureBuilder(config, seed + seed_offset + fold, n_splits)
            rtr, gtr, dtr, _, _ = builder.fit_transform(X[tr], y[tr])
            rva, gva, dva, _, _ = builder.transform(X[va])
            blocks = _scale_blocks(rtr, rva, gtr, gva, dtr, dva)
            auc, pred = validation_auc(blocks, y[tr], y[va], params)
            aucs.append(auc); oof[repeat, va] = pred
    return np.asarray(aucs), oof


# ---------------------------------------------------------------------------
# Monotone low-rank metric
# ---------------------------------------------------------------------------

def _require_torch():
    import torch
    return torch


class MonotoneMetricModel:
    """Factory wrapper so importing this module does not eagerly initialize torch."""

    @staticmethod
    def build(n_features, rank):
        torch = _require_torch()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diag_raw = torch.nn.Parameter(torch.full((n_features,), -3.0))
                self.U_raw = None if rank == 0 else torch.nn.Parameter(torch.full((n_features, rank), -4.0))
                self.bias = torch.nn.Parameter(torch.zeros(()))

            def forward(self, z):
                diag = torch.nn.functional.softplus(self.diag_raw)
                distance = (z.square() * diag).sum(1)
                if self.U_raw is not None:
                    U = torch.nn.functional.softplus(self.U_raw)
                    distance = distance + (z @ U).square().sum(1)
                return self.bias - distance

        return Model()


def train_monotone_metric(X_train, y_train, X_stop, y_stop, rank=8, lr=1e-3,
                          weight_decay=1e-4, max_epochs=400, patience=40,
                          batch_size=256, hybrid_bce=0.0, lambda_diag=1e-7,
                          lambda_lowrank=1e-7, seed=42, device=None):
    torch = _require_torch()
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scale = robust_positive_scale(X_train)
    A = torch.as_tensor(np.log1p(_array(X_train) / scale), dtype=torch.float32, device=device)
    B = torch.as_tensor(np.log1p(_array(X_stop) / scale), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(y_train), dtype=torch.float32, device=device)
    model = MonotoneMetricModel.build(A.shape[1], rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos = torch.nonzero(y == 1, as_tuple=True)[0]
    neg = torch.nonzero(y == 0, as_tuple=True)[0]
    best_auc, best_state, stale = -np.inf, None, 0
    steps = max(1, int(np.ceil(max(len(pos), len(neg)) / batch_size)))
    for epoch in range(max_epochs):
        model.train()
        for _ in range(steps):
            ip = pos[torch.randint(len(pos), (batch_size,), device=device)]
            ine = neg[torch.randint(len(neg), (batch_size,), device=device)]
            sp, sn = model(A[ip]), model(A[ine])
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            diag = torch.nn.functional.softplus(model.diag_raw)
            loss = loss + lambda_diag * diag.square().mean()
            if model.U_raw is not None:
                U = torch.nn.functional.softplus(model.U_raw)
                loss = loss + lambda_lowrank * U.square().mean()
            if hybrid_bce:
                idx = torch.cat([ip, ine])
                loss = loss + hybrid_bce * torch.nn.functional.binary_cross_entropy_with_logits(model(A[idx]), y[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(B).detach().cpu().numpy()
        auc = roc_auc_score(y_stop, pred)
        if auc > best_auc + 1e-5:
            best_auc, stale = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, scale, best_auc, epoch + 1, device


def predict_monotone_metric(model, scale, X, device):
    torch = _require_torch()
    z = torch.as_tensor(np.log1p(_array(X) / scale), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        return model(z).detach().cpu().numpy()


def fit_monotone_fixed_epochs(X, y, rank, epochs, lr=1e-3, weight_decay=1e-4,
                              batch_size=256, hybrid_bce=0.0, lambda_diag=1e-7,
                              lambda_lowrank=1e-7, seed=42, device=None):
    """Refit on all training rows after inner early stopping chose epoch count."""
    torch = _require_torch()
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scale = robust_positive_scale(X)
    A = torch.as_tensor(np.log1p(_array(X) / scale), dtype=torch.float32, device=device)
    target = torch.as_tensor(np.asarray(y).copy(), dtype=torch.float32, device=device)
    model = MonotoneMetricModel.build(A.shape[1], rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos = torch.nonzero(target == 1, as_tuple=True)[0]
    neg = torch.nonzero(target == 0, as_tuple=True)[0]
    steps = max(1, int(np.ceil(max(len(pos), len(neg)) / batch_size)))
    for _ in range(int(epochs)):
        model.train()
        for _ in range(steps):
            ip = pos[torch.randint(len(pos), (batch_size,), device=device)]
            ine = neg[torch.randint(len(neg), (batch_size,), device=device)]
            sp, sn = model(A[ip]), model(A[ine])
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            diag = torch.nn.functional.softplus(model.diag_raw)
            loss = loss + lambda_diag * diag.square().mean()
            if model.U_raw is not None:
                U = torch.nn.functional.softplus(model.U_raw)
                loss = loss + lambda_lowrank * U.square().mean()
            if hybrid_bce:
                idx = torch.cat([ip, ine])
                loss = loss + hybrid_bce * torch.nn.functional.binary_cross_entropy_with_logits(model(A[idx]), target[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model, scale, device


def repeated_metric_cv(X, y, seeds, ranks=(0, 4, 8, 16, 32), n_splits=5, **train_kwargs):
    X, y = _array(X), np.asarray(y)
    results = {}
    for rank in ranks:
        aucs, oof = [], np.full((len(seeds), len(X)), np.nan)
        epochs = []
        for repeat, seed in enumerate(seeds):
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
            for fold, (tr, va) in enumerate(cv.split(X, y)):
                print(f"metric rank={rank}, repeat={repeat + 1}, fold={fold + 1}")
                model, scale, auc, used, device = train_monotone_metric(
                    X[tr], y[tr], X[va], y[va], rank=rank,
                    seed=seed * 100 + fold, **train_kwargs,
                )
                pred = predict_monotone_metric(model, scale, X[va], device)
                aucs.append(roc_auc_score(y[va], pred)); epochs.append(used)
                oof[repeat, va] = pred
        results[rank] = {"auc_per_fold": np.asarray(aucs), "oof": oof,
                         "mean_epochs": float(np.mean(epochs))}
    return results


def metric_external_validation(X_train, y_train, X_valid, y_valid, rank, seed=42, **train_kwargs):
    indices = np.arange(len(X_train))
    fit_idx, stop_idx = train_test_split(indices, test_size=0.15, stratify=y_train, random_state=seed)
    _, _, _, epochs, device = train_monotone_metric(
        _array(X_train)[fit_idx], np.asarray(y_train)[fit_idx],
        _array(X_train)[stop_idx], np.asarray(y_train)[stop_idx],
        rank=rank, seed=seed, **train_kwargs,
    )
    refit_kwargs = {k: v for k, v in train_kwargs.items() if k not in {"max_epochs", "patience"}}
    model, scale, device = fit_monotone_fixed_epochs(
        X_train, y_train, rank, epochs, seed=seed, device=device, **refit_kwargs,
    )
    score = predict_monotone_metric(model, scale, X_valid, device)
    return roc_auc_score(y_valid, score), score, epochs


# ---------------------------------------------------------------------------
# Gaussian copula diagnostics
# ---------------------------------------------------------------------------

class GaussianCopulaFeatures:
    def __init__(self, n_bins=8, alpha=5.0, n_splits=5, seed=42):
        self.n_bins, self.alpha, self.n_splits, self.seed = n_bins, alpha, n_splits, seed

    @staticmethod
    def _ecdf_transform(X, sorted_columns):
        Z = np.empty_like(X, dtype=np.float64)
        for j, values in enumerate(sorted_columns):
            u = (np.searchsorted(values, X[:, j], side="right") + 0.5) / (len(values) + 1.0)
            Z[:, j] = norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
        return Z

    def fit(self, X, y):
        X, y = _array(X), np.asarray(y)
        self.sorted_ = [[np.sort(X[y == c, j]) for j in range(X.shape[1])] for c in (0, 1)]
        self.cov_, self.logdet_ = [], []
        for c in (0, 1):
            Z = self._ecdf_transform(X[y == c], self.sorted_[c])
            lw = LedoitWolf().fit(Z)
            self.cov_.append(lw)
            self.logdet_.append(np.linalg.slogdet(lw.covariance_)[1])
        self.bin_models_ = []
        n0, n1 = (y == 0).sum(), (y == 1).sum()
        for j in range(X.shape[1]):
            positive = X[:, j][X[:, j] > 0]
            edges = np.unique(np.quantile(positive, np.linspace(0, 1, self.n_bins + 1)[1:-1])) if len(positive) else np.array([])
            ids = np.where(X[:, j] == 0, 0, 1 + np.searchsorted(edges, X[:, j], side="right"))
            B = len(edges) + 2
            c0 = np.bincount(ids[y == 0], minlength=B); c1 = np.bincount(ids[y == 1], minlength=B)
            lp0 = np.log((c0 + self.alpha) / (n0 + self.alpha * B))
            lp1 = np.log((c1 + self.alpha) / (n1 + self.alpha * B))
            self.bin_models_.append((edges, lp0, lp1))
        return self

    def transform(self, X):
        X = _array(X)
        copula = []
        for c in (0, 1):
            Z = self._ecdf_transform(X, self.sorted_[c])
            quad = np.einsum("ij,jk,ik->i", Z, self.cov_[c].precision_, Z, optimize=True)
            # log multivariate Gaussian - sum log standard-normal marginals
            copula.append(-0.5 * (self.logdet_[c] + quad - np.square(Z).sum(1)))
        marginal = [np.zeros(len(X)), np.zeros(len(X))]
        for j, (edges, lp0, lp1) in enumerate(self.bin_models_):
            ids = np.where(X[:, j] == 0, 0, 1 + np.searchsorted(edges, X[:, j], side="right"))
            marginal[0] += lp0[ids]; marginal[1] += lp1[ids]
        return pd.DataFrame({
            "copula_logp0": copula[0], "copula_logp1": copula[1],
            "copula_llr": copula[1] - copula[0],
            "marginal_llr": marginal[1] - marginal[0],
            "marginal_plus_copula_llr": marginal[1] + copula[1] - marginal[0] - copula[0],
        })

    def fit_transform(self, X, y):
        X, y = _array(X), np.asarray(y)
        self.fit(X, y)
        cv = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        parts = []
        for tr, va in cv.split(X, y):
            model = GaussianCopulaFeatures(self.n_bins, self.alpha, self.n_splits, self.seed).fit(X[tr], y[tr])
            frame = model.transform(X[va]); frame.index = va; parts.append(frame)
        return pd.concat(parts).sort_index().reset_index(drop=True)


def repeated_copula_cv(X, y, seeds, n_splits=5, n_bins=8, alpha=5.0):
    X, y = np.log1p(_array(X)), np.asarray(y)
    variants = ("copula_llr", "marginal_llr", "marginal_plus_copula_llr")
    result = {name: {"aucs": [], "oof": np.full((len(seeds), len(X)), np.nan)} for name in variants}
    for repeat, seed in enumerate(seeds):
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            print(f"copula repeat={repeat + 1}, fold={fold + 1}")
            model = GaussianCopulaFeatures(n_bins, alpha, n_splits, seed).fit(X[tr], y[tr])
            score = model.transform(X[va])
            for name in variants:
                pred = score[name].to_numpy()
                result[name]["aucs"].append(roc_auc_score(y[va], pred))
                result[name]["oof"][repeat, va] = pred
    for name in variants:
        result[name]["aucs"] = np.asarray(result[name]["aucs"])
    return result
