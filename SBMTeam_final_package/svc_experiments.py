"""Leakage-safe SVC experiments for ``svc.ipynb``.

The important invariant is that supervised features used for an outer-fold
training matrix are themselves out-of-fold.  The matching validation matrix is
created by a transformer fitted on the complete outer-fold training part.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


EPS = 1e-10
RDA_LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
PPCA_COMPONENTS = (16, 32, 64, 96)


def _as_array(X) -> np.ndarray:
    return np.asarray(X, dtype=np.float64)


def _safe_auc(y, score) -> float:
    score = np.nan_to_num(np.asarray(score), nan=0.0, posinf=1e12, neginf=-1e12)
    return roc_auc_score(y, score) if np.ptp(score) > 0 else 0.5


def _precision_from_covariance(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(covariance)
    floor = max(float(values.max()) * 1e-10, EPS)
    values = np.maximum(values, floor)
    return (vectors * (1.0 / values)) @ vectors.T


def _inv_sqrt_from_covariance(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(covariance)
    floor = max(float(values.max()) * 1e-10, EPS)
    values = np.maximum(values, floor)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def _squared_mahalanobis(X, mean, precision) -> np.ndarray:
    delta = X - mean
    return np.einsum("ij,jk,ik->i", delta, precision, delta, optimize=True)


class GeometryFeatures(BaseEstimator, TransformerMixin):
    """Aggregate, discriminant-geometry, RDA and optional density features."""

    def __init__(
        self,
        level: str = "enhanced",
        ppca_components: Iterable[int] = (),
        fa_components: Iterable[int] = (),
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.level = level
        self.ppca_components = tuple(ppca_components)
        self.fa_components = tuple(fa_components)
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _aggregates(X):
        total = X.sum(axis=1)
        mean = X.mean(axis=1)
        return {
            "total": total,
            "count_zeros": (X == 0).sum(axis=1),
            "relative_std": X.std(axis=1) / np.maximum(np.abs(mean), EPS),
            "max_share": X.max(axis=1) / np.maximum(np.abs(total), EPS),
            "max": X.max(axis=1),
            "log1p_sum": np.log1p(np.maximum(X, 0)).sum(axis=1),
        }

    def fit(self, X, y):
        X, y = _as_array(X), np.asarray(y)
        X0, X1 = X[y == 0], X[y == 1]
        self.mu0_, self.mu1_ = X0.mean(0), X1.mean(0)
        self.mid_ = 0.5 * (self.mu0_ + self.mu1_)
        self.w_ = self.mu1_ - self.mu0_
        self.u_ = self.w_ / (np.linalg.norm(self.w_) + EPS)

        self.lw0_, self.lw1_, self.lwp_ = LedoitWolf().fit(X0), LedoitWolf().fit(X1), LedoitWolf().fit(X)
        self.prec0_, self.prec1_ = self.lw0_.precision_, self.lw1_.precision_
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
        try:
            self.qda_ = QuadraticDiscriminantAnalysis(reg_param=0.05).fit(X, y)
        except np.linalg.LinAlgError:
            # Small smoke-test folds can have n_class <= d. The real data do not,
            # but this Ledoit-Wolf fallback keeps the transformer well-defined.
            self.qda_ = None
        self.class_log_prior_ = np.log(np.bincount(y.astype(int), minlength=2) / len(y) + EPS)
        self.logdet0_ = np.linalg.slogdet(self.lw0_.covariance_)[1]
        self.logdet1_ = np.linalg.slogdet(self.lw1_.covariance_)[1]
        legacy_n = min(15, X0.shape[0] - 1, X1.shape[0] - 1, X.shape[1])
        self.legacy_pca0_ = PCA(legacy_n, random_state=self.random_state).fit(X0)
        self.legacy_pca1_ = PCA(legacy_n, random_state=self.random_state).fit(X1)
        k = min(15, len(X0), len(X1))
        self.legacy_k_ = k
        self.nn0_ = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X0)
        self.nn1_ = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X1)
        self.nn_all_ = NearestNeighbors(n_neighbors=min(k + 1, len(X)), n_jobs=-1).fit(X)
        self.fit_y_ = y
        self.rda_precisions_ = []
        for lam in RDA_LAMBDAS:
            c0 = (1 - lam) * self.lwp_.covariance_ + lam * self.lw0_.covariance_
            c1 = (1 - lam) * self.lwp_.covariance_ + lam * self.lw1_.covariance_
            self.rda_precisions_.append((_precision_from_covariance(c0), _precision_from_covariance(c1)))

        self.whitener_ = _inv_sqrt_from_covariance(self.lwp_.covariance_)
        self.fisher_direction_ = self.whitener_ @ self.w_
        self.fisher_direction_ /= np.linalg.norm(self.fisher_direction_) + EPS

        self.ppca_models_ = []
        for n in self.ppca_components:
            n0, n1 = min(n, X0.shape[0] - 1, X.shape[1]), min(n, X1.shape[0] - 1, X.shape[1])
            self.ppca_models_.append((n, PCA(n_components=n0, svd_solver="randomized", random_state=self.random_state).fit(X0),
                                      PCA(n_components=n1, svd_solver="randomized", random_state=self.random_state).fit(X1)))
        self.fa_models_ = []
        for n in self.fa_components:
            nc = min(n, X.shape[1], X0.shape[0] - 1, X1.shape[0] - 1)
            fa0 = FactorAnalysis(n_components=nc, random_state=self.random_state, max_iter=500).fit(X0)
            fa1 = FactorAnalysis(n_components=nc, random_state=self.random_state, max_iter=500).fit(X1)
            self.fa_models_.append((n, fa0, fa1))
        return self

    @staticmethod
    def _axis_features(Z, unit, prefix):
        axis = Z @ unit
        norm2 = np.einsum("ij,ij->i", Z, Z)
        orth2 = np.maximum(norm2 - axis**2, 0.0)
        denom_dim = max(Z.shape[1] - 1, 1)
        return {
            f"{prefix}_axis": axis,
            f"{prefix}_orth": np.sqrt(orth2),
            f"{prefix}_cos": axis / (np.sqrt(norm2) + EPS),
            f"{prefix}_axis_snr": axis / (np.sqrt(orth2 / denom_dim) + EPS),
        }

    def transform(self, X):
        X = _as_array(X)
        out = self._aggregates(X)
        d0, d1 = np.linalg.norm(X - self.mu0_, axis=1), np.linalg.norm(X - self.mu1_, axis=1)
        proj = X @ self.w_
        m0 = _squared_mahalanobis(X, self.mu0_, self.prec0_)
        m1 = _squared_mahalanobis(X, self.mu1_, self.prec1_)
        qda_score = self.qda_.decision_function(X) if self.qda_ is not None else (
            -0.5 * (m1 + self.logdet1_) + self.class_log_prior_[1]
            +0.5 * (m0 + self.logdet0_) - self.class_log_prior_[0]
        )
        out.update({
            "d_euclid_diff": d0 - d1,
            "proj_w": proj,
            "cos_w": proj / ((np.linalg.norm(X, axis=1) + EPS) * (np.linalg.norm(self.w_) + EPS)),
            "lda_score": self.lda_.decision_function(X),
            "qda_score": qda_score,
        })
        out.update({"d_mahal_0": np.sqrt(np.maximum(m0, 0)), "d_mahal_1": np.sqrt(np.maximum(m1, 0)),
                    "d_mahal_diff": np.sqrt(np.maximum(m0, 0)) - np.sqrt(np.maximum(m1, 0))})
        rec0 = self.legacy_pca0_.inverse_transform(self.legacy_pca0_.transform(X))
        rec1 = self.legacy_pca1_.inverse_transform(self.legacy_pca1_.transform(X))
        idx = self.nn_all_.kneighbors(X, return_distance=False)
        out.update({
            "pca_subspace_diff": np.linalg.norm(X - rec0, axis=1) - np.linalg.norm(X - rec1, axis=1),
            "knn_dist_diff_k15": self.nn0_.kneighbors(X)[0].mean(1) - self.nn1_.kneighbors(X)[0].mean(1),
            "knn_prob_class1_k15": np.mean(self.fit_y_[idx] == 1, axis=1),
        })
        if self.level != "baseline":
            out.update({
                "mahal_sq_0": m0,
                "mahal_sq_1": m1,
                "mahal_sq_diff": m0 - m1,
                "mahal_norm_diff": (m0 - m1) / (m0 + m1 + EPS),
                "mahal_log_ratio": np.log(m0 + EPS) - np.log(m1 + EPS),
            })
            for lam, (p0, p1) in zip(RDA_LAMBDAS, self.rda_precisions_):
                r0 = _squared_mahalanobis(X, self.mu0_, p0)
                r1 = _squared_mahalanobis(X, self.mu1_, p1)
                tag = str(lam).replace(".", "p")
                out[f"rda_diff_{tag}"] = r0 - r1
                out[f"rda_norm_diff_{tag}"] = (r0 - r1) / (r0 + r1 + EPS)
            out.update(self._axis_features(X - self.mid_, self.u_, "mid"))
            whitened = (X - self.mid_) @ self.whitener_
            out.update(self._axis_features(whitened, self.fisher_direction_, "fisher"))
        for n, p0, p1 in self.ppca_models_:
            lp0, lp1 = p0.score_samples(X), p1.score_samples(X)
            out[f"ppca_logp0_{n}"] = lp0
            out[f"ppca_logp1_{n}"] = lp1
            out[f"ppca_llr_{n}"] = lp1 - lp0
        for n, f0, f1 in self.fa_models_:
            lp0, lp1 = f0.score_samples(X), f1.score_samples(X)
            out[f"fa_logp0_{n}"] = lp0
            out[f"fa_logp1_{n}"] = lp1
            out[f"fa_llr_{n}"] = lp1 - lp0
        return pd.DataFrame(out).replace([np.inf, -np.inf], 0).fillna(0)

    def fit_transform(self, X, y, **fit_params):
        X, y = _as_array(X), np.asarray(y)
        self.fit(X, y)  # state used later by transform(validation/test)
        cv = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        parts = []
        for tr, va in cv.split(X, y):
            inner = GeometryFeatures(self.level, self.ppca_components, self.fa_components,
                                     self.n_splits, self.random_state).fit(X[tr], y[tr])
            fold = inner.transform(X[va])
            fold.index = va
            parts.append(fold)
        return pd.concat(parts).sort_index().reset_index(drop=True)


class CoordinateLikelihoodFeatures(BaseEstimator, TransformerMixin):
    """Zero-aware quantile WoE with fold-local monotonic smoothing/ranking."""

    def __init__(self, n_bins=8, alpha=5.0, include_monotonic=True, include_groups=True,
                 n_splits=5, random_state=42):
        self.n_bins, self.alpha = n_bins, alpha
        self.include_monotonic, self.include_groups = include_monotonic, include_groups
        self.n_splits, self.random_state = n_splits, random_state

    def fit(self, X, y):
        X, y = _as_array(X), np.asarray(y)
        self.models_, strengths = [], np.zeros(X.shape[1])
        n0, n1 = (y == 0).sum(), (y == 1).sum()
        for j in range(X.shape[1]):
            x = X[:, j]
            positive = x[x > 0]
            edges = np.unique(np.quantile(positive, np.linspace(0, 1, self.n_bins + 1)[1:-1])) if positive.size else np.array([])
            ids = np.where(x == 0, 0, 1 + np.searchsorted(edges, x, side="right"))
            B = len(edges) + 2
            c0 = np.bincount(ids[y == 0], minlength=B)
            c1 = np.bincount(ids[y == 1], minlength=B)
            score = np.log((c1 + self.alpha) / (n1 + self.alpha * B)) - np.log((c0 + self.alpha) / (n0 + self.alpha * B))
            centers = np.array([np.median(x[ids == b]) if np.any(ids == b) else float(b) for b in range(B)])
            auc = _safe_auc(y, x)
            iso = IsotonicRegression(increasing=auc >= 0.5, out_of_bounds="clip")
            monotone = iso.fit_transform(centers, score, sample_weight=c0 + c1)
            self.models_.append((edges, score, monotone))
            strengths[j] = abs(auc - 0.5)
        self.ranking_ = np.argsort(strengths)[::-1]
        return self

    def transform(self, X):
        X = _as_array(X)
        raw, mono = np.empty_like(X), np.empty_like(X)
        for j, (edges, score, monotone) in enumerate(self.models_):
            ids = np.where(X[:, j] == 0, 0, 1 + np.searchsorted(edges, X[:, j], side="right"))
            raw[:, j], mono[:, j] = score[ids], monotone[ids]
        out = {"coord_llr_total": raw.sum(1)}
        if self.include_monotonic:
            out["coord_llr_monotonic_total"] = mono.sum(1)
        if self.include_groups:
            cuts = ((0, 64, "top64"), (64, 128, "64_128"), (128, 256, "128_256"),
                    (256, X.shape[1], "256_512"))
            for lo, hi, name in cuts:
                idx = self.ranking_[lo:min(hi, X.shape[1])]
                out[f"coord_llr_{name}"] = raw[:, idx].sum(1)
                if self.include_monotonic:
                    out[f"coord_llr_monotonic_{name}"] = mono[:, idx].sum(1)
        return pd.DataFrame(out).replace([np.inf, -np.inf], 0).fillna(0)

    def fit_transform(self, X, y, **fit_params):
        X, y = _as_array(X), np.asarray(y)
        self.fit(X, y)
        cv = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        parts = []
        for tr, va in cv.split(X, y):
            inner = CoordinateLikelihoodFeatures(self.n_bins, self.alpha, self.include_monotonic,
                                                 self.include_groups, self.n_splits, self.random_state).fit(X[tr], y[tr])
            fold = inner.transform(X[va]); fold.index = va; parts.append(fold)
        return pd.concat(parts).sort_index().reset_index(drop=True)


class LeakageSafeFeatureBuilder:
    """Compose feature families while keeping their fitted full-train state."""

    def __init__(self, config: str, seed: int = 42, n_splits: int = 5, n_bins: int = 8, alpha: float = 5.0):
        self.config, self.seed, self.n_splits, self.n_bins, self.alpha = config, seed, n_splits, n_bins, alpha

    def _make(self):
        ppca = PPCA_COMPONENTS if self.config in {"ppca", "best_ppca"} else ()
        fa = (16, 32) if self.config == "fa" else ()
        level = "baseline" if self.config == "baseline" else "enhanced"
        geom = GeometryFeatures(level, ppca, fa, self.n_splits, self.seed)
        coord = None
        if self.config in {"coord", "coord_monotonic", "best_coord"}:
            coord = CoordinateLikelihoodFeatures(self.n_bins, self.alpha,
                                                 include_monotonic=self.config != "coord",
                                                 include_groups=self.config != "coord",
                                                 n_splits=self.n_splits, random_state=self.seed)
        return geom, coord

    def fit_transform(self, X, y):
        Xlog = np.log1p(_as_array(X))
        self.geom_, self.coord_ = self._make()
        geom = self.geom_.fit_transform(Xlog, y)
        density = self.coord_.fit_transform(Xlog, y) if self.coord_ is not None else pd.DataFrame(index=geom.index)
        return Xlog, geom.to_numpy(), density.to_numpy(), list(geom.columns), list(density.columns)

    def transform(self, X):
        Xlog = np.log1p(_as_array(X))
        geom = self.geom_.transform(Xlog)
        density = self.coord_.transform(Xlog) if self.coord_ is not None else pd.DataFrame(index=geom.index)
        return Xlog, geom.to_numpy(), density.to_numpy(), list(geom.columns), list(density.columns)


@dataclass
class FoldData:
    raw_train: np.ndarray
    raw_valid: np.ndarray
    geom_train: np.ndarray
    geom_valid: np.ndarray
    density_train: np.ndarray
    density_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray


def _scale_blocks(raw_tr, raw_va, geom_tr, geom_va, den_tr, den_va):
    rs, gs = StandardScaler(), StandardScaler()
    raw_tr, raw_va = rs.fit_transform(raw_tr), rs.transform(raw_va)
    geom_tr, geom_va = gs.fit_transform(geom_tr), gs.transform(geom_va)
    if den_tr.shape[1]:
        ds = StandardScaler()
        den_tr, den_va = ds.fit_transform(den_tr), ds.transform(den_va)
    return raw_tr, raw_va, geom_tr, geom_va, den_tr, den_va


def cache_outer_folds(X, y, config, seed=42, n_splits=5, n_bins=8, alpha=5.0):
    """Expensive operation: build all target-derived features once per outer fold."""
    X, y = _as_array(X), np.asarray(y)
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_no, (tr, va) in enumerate(cv.split(X, y), 1):
        print(f"  cache {config}: outer fold {fold_no}/{n_splits}")
        builder = LeakageSafeFeatureBuilder(config, seed + fold_no, n_splits, n_bins, alpha)
        rtr, gtr, dtr, _, _ = builder.fit_transform(X[tr], y[tr])
        rva, gva, dva, _, _ = builder.transform(X[va])
        blocks = _scale_blocks(rtr, rva, gtr, gva, dtr, dva)
        folds.append(FoldData(*blocks, y[tr], y[va]))
    return folds


def fit_validation_blocks(X_train, y_train, X_valid, config, seed=42, n_bins=8, alpha=5.0):
    builder = LeakageSafeFeatureBuilder(config, seed, 5, n_bins, alpha)
    rtr, gtr, dtr, geom_names, density_names = builder.fit_transform(X_train, y_train)
    rva, gva, dva, _, _ = builder.transform(X_valid)
    blocks = _scale_blocks(rtr, rva, gtr, gva, dtr, dva)
    return builder, blocks, geom_names, density_names


def _kernel(fold: FoldData, params, train=True):
    rr = fold.raw_train
    rv = fold.raw_train if train else fold.raw_valid
    gg = np.column_stack([fold.geom_train, fold.density_train])
    gv = gg if train else np.column_stack([fold.geom_valid, fold.density_valid])
    Dr = cdist(rv, rr, metric="sqeuclidean")
    Dg = cdist(gv, gg, metric="sqeuclidean")
    Kr, Kg = np.exp(-params["gamma_raw"] * Dr), np.exp(-params["gamma_geom"] * Dg)
    if params["kernel_type"] == "product":
        return Kr * Kg
    return params["beta"] * Kr + (1 - params["beta"]) * Kg


def score_cached_folds(folds, params):
    scores = []
    for f in folds:
        if params["kernel_type"] == "rbf":
            Xtr = np.column_stack([f.raw_train, f.geom_train, f.density_train])
            Xva = np.column_stack([f.raw_valid, f.geom_valid, f.density_valid])
            model = SVC(C=params["C"], kernel="rbf", gamma=params["gamma"])
            model.fit(Xtr, f.y_train); pred = model.decision_function(Xva)
        else:
            model = SVC(C=params["C"], kernel="precomputed")
            model.fit(_kernel(f, params, True), f.y_train)
            pred = model.decision_function(_kernel(f, params, False))
        scores.append(roc_auc_score(f.y_valid, pred))
    return np.asarray(scores)


def optimize_svc(folds, kernel_type="rbf", n_trials=20, seed=42):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = {"kernel_type": kernel_type, "C": trial.suggest_float("C", 0.3, 30, log=True)}
        if kernel_type == "rbf":
            p["gamma"] = trial.suggest_float("gamma", 1e-5, 3e-2, log=True)
        else:
            p["gamma_raw"] = trial.suggest_float("gamma_raw", 1e-4, 3e-2, log=True)
            p["gamma_geom"] = trial.suggest_float("gamma_geom", 1e-3, 1.0, log=True)
            if kernel_type == "additive": p["beta"] = trial.suggest_float("beta", 0.1, 0.95)
        aucs = score_cached_folds(folds, p)
        trial.set_user_attr("std", float(aucs.std()))
        return float(aucs.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    params = {"kernel_type": kernel_type, **study.best_params}
    return study, params, score_cached_folds(folds, params)


def validation_auc(blocks, y_train, y_valid, params):
    rtr, rva, gtr, gva, dtr, dva = blocks
    fold = FoldData(rtr, rva, gtr, gva, dtr, dva, np.asarray(y_train), np.asarray(y_valid))
    if params["kernel_type"] == "rbf":
        model = SVC(C=params["C"], kernel="rbf", gamma=params["gamma"])
        model.fit(np.column_stack([rtr, gtr, dtr]), y_train)
        pred = model.decision_function(np.column_stack([rva, gva, dva]))
    else:
        model = SVC(C=params["C"], kernel="precomputed")
        model.fit(_kernel(fold, params, True), y_train)
        pred = model.decision_function(_kernel(fold, params, False))
    return roc_auc_score(y_valid, pred), pred


def feature_quality_table(X_train, y_train, X_valid, y_valid, seed=42):
    feat = GeometryFeatures("enhanced", PPCA_COMPONENTS, (), 5, seed)
    oof = feat.fit_transform(np.log1p(_as_array(X_train)), y_train)
    valid = feat.transform(np.log1p(_as_array(X_valid)))
    rows = []
    for col in oof:
        rows.append({"feature": col, "OOF Spearman": spearmanr(oof[col], y_train).statistic,
                     "OOF single-feature ROC-AUC": _safe_auc(y_train, oof[col]),
                     "validation single-feature ROC-AUC": _safe_auc(y_valid, valid[col])})
    result = pd.DataFrame(rows)
    result["OOF -> validation gap"] = result["validation single-feature ROC-AUC"] - result["OOF single-feature ROC-AUC"]
    return result.sort_values("OOF single-feature ROC-AUC", ascending=False), feat


def experiment_row(name, cv_scores, val_auc, raw_count, geom_count, params, fit_time):
    return {
        "experiment": name, "CV_mean": np.mean(cv_scores), "CV_std": np.std(cv_scores),
        "validation_auc": val_auc, "n_features_raw": raw_count, "n_features_geometry": geom_count,
        "C": params.get("C", np.nan), "kernel_type": params["kernel_type"],
        "gamma_raw": params.get("gamma_raw", params.get("gamma", np.nan)),
        "gamma_geom": params.get("gamma_geom", np.nan), "beta": params.get("beta", np.nan),
        "fit_time": fit_time,
    }


def fit_final_and_predict(X_all, y_all, X_test, config, params, seed=42, n_bins=8, alpha=5.0):
    """Refit every supervised transformer on train+validation and return monotone [0,1] scores."""
    builder, blocks, _, _ = fit_validation_blocks(X_all, y_all, X_test, config, seed, n_bins, alpha)
    rtr, rte, gtr, gte, dtr, dte = blocks
    fold = FoldData(rtr, rte, gtr, gte, dtr, dte, np.asarray(y_all), np.zeros(len(X_test)))
    if params["kernel_type"] == "rbf":
        model = SVC(C=params["C"], kernel="rbf", gamma=params["gamma"])
        model.fit(np.column_stack([rtr, gtr, dtr]), y_all)
        decision = model.decision_function(np.column_stack([rte, gte, dte]))
    else:
        model = SVC(C=params["C"], kernel="precomputed")
        model.fit(_kernel(fold, params, True), y_all)
        decision = model.decision_function(_kernel(fold, params, False))
    return expit(decision), builder, model
