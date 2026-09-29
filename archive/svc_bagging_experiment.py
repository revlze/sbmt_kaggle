import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ансамбль SVC со случайными объектами и признаками

    Эксперимент по мотивам `main.ipynb`. `BaggingClassifier` обучает каждую SVC
    на bootstrap-выборке строк и случайном подмножестве исходных столбцов.
    Внутри каждого базового пайплайна `FullFeatureEngineering` из `main.ipynb`
    вычисляет 6 построчных статистик и 14 геометрических признаков **после
    отбора столбцов**. В SVC подаются также сами выбранные исходные признаки.
    Порядок: `log1p` → FullFeatureEngineering → StandardScaler → RBF SVC.
    Прогноз ансамбля — среднее вероятностей.

    В конфигурации `OBJECTS_PERCENT` и `FEATURES_PERCENT` задаются числами
    от 0 до 100: например, `80` означает 80% объектов в каждой bootstrap-выборке.

    Validation используется только для оценки. Никакие строки validation или
    test не участвуют в обучении. Полный запуск может занять несколько минут.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.svm import SVC

    return (
        BaggingClassifier,
        BaseEstimator,
        CalibratedClassifierCV,
        ClassifierMixin,
        FunctionTransformer,
        GaussianMixture,
        LedoitWolf,
        LinearDiscriminantAnalysis,
        NearestNeighbors,
        PCA,
        Path,
        Pipeline,
        QuadraticDiscriminantAnalysis,
        SVC,
        StandardScaler,
        StratifiedGroupKFold,
        StratifiedKFold,
        TransformerMixin,
        cross_val_score,
        mo,
        np,
        pd,
        roc_auc_score,
    )


@app.cell
def _(Path):
    SEED = 0xFACED  # Как в main.ipynb.
    DATA_DIR = Path("data")
    OBJECTS_PERCENT = 80
    FEATURES_PERCENT = 75

    if not 0 < OBJECTS_PERCENT <= 100 or not 0 < FEATURES_PERCENT <= 100:
        raise ValueError("Проценты объектов и признаков должны быть в диапазоне (0, 100]")
    if int(512 * FEATURES_PERCENT / 100) < 2:
        raise ValueError("Для FullFeatureEngineering нужны минимум два признака")
    
    ensemble_params = {
        "n_estimators": 50,
        "max_samples": OBJECTS_PERCENT / 100,
        "max_features": FEATURES_PERCENT / 100,
        "bootstrap": True,
        "bootstrap_features": False,
        "n_jobs": -1,
        "random_state": SEED,
    }
    svc_params = {"C": 10.0, "kernel": "rbf", "gamma": "scale"}
    feature_engineering_params = {
        "n_splits": 5,
        "k_neighbors": 15,
        "n_pca_components": 15,
        "gmm_pca_components": 96,
        "gmm_n_components": 3,
        "gmm_covariance_type": "tied",
        "gmm_reg_covar": 1e-2,
        "gmm_n_init": 5,
        "random_state": SEED,
    }
    CV_FOLDS = 5
    return (
        CV_FOLDS,
        DATA_DIR,
        FEATURES_PERCENT,
        OBJECTS_PERCENT,
        SEED,
        ensemble_params,
        feature_engineering_params,
        svc_params,
    )


@app.cell
def _(DATA_DIR, np, pd):
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

    feature_columns = [c for c in train_df if c not in ("row_id", "target")]
    assert len(feature_columns) == 512
    assert feature_columns == [c for c in val_df if c not in ("row_id", "target")]
    assert feature_columns == [c for c in test_df if c != "row_id"]
    for frame in (train_df, val_df, test_df):
        assert frame["row_id"].is_unique
        values = frame[feature_columns].to_numpy()
        assert np.isfinite(values).all() and (values >= 0).all()
    assert set(train_df["target"].unique()) == {0, 1}

    X_train = train_df[feature_columns]
    y_train = train_df["target"]
    X_val = val_df[feature_columns]
    y_val = val_df["target"]
    X_test = test_df[feature_columns]
    print(f"train={X_train.shape}, validation={X_val.shape}, test={X_test.shape}")
    return X_test, X_train, X_val, sample_submission, test_df, y_train, y_val


@app.cell
def _(
    BaseEstimator,
    GaussianMixture,
    LedoitWolf,
    LinearDiscriminantAnalysis,
    NearestNeighbors,
    PCA,
    QuadraticDiscriminantAnalysis,
    StandardScaler,
    StratifiedGroupKFold,
    TransformerMixin,
    np,
    pd,
):
    class FullFeatureEngineering(BaseEstimator, TransformerMixin):

        def __init__(
            self,
            n_splits=5,
            k_neighbors=15,
            n_pca_components=15,
            random_state=42,
            gmm_pca_components=96,
            gmm_n_components=3,
            gmm_covariance_type='tied',
            gmm_reg_covar=1e-2,
            gmm_n_init=5,
        ):
            self.n_splits = n_splits
            self.k_neighbors = k_neighbors
            self.n_pca_components = n_pca_components
            self.random_state = random_state
            self.gmm_pca_components = gmm_pca_components
            self.gmm_n_components = gmm_n_components
            self.gmm_covariance_type = gmm_covariance_type
            self.gmm_reg_covar = gmm_reg_covar
            self.gmm_n_init = gmm_n_init

        def _compute_unsupervised_aggs(self, X_df):
            vals = X_df.values
            total = vals.sum(axis=1)
            mean = vals.mean(axis=1)
            mean_safe = np.where(mean == 0, np.nan, mean)
            total_safe = np.where(total == 0, np.nan, total)

            aggs = {
                'total': total,
                'count_zeros': (vals == 0).sum(axis=1),
                'relative_std': vals.std(axis=1) / mean_safe,
                'max_share': vals.max(axis=1) / total_safe,
                'max': vals.max(axis=1),
                'log1p_sum': np.log1p(vals).sum(axis=1),
            }
            return pd.DataFrame(aggs, index=X_df.index).fillna(0)

        def _fit_gmm_geometry(self, X_mat, y_mat):
            # На вход уже подан log1p из Pipeline; scaler и PCA учим только на этом train/fold.
            scaler = StandardScaler().fit(X_mat)
            X_scaled = scaler.transform(X_mat)
            n_components = min(self.gmm_pca_components, *X_scaled.shape)
            pca = PCA(
                n_components=n_components, svd_solver='randomized',
                random_state=self.random_state,
            ).fit(X_scaled)
            X_pca = pca.transform(X_scaled)
            gmm_params = dict(
                n_components=self.gmm_n_components,
                covariance_type=self.gmm_covariance_type,
                reg_covar=self.gmm_reg_covar,
                n_init=self.gmm_n_init,
                random_state=self.random_state,
            )
            gmm_0 = GaussianMixture(**gmm_params).fit(X_pca[y_mat == 0])
            gmm_1 = GaussianMixture(**gmm_params).fit(X_pca[y_mat == 1])
            return scaler, pca, gmm_0, gmm_1

        @staticmethod
        def _gmm_scores(X_mat, scaler, pca, gmm_0, gmm_1):
            X_pca = pca.transform(scaler.transform(X_mat))
            logp_0 = gmm_0.score_samples(X_pca)
            logp_1 = gmm_1.score_samples(X_pca)
            return logp_0, logp_1, logp_1 - logp_0

        def fit(self, X, y):
            X_mat = np.asarray(X)
            y_mat = np.asarray(y)
            self.n_features_in_ = X_mat.shape[1]
            self.n_samples_in_ = X_mat.shape[0]

            X_tr0, X_tr1 = X_mat[y_mat == 0], X_mat[y_mat == 1]

            # 1. Центроиды
            self.mu0_ = X_tr0.mean(axis=0)
            self.mu1_ = X_tr1.mean(axis=0)
            self.w_ = self.mu1_ - self.mu0_
            w_norm = np.linalg.norm(self.w_)
            self.w_norm_ = w_norm if w_norm > 0 else 1e-8

            # 2. Дискриминантные модели (LDA & QDA)
            self.lda_ = LinearDiscriminantAnalysis(
                solver='lsqr', shrinkage='auto'
            ).fit(X_mat, y_mat)
            self.qda_ = QuadraticDiscriminantAnalysis(
                solver="eigen", shrinkage="auto", reg_param=0.05,
            ).fit(
                X_mat, y_mat
            )

            # 3. Ковариации (LedoitWolf)
            self.cov0_ = LedoitWolf().fit(X_tr0)
            self.cov1_ = LedoitWolf().fit(X_tr1)

            # 4. Подпространства (PCA)
            n_comp0 = min(self.n_pca_components, X_tr0.shape[0] - 1)
            n_comp1 = min(self.n_pca_components, X_tr1.shape[0] - 1)
            self.pca0_ = PCA(
                n_components=n_comp0, random_state=self.random_state
            ).fit(X_tr0)
            self.pca1_ = PCA(
                n_components=n_comp1, random_state=self.random_state
            ).fit(X_tr1)

            # 5. Локальная топология (KNN)
            self.nn0_ = NearestNeighbors(
                n_neighbors=self.k_neighbors, n_jobs=-1
            ).fit(X_tr0)
            self.nn1_ = NearestNeighbors(
                n_neighbors=self.k_neighbors, n_jobs=-1
            ).fit(X_tr1)
            self.nn_all_ = NearestNeighbors(
                n_neighbors=self.k_neighbors + 1, n_jobs=-1
            ).fit(X_mat)
            self.y_mat_ = y_mat

            # Полный train: используется только при transform(validation/test).
            self.gmm_scaler_, self.gmm_pca_, self.gmm_0_, self.gmm_1_ = (
                self._fit_gmm_geometry(X_mat, y_mat)
            )

            return self

        def fit_transform(self, X, y):
            X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
            X_mat = X_df.values
            y_mat = np.asarray(y)

            # Обучаем глобальные параметры на текущем датасете (Train или Train+Val)
            self.fit(X_mat, y_mat)

            # 1. Неразмеченные агрегаты
            df_aggs = self._compute_unsupervised_aggs(X_df)

            # 2. OOF Геометрия
            # Bootstrap создаёт точные копии строк. Держим их в одном OOF-фолде.
            groups = np.unique(X_mat, axis=0, return_inverse=True)[1]
            skf = StratifiedGroupKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )

            cols_geom = [
                'd_euclid_diff',
                'proj_w',
                'cos_w',
                'lda_score',
                'qda_score',
                'd_mahal_0',
                'd_mahal_1',
                'd_mahal_diff',
                'pca_subspace_diff',
                'knn_dist_diff_k15',
                'knn_prob_class1_k15',
                'gmm_logp_0',
                'gmm_logp_1',
                'gmm_llr',
            ]
            oof_geom = np.zeros((len(X_mat), len(cols_geom)))

            for train_idx, val_idx in skf.split(X_mat, y_mat, groups):
                X_tr, y_tr = X_mat[train_idx], y_mat[train_idx]
                X_va = X_mat[val_idx]
                X_tr0, X_tr1 = X_tr[y_tr == 0], X_tr[y_tr == 1]

                # Центроиды и Евклид
                mu0, mu1 = X_tr0.mean(axis=0), X_tr1.mean(axis=0)
                w = mu1 - mu0
                w_norm = np.linalg.norm(w) if np.linalg.norm(w) > 0 else 1e-8

                d0 = np.linalg.norm(X_va - mu0, axis=1)
                d1 = np.linalg.norm(X_va - mu1, axis=1)
                proj_w = X_va @ w
                x_norms = np.where(
                    np.linalg.norm(X_va, axis=1) == 0,
                    1e-8,
                    np.linalg.norm(X_va, axis=1),
                )
                cos_w = proj_w / (x_norms * w_norm)

                # LDA & QDA
                lda = LinearDiscriminantAnalysis(
                    solver='lsqr', shrinkage='auto'
                ).fit(X_tr, y_tr)
                qda = QuadraticDiscriminantAnalysis(
                    solver="eigen", shrinkage="auto", reg_param=0.05,
                ).fit(X_tr, y_tr)
                lda_s = lda.decision_function(X_va)
                qda_s = qda.decision_function(X_va)

                # Mahalanobis
                cov0, cov1 = LedoitWolf().fit(X_tr0), LedoitWolf().fit(X_tr1)
                dm0 = np.sqrt(cov0.mahalanobis(X_va))
                dm1 = np.sqrt(cov1.mahalanobis(X_va))

                # PCA Subspaces
                n_comp0 = min(self.n_pca_components, X_tr0.shape[0] - 1)
                n_comp1 = min(self.n_pca_components, X_tr1.shape[0] - 1)
                pca0 = PCA(
                    n_components=n_comp0, random_state=self.random_state
                ).fit(X_tr0)
                pca1 = PCA(
                    n_components=n_comp1, random_state=self.random_state
                ).fit(X_tr1)

                rec0 = pca0.inverse_transform(pca0.transform(X_va))
                rec1 = pca1.inverse_transform(pca1.transform(X_va))
                pca_d0 = np.linalg.norm(X_va - rec0, axis=1)
                pca_d1 = np.linalg.norm(X_va - rec1, axis=1)

                # KNN Топология
                nn0 = NearestNeighbors(
                    n_neighbors=self.k_neighbors, n_jobs=-1
                ).fit(X_tr0)
                nn1 = NearestNeighbors(
                    n_neighbors=self.k_neighbors, n_jobs=-1
                ).fit(X_tr1)
                knn_d0 = nn0.kneighbors(X_va)[0].mean(axis=1)
                knn_d1 = nn1.kneighbors(X_va)[0].mean(axis=1)

                nn_all = NearestNeighbors(
                    n_neighbors=self.k_neighbors + 1, n_jobs=-1
                ).fit(X_tr)
                indices = nn_all.kneighbors(X_va, return_distance=False)
                knn_prob1 = np.mean(y_tr[indices] == 1, axis=1)

                # GMM признаки для train строго OOF: scaler/PCA/GMM обучены без val_idx.
                gmm_scaler, gmm_pca, gmm_0, gmm_1 = self._fit_gmm_geometry(X_tr, y_tr)
                gmm_logp_0, gmm_logp_1, gmm_llr = self._gmm_scores(
                    X_va, gmm_scaler, gmm_pca, gmm_0, gmm_1
                )

                oof_geom[val_idx] = np.column_stack([
                    d0 - d1,
                    proj_w,
                    cos_w,
                    lda_s,
                    qda_s,
                    dm0,
                    dm1,
                    dm0 - dm1,
                    pca_d0 - pca_d1,
                    knn_d0 - knn_d1,
                    knn_prob1,
                    gmm_logp_0,
                    gmm_logp_1,
                    gmm_llr,
                ])

            df_geom = pd.DataFrame(oof_geom, columns=cols_geom, index=X_df.index)
            return np.column_stack((X_mat, df_aggs.to_numpy(), df_geom.to_numpy()))

        def transform(self, X):
            X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
            X_mat = X_df.values
            if X_mat.shape[1] != self.n_features_in_:
                raise ValueError("Число выбранных признаков отличается от обучения")

            # 1. Неразмеченные агрегаты
            df_aggs = self._compute_unsupervised_aggs(X_df)

            # 2. Применение глобально обученных трансформеров
            d0 = np.linalg.norm(X_mat - self.mu0_, axis=1)
            d1 = np.linalg.norm(X_mat - self.mu1_, axis=1)
            proj_w = X_mat @ self.w_
            x_norms = np.where(
                np.linalg.norm(X_mat, axis=1) == 0,
                1e-8,
                np.linalg.norm(X_mat, axis=1),
            )
            cos_w = proj_w / (x_norms * self.w_norm_)

            lda_s = self.lda_.decision_function(X_mat)
            qda_s = self.qda_.decision_function(X_mat)

            dm0 = np.sqrt(self.cov0_.mahalanobis(X_mat))
            dm1 = np.sqrt(self.cov1_.mahalanobis(X_mat))

            rec0 = self.pca0_.inverse_transform(self.pca0_.transform(X_mat))
            rec1 = self.pca1_.inverse_transform(self.pca1_.transform(X_mat))
            pca_d0 = np.linalg.norm(X_mat - rec0, axis=1)
            pca_d1 = np.linalg.norm(X_mat - rec1, axis=1)

            knn_d0 = self.nn0_.kneighbors(X_mat)[0].mean(axis=1)
            knn_d1 = self.nn1_.kneighbors(X_mat)[0].mean(axis=1)

            indices = self.nn_all_.kneighbors(X_mat, return_distance=False)
            knn_prob1 = np.mean(self.y_mat_[indices] == 1, axis=1)

            gmm_logp_0, gmm_logp_1, gmm_llr = self._gmm_scores(
                X_mat, self.gmm_scaler_, self.gmm_pca_, self.gmm_0_, self.gmm_1_
            )

            cols_geom = [
                'd_euclid_diff',
                'proj_w',
                'cos_w',
                'lda_score',
                'qda_score',
                'd_mahal_0',
                'd_mahal_1',
                'd_mahal_diff',
                'pca_subspace_diff',
                'knn_dist_diff_k15',
                'knn_prob_class1_k15',
                'gmm_logp_0',
                'gmm_logp_1',
                'gmm_llr',
            ]

            res_geom = np.column_stack([
                d0 - d1,
                proj_w,
                cos_w,
                lda_s,
                qda_s,
                dm0,
                dm1,
                dm0 - dm1,
                pca_d0 - pca_d1,
                knn_d0 - knn_d1,
                knn_prob1,
                gmm_logp_0,
                gmm_logp_1,
                gmm_llr,
            ])

            df_geom = pd.DataFrame(res_geom, columns=cols_geom, index=X_df.index)
            return np.column_stack((X_mat, df_aggs.to_numpy(), df_geom.to_numpy()))

    class DuplicateGroupedCV:
        """Не разносит копии bootstrap-строки между train и calibration."""

        def __init__(self, n_splits=3, random_state=42):
            self.n_splits = n_splits
            self.random_state = random_state

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

        def split(self, X, y, groups=None):
            X_mat = np.asarray(X)
            row_groups = np.unique(X_mat, axis=0, return_inverse=True)[1]
            cv = StratifiedGroupKFold(
                n_splits=self.n_splits, shuffle=True,
                random_state=self.random_state,
            )
            yield from cv.split(X_mat, y, row_groups)

    return DuplicateGroupedCV, FullFeatureEngineering


@app.cell
def _(BaseEstimator, CalibratedClassifierCV, ClassifierMixin, np):
    class CalibratedSVCNoWeights(ClassifierMixin, BaseEstimator):
        """Bagging передаёт выбранные строки, а не веса всему датасету."""

        def __init__(self, estimator, cv):
            self.estimator = estimator
            self.cv = cv

        def fit(self, X, y):
            self.n_features_in_ = X.shape[1]
            self.model_ = CalibratedClassifierCV(
                estimator=self.estimator, method="sigmoid",
                cv=self.cv, ensemble=False,
            ).fit(X, y)
            self.classes_ = self.model_.classes_
            return self

        def predict_proba(self, X):
            return self.model_.predict_proba(X)

        def predict(self, X):
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    return (CalibratedSVCNoWeights,)


@app.cell
def _(
    BaggingClassifier,
    CalibratedSVCNoWeights,
    DuplicateGroupedCV,
    FullFeatureEngineering,
    FunctionTransformer,
    Pipeline,
    SEED,
    SVC,
    StandardScaler,
    ensemble_params,
    feature_engineering_params,
    np,
    svc_params,
):
    base_pipeline = Pipeline([
        ("log1p", FunctionTransformer(np.log1p)),
        ("features", FullFeatureEngineering(**feature_engineering_params)),
        ("scaler", StandardScaler()),
        ("svc", SVC(**svc_params)),
    ])
    base_svc = CalibratedSVCNoWeights(
        estimator=base_pipeline,
        cv=DuplicateGroupedCV(n_splits=3, random_state=SEED),
    )
    ensemble = BaggingClassifier(estimator=base_svc, **ensemble_params)
    return (ensemble,)


@app.cell
def _(
    CV_FOLDS,
    SEED,
    StratifiedKFold,
    X_train,
    cross_val_score,
    ensemble,
    ensemble_params,
    np,
    y_train,
):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        ensemble, X_train, y_train, cv=cv, scoring="roc_auc",
        n_jobs=ensemble_params["n_jobs"], error_score="raise",
    )
    print("CV ROC-AUC по фолдам:", np.round(cv_scores, 5))
    print(f"CV ROC-AUC: {cv_scores.mean():.5f} ± {cv_scores.std():.5f}")
    return (cv_scores,)


@app.cell
def _(
    FEATURES_PERCENT,
    OBJECTS_PERCENT,
    X_train,
    X_val,
    cv_scores,
    ensemble,
    np,
    pd,
    roc_auc_score,
    y_train,
    y_val,
):
    ensemble.fit(X_train, y_train)
    val_prob = ensemble.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_prob)
    assert np.isfinite(val_prob).all() and ((0 <= val_prob) & (val_prob <= 1)).all()

    # Сверяем выборки и пересчёт признаков для каждой обученной SVC.
    feature_counts = [len(cols) for cols in ensemble.estimators_features_]
    assert all(count <= X_train.shape[1] for count in feature_counts)
    train_values = X_train.to_numpy()
    val_values = X_val.to_numpy()
    train_labels = y_train.to_numpy()
    for estimator, rows, columns in zip(
        ensemble.estimators_, ensemble.estimators_samples_,
        ensemble.estimators_features_,
    ):
        features = estimator.model_.calibrated_classifiers_[0].estimator.named_steps[
            "features"
        ]
        assert len(rows) == int(len(X_train) * OBJECTS_PERCENT / 100)
        assert len(columns) == int(X_train.shape[1] * FEATURES_PERCENT / 100)
        assert features.n_samples_in_ == len(rows)
        assert features.n_features_in_ == len(columns)

        # Центроиды геометрии должны быть обучены на выбранных объектах и столбцах.
        selected_train = np.log1p(train_values[np.ix_(rows, columns)])
        selected_labels = train_labels[rows]
        assert np.allclose(
            features.mu0_, selected_train[selected_labels == 0].mean(axis=0)
        )
        assert np.allclose(
            features.mu1_, selected_train[selected_labels == 1].mean(axis=0)
        )

        # Агрегат total и геометрическое расстояние пересчитываются на тех же столбцах.
        selected_val = np.log1p(val_values[:1, columns])
        engineered = features.transform(selected_val)
        width = len(columns)
        assert engineered.shape[1] == width + 20
        assert np.allclose(engineered[:, :width], selected_val)
        assert np.allclose(engineered[:, width], selected_val.sum(axis=1))
        distance_diff = (
            np.linalg.norm(selected_val - features.mu0_, axis=1)
            - np.linalg.norm(selected_val - features.mu1_, axis=1)
        )
        assert np.allclose(engineered[:, width + 6], distance_diff)
    print("Проверено: каждый элемент пересчитал агрегаты и геометрию по своей выборке")
    comparison = pd.DataFrame([{
        "Model": "Bagged RBF SVC",
        "CV ROC-AUC": cv_scores.mean(),
        "CV STD": cv_scores.std(),
        "Val ROC-AUC": val_auc,
        "Val - CV": val_auc - cv_scores.mean(),
        "Models": len(ensemble.estimators_),
        "Rows per model": len(ensemble.estimators_samples_[0]),
        "Raw features per model": feature_counts[0],
        "Total features per SVC": feature_counts[0] + 20,
    }])
    print(comparison.to_string(index=False))
    return


@app.cell
def _(Path):
    SAVE_SUBMISSION = False  # True: сохранить прогноз уже обученного ансамбля.
    SUBMISSION_PATH = Path("submission_svc_bagging.csv")
    return SAVE_SUBMISSION, SUBMISSION_PATH


@app.cell
def _(
    SAVE_SUBMISSION,
    SUBMISSION_PATH,
    X_test,
    ensemble,
    np,
    pd,
    sample_submission,
    test_df,
):
    if SAVE_SUBMISSION:
        assert sample_submission["row_id"].is_unique
        assert set(sample_submission["row_id"]) == set(test_df["row_id"])
        test_prob = ensemble.predict_proba(X_test)[:, 1]
        assert np.isfinite(test_prob).all() and ((0 <= test_prob) & (test_prob <= 1)).all()
        submission = sample_submission[["row_id"]].merge(
            pd.DataFrame({"row_id": test_df["row_id"], "target": test_prob}),
            on="row_id", how="left", sort=False, validate="one_to_one",
        )
        assert submission["target"].notna().all()
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Сохранено: {SUBMISSION_PATH.resolve()} | {submission.shape}")
    return


if __name__ == "__main__":
    app.run()
