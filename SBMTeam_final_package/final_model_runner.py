"""Offline, deterministic reproduction of SBMTeam's two final SVC submissions."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial.distance import cdist
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import svc_multikernel_experiments as legacy
import svc_final_rescue_experiments as rescue


SEED = 0xFACED
COORD_N_BINS = 8
COORD_ALPHA = 5.0
PCA_COMPONENTS = 128

FINAL_REPRESENTATIONS = {
    "raw": "raw_orig",
    "geometry": "geometry_white16",
    "density": "density_white5",
    "coord": "coord_white4",
    "aggregates": "aggregates_white4",
    "knn": "knn_orig",
    "pca128": "pca128_orig",
}
FINAL_GAMMAS = {
    "raw": 0.002496,
    "geometry": 0.0034171616918486107,
    "density": 0.026340695648809757,
    "coord": 0.02582090408371943,
    "aggregates": 0.0368650778749846,
    "knn": 0.04697254731412588,
    "pca128": 0.0027678574834286906,
}
LEGACY_REPRESENTATIONS = {
    block: f"{block}_orig" for block in
    ("raw", "geometry", "density", "aggregates", "knn", "coord", "pca128")
}
LEGACY_FINE6_WEIGHTS = {
    "raw": 0.68, "geometry": 0.10, "aggregates": 0.05,
    "density": 0.10, "knn": 0.05, "coord": 0.02, "pca128": 0.00,
}
LEGACY_FINE6_GAMMAS = {
    "raw": 0.002496, "geometry": 0.012772,
    "aggregates": 0.073837, "density": 0.023941,
    "knn": 0.187890, "coord": 0.055470, "pca128": 0.001384,
}
K0_WEIGHTS = dict(LEGACY_FINE6_WEIGHTS)
AUX_WEIGHTS = {"pca128": 0.75, "aggregates": 0.25}
S100_WEIGHTS = {
    "raw_sqrt": 0.68, "geometry": 0.10, "density": 0.10,
    "aggregates": 0.05, "knn": 0.05, "coord": 0.02,
}
LEGACY_FINE6_C = 6.0
K0_C = 3.0
AUX_C = 30.0
S100_C = 3.0
SQRT_GAMMA = 0.002744811743389125
TREE_FEATURE_REPRESENTATIONS = (
    "raw_orig", "geometry_white16", "density_white5",
    "aggregates_white4", "coord_white4", "knn_orig", "pca128_orig",
)

MODEL_CONFIGURATIONS = [
    {
        "model": "Дерево решений",
        "role": "Простой интерпретируемый baseline: показывает, сколько структуры можно извлечь одним регуляризованным деревом без ансамблирования.",
        "parameters": "max_depth=5, min_samples_leaf=20",
        "feature_set": "raw + leakage-safe whitened geometry, density, aggregates, coord, kNN и PCA128",
    },
    {
        "model": "Случайный лес",
        "role": "Bagging-baseline для нелинейных разбиений; снижает дисперсию одиночного дерева и проверяет взаимодействия engineered-признаков.",
        "parameters": "n_estimators=1000, max_depth=8, min_samples_leaf=40, min_samples_split=80, max_features=sqrt",
        "feature_set": "raw + leakage-safe whitened geometry, density, aggregates, coord, kNN и PCA128",
    },
    {
        "model": "Градиентный бустинг (GBDT)",
        "role": "Последовательный ансамбль неглубоких деревьев; служит сильным контрастом kernel-методу на том же расширенном наборе признаков.",
        "parameters": "n_estimators=2000, learning_rate=0.01, max_depth=3, min_samples_leaf=80, subsample=0.7",
        "feature_set": "raw + leakage-safe whitened geometry, density, aggregates, coord, kNN и PCA128",
    },
    {
        "model": "Простой raw RBF SVC",
        "role": "Контрольная SVC без engineered-блоков: отделяет эффект самого RBF-классификатора от эффекта разработанных представлений.",
        "parameters": "log1p, StandardScaler, C=10, gamma=scale",
        "feature_set": "только 512 исходных признаков",
    },
    {
        "model": "Fine6 SVC",
        "role": "Исторический multi-kernel anchor, с которым сравнивались последующие исправления представлений и prediction blending.",
        "parameters": "C=6; additive weights raw=.68, geometry=.10, density=.10, aggregates=.05, knn=.05, coord=.02",
        "feature_set": "исходный raw и original engineered-блоки",
    },
    {
        "model": "Corrected K0 SVC",
        "role": "Одиночная основная SVC после исправления whitening-представлений и gamma; показывает качество engineered-модели без блендинга.",
        "parameters": "C=3; Fine6 weights with corrected frozen block gammas",
        "feature_set": "raw + whitened geometry/density/aggregates/coord + kNN",
    },
    {
        "model": "S100 SVC",
        "role": "Главная одиночная SVC финального решения: использует sqrt-метрику raw и исправленные engineered-блоки, но ещё без auxiliary blend.",
        "parameters": "C=3; sqrt_raw=.68, geometry=.10, density=.10, aggregates=.05, knn=.05, coord=.02",
        "feature_set": "sqrt(raw) + исправленные engineered-блоки",
    },
    {
        "model": "Вспомогательный clean_aux SVC",
        "role": "Специально отличающаяся дополнительная модель: слабее основной отдельно, но даёт полезный complementary ranking для ансамбля.",
        "parameters": "C=30; pca128=.75, aggregates_white4=.25",
        "feature_set": "PCA128 + whitened aggregates",
    },
    {
        "model": "Rescue T6",
        "role": "Вторая финальная модель: консервативный rank-blend исторического Fine6, corrected K0 и complementary auxiliary SVC.",
        "parameters": "rank blend: Fine6=.40, corrected K0=.40, clean_aux=.20",
        "feature_set": "три разных SVC-представления",
    },
    {
        "model": "NEW5 (финальная)",
        "role": "Основная designated-модель: сочетает сильную sqrt-raw S100 и отличающийся PCA128 auxiliary ranking.",
        "parameters": "rank blend: S100=.80, clean_aux=.20",
        "feature_set": "sqrt/engineered SVC + PCA128 auxiliary SVC",
    },
]


def set_all_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def locate_data_dir():
    candidates = [
        Path("data"),
        Path("../data"),
        Path("../input/week-03-hidden-pairs"),
        Path("../input/competitions/week-03-hidden-pairs"),
    ]
    required = ("train.csv", "validation.csv", "test.csv", "sample_submission.csv")
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate
    raise FileNotFoundError(
        "Competition files not found. Expected data/ or "
        "../input/week-03-hidden-pairs/ with train.csv, validation.csv, "
        "test.csv and sample_submission.csv."
    )


def _fit_s100(bundle, X_train, X_external, y_train):
    train_views = bundle["train_views"]
    external_views = bundle["external_views"]
    scaler = StandardScaler()
    sqrt_train = scaler.fit_transform(np.sqrt(X_train))
    sqrt_external = scaler.transform(np.sqrt(X_external))
    Dtr = cdist(sqrt_train, sqrt_train, metric="sqeuclidean")
    Dext = cdist(sqrt_external, sqrt_train, metric="sqeuclidean")
    Ktr = (np.float32(S100_WEIGHTS["raw_sqrt"])
           * np.exp(-SQRT_GAMMA * Dtr).astype(np.float32))
    Kext = (np.float32(S100_WEIGHTS["raw_sqrt"])
            * np.exp(-SQRT_GAMMA * Dext).astype(np.float32))
    del Dtr, Dext, sqrt_train, sqrt_external

    for block in ("geometry", "density", "aggregates", "knn", "coord"):
        representation = FINAL_REPRESENTATIONS[block]
        A = np.asarray(train_views[representation], dtype=np.float32)
        B = np.asarray(external_views[representation], dtype=np.float32)
        Dtr = cdist(A, A, metric="sqeuclidean")
        Dext = cdist(B, A, metric="sqeuclidean")
        Ktr += (np.float32(S100_WEIGHTS[block])
                * np.exp(-FINAL_GAMMAS[block] * Dtr).astype(np.float32))
        Kext += (np.float32(S100_WEIGHTS[block])
                 * np.exp(-FINAL_GAMMAS[block] * Dext).astype(np.float32))
        del A, B, Dtr, Dext

    model = SVC(C=S100_C, kernel="precomputed")
    model.fit(Ktr, y_train)
    score = model.decision_function(Kext)
    del Ktr, Kext, model
    return score


def _rank_blend(component_scores, weights):
    blended = sum(
        float(weight) * legacy.percentile_rank(component_scores[name])
        for name, weight in weights.items()
    )
    return legacy.percentile_rank(blended)


def _submission(row_ids, target):
    frame = pd.DataFrame({
        "row_id": np.asarray(row_ids),
        "target": np.asarray(target, dtype=float),
    })
    assert np.array_equal(frame["row_id"].to_numpy(), np.asarray(row_ids))
    assert np.isfinite(frame["target"]).all()
    assert frame["target"].between(0.0, 1.0).all()
    return frame[["row_id", "target"]]


def _tree_feature_matrices(bundle):
    """Return the same leakage-safe expanded views for all three tree models."""
    train = np.column_stack([
        bundle["train_views"][name]
        for name in TREE_FEATURE_REPRESENTATIONS
    ])
    external = np.column_stack([
        bundle["external_views"][name]
        for name in TREE_FEATURE_REPRESENTATIONS
    ])
    return train, external


def _fit_comparison_baselines(X_train_raw, X_validation_raw,
                              X_train_tree, X_validation_tree,
                              y_train, y_validation):
    """Fit four representative baselines for the final comparison table."""
    models = {
        "Дерево решений": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=SEED),
        "Случайный лес": RandomForestClassifier(
            n_estimators=1000, max_depth=8, min_samples_leaf=40,
            min_samples_split=80, max_features="sqrt",
            random_state=SEED, n_jobs=-1),
        "Градиентный бустинг (GBDT)": GradientBoostingClassifier(
            n_estimators=2000, learning_rate=0.01, max_depth=3,
            min_samples_leaf=80, min_samples_split=150,
            subsample=0.7, max_features=0.2, random_state=SEED),
        "Простой raw RBF SVC": Pipeline([
            ("log1p", FunctionTransformer(
                np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
            ("svc", SVC(C=10.0, kernel="rbf", gamma="scale")),
        ]),
    }
    rows = []
    for name, model in models.items():
        if name == "Простой raw RBF SVC":
            model_X_train, model_X_validation = X_train_raw, X_validation_raw
        else:
            model_X_train, model_X_validation = X_train_tree, X_validation_tree
        started = perf_counter()
        model.fit(model_X_train, y_train)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(model_X_validation)[:, 1]
        else:
            score = model.decision_function(model_X_validation)
        elapsed = perf_counter() - started
        config = next(item for item in MODEL_CONFIGURATIONS
                      if item["model"] == name)
        rows.append({
            "model": name,
            "role": config["role"],
            "parameters": config["parameters"],
            "feature_set": config["feature_set"],
            "validation_ROC_AUC": roc_auc_score(y_validation, score),
            "runtime_seconds": elapsed,
            "seed": SEED,
        })
    return rows


def run(output_dir="."):
    set_all_seeds()
    started = perf_counter()
    data_dir = locate_data_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    validation_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    features = [c for c in train_df.columns if c not in {"row_id", "target"}]
    assert len(features) == 512
    assert features == [c for c in validation_df.columns if c not in {"row_id", "target"}]
    assert features == [c for c in test_df.columns if c != "row_id"]
    assert np.array_equal(test_df.row_id.to_numpy(), sample_submission.row_id.to_numpy())

    X_train = train_df[features].to_numpy()
    y_train = train_df.target.to_numpy()
    X_validation = validation_df[features].to_numpy()
    y_validation = validation_df.target.to_numpy()
    X_test = test_df[features].to_numpy()
    preprocessing_seconds = 0.0
    rescue_seconds = 0.0
    s100_seconds = 0.0

    def fit_and_score_external(X_external, keep_bundle=False):
        nonlocal preprocessing_seconds, rescue_seconds, s100_seconds
        stage_started = perf_counter()
        bundle = rescue.build_external_views(
            X_train, y_train, X_external,
            builder_seed=SEED + legacy.INNER_SEED_OFFSET,
            n_bins=COORD_N_BINS,
            alpha=COORD_ALPHA,
            pca_components=PCA_COMPONENTS,
        )
        preprocessing_seconds += perf_counter() - stage_started

        stage_started = perf_counter()
        raw = rescue.fit_external_component_scores(
            bundle, y_train,
            FINAL_REPRESENTATIONS, FINAL_GAMMAS,
            K0_WEIGHTS, K0_C,
            AUX_WEIGHTS, AUX_C,
            LEGACY_FINE6_WEIGHTS, LEGACY_FINE6_GAMMAS, LEGACY_FINE6_C,
        )
        rescue_seconds += perf_counter() - stage_started
        scores = {
            "legacy_fine6": raw["legacy_fine6"],
            "corrected_clean_kernel": raw["best_clean_kernel"],
            "clean_aux": raw["clean_aux"],
        }

        stage_started = perf_counter()
        scores["s100"] = _fit_s100(bundle, X_train, X_external, y_train)
        s100_seconds += perf_counter() - stage_started
        return scores, (bundle if keep_bundle else None)

    # Keep validation and test transforms separate. Some engineered legacy
    # transformers are batch-sensitive; this exactly matches the two original
    # final notebooks and is required for 1e-6 submission identity.
    validation_component_scores, validation_bundle = fit_and_score_external(
        X_validation, keep_bundle=True)
    test_component_scores, _ = fit_and_score_external(X_test)

    new5_weights = {"s100": 0.80, "clean_aux": 0.20}
    rescue_weights = {
        "legacy_fine6": 0.40,
        "corrected_clean_kernel": 0.40,
        "clean_aux": 0.20,
    }
    new5_validation = _rank_blend(validation_component_scores, new5_weights)
    new5_test = _rank_blend(test_component_scores, new5_weights)
    rescue_validation = _rank_blend(
        validation_component_scores, rescue_weights)
    rescue_test = _rank_blend(test_component_scores, rescue_weights)

    new5_submission = _submission(test_df.row_id, new5_test)
    rescue_submission = _submission(test_df.row_id, rescue_test)
    new5_path = output_dir / "submission_NEW5_sqrt_s100_aux20.csv"
    rescue_path = output_dir / "submission_rescue_corrected_best.csv"
    final_path = output_dir / "final_submission.csv"
    new5_submission.to_csv(new5_path, index=False)
    rescue_submission.to_csv(rescue_path, index=False)
    new5_submission.to_csv(final_path, index=False)

    X_train_tree, X_validation_tree = _tree_feature_matrices(validation_bundle)
    comparison_rows = _fit_comparison_baselines(
        X_train, X_validation,
        X_train_tree, X_validation_tree,
        y_train, y_validation)
    configuration_by_name = {
        item["model"]: item for item in MODEL_CONFIGURATIONS}
    single_components = [
        ("Fine6 SVC", "legacy_fine6", rescue_seconds),
        ("Corrected K0 SVC", "corrected_clean_kernel", rescue_seconds),
        ("S100 SVC", "s100", s100_seconds),
        ("Вспомогательный clean_aux SVC", "clean_aux", rescue_seconds),
    ]
    for model_name, component_name, component_runtime in single_components:
        config = configuration_by_name[model_name]
        comparison_rows.append({
            "model": model_name,
            "role": config["role"],
            "parameters": config["parameters"],
            "feature_set": config["feature_set"],
            "validation_ROC_AUC": roc_auc_score(
                y_validation, validation_component_scores[component_name]),
            "runtime_seconds": component_runtime,
            "seed": SEED,
        })

    total_seconds = perf_counter() - started
    for model_name, validation_score in (
            ("Rescue T6", rescue_validation),
            ("NEW5 (финальная)", new5_validation)):
        config = configuration_by_name[model_name]
        comparison_rows.append({
            "model": model_name,
            "role": config["role"],
            "parameters": config["parameters"],
            "feature_set": config["feature_set"],
            "validation_ROC_AUC": roc_auc_score(
                y_validation, validation_score),
            "runtime_seconds": total_seconds,
            "seed": SEED,
        })
    order = {item["model"]: index
             for index, item in enumerate(MODEL_CONFIGURATIONS)}
    experiment_table = pd.DataFrame(comparison_rows)
    experiment_table["_order"] = experiment_table["model"].map(order)
    experiment_table = experiment_table.sort_values(
        "_order").drop(columns="_order").reset_index(drop=True)
    assert len(experiment_table) == 10
    experiment_table = experiment_table[[
        "model", "validation_ROC_AUC", "runtime_seconds", "seed"
    ]].rename(columns={
        "model": "Модель",
        "validation_ROC_AUC": "Validation ROC-AUC",
        "runtime_seconds": "Время, сек",
        "seed": "Seed",
    })
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    timings = {
        "preprocessing_seconds": preprocessing_seconds,
        "rescue_components_fit_predict_seconds": rescue_seconds,
        "s100_fit_predict_seconds": s100_seconds,
        "total_seconds": total_seconds,
    }
    assert total_seconds < 15 * 60, f"runtime limit exceeded: {total_seconds:.1f}s"
    return {
        "experiment_table": experiment_table,
        "versions": versions,
        "timings": timings,
        "paths": [new5_path, rescue_path, final_path],
        "new5_submission": new5_submission,
        "rescue_submission": rescue_submission,
    }


if __name__ == "__main__":
    result = run(".")
    print(result["experiment_table"].to_string(index=False))
    print(result["versions"])
    print(result["timings"])
    for path in result["paths"]:
        print(path.resolve())
