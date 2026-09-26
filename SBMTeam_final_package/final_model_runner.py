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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

    def fit_and_score_external(X_external):
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
        return scores

    # Keep validation and test transforms separate. Some engineered legacy
    # transformers are batch-sensitive; this exactly matches the two original
    # final notebooks and is required for 1e-6 submission identity.
    validation_component_scores = fit_and_score_external(X_validation)
    test_component_scores = fit_and_score_external(X_test)

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

    total_seconds = perf_counter() - started
    experiment_table = pd.DataFrame([
        {
            "model": "NEW5 (designated)",
            "feature_set": "sqrt(raw)+white geometry/density/aggregates/coord+knn; rank blend with PCA128 aux",
            "validation_ROC_AUC": roc_auc_score(y_validation, new5_validation),
            "runtime_seconds": total_seconds,
            "seed": SEED,
        },
        {
            "model": "Rescue T6",
            "feature_set": "Fine6 originals + corrected K0 whites + PCA128/aggregates aux rank blend",
            "validation_ROC_AUC": roc_auc_score(y_validation, rescue_validation),
            "runtime_seconds": total_seconds,
            "seed": SEED,
        },
    ])
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
