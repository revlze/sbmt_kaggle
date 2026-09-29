# SBMTeam — final Moodle package

## Team

- **Kaggle team:** SBMTeam
- **Peretiatko Vsevolod** (`sevaperetiatko`)
- **Platon Usachev** (`omnistudent`)

Both team members worked jointly on feature engineering, leakage-safe validation,
SVC model design, ensembling, experiment review and final submission selection.

## Designated submission

- **Designated Kaggle filename:** `submission_NEW5_sqrt_s100_aux20.csv`
- **Required package filename:** `final_submission.csv`
- `final_submission.csv` is an exact copy of the designated NEW5 predictions.
- **Second reproduced candidate:** `submission_rescue_corrected_best.csv`

Run `final_submission_notebook.ipynb` with **Restart and Run All**. It trains both
final models plus eight representative comparison baselines from the supplied
competition CSV files and regenerates all three output files. No internet,
cached folds, fitted models or hidden local files are used.

## Experiment summary

The executable notebook prints a measured ten-model comparison table containing
classic trees and boosting trained on the same leakage-safe engineered feature
blocks, a deliberately plain raw-only SVC baseline, engineered single SVC models,
and both final ensembles. The frozen selection metrics from the original leakage-safe 25-fold
experiment were:

| Model | Feature set | CV ROC-AUC | Validation ROC-AUC | Seed |
|---|---|---:|---:|---:|
| NEW5 (designated) | S100 sqrt/raw-derived engineered kernel + 20% PCA128 auxiliary rank blend | 0.934295 | 0.954936 | 1027309 |
| Rescue T6 | 40% Fine6 + 40% corrected K0 + 20% PCA128 auxiliary rank blend | 0.933561 | 0.955179 | 1027309 |

Runtime is measured and displayed by the notebook because it depends on the
Kaggle CPU host. A clean local CPU run completed in **94.6 seconds**. The
notebook enforces the 15-minute total-runtime requirement.

## Files

- `final_submission_notebook.ipynb` — executable entry point.
- `final_model_runner.py` — frozen training, prediction and audit pipeline.
- `EXPERIMENTS.md` — concise history of the attempted model families and features.
- `svc_experiments.py` — supervised feature transformers.
- `svc_multikernel_experiments.py` — feature-block construction utilities.
- `svc_final_rescue_experiments.py` — whitening and additive-kernel utilities.
- `requirements.txt` — versions used to generate the designated predictions.
- `final_submission.csv` — designated predictions.
- `submission_NEW5_sqrt_s100_aux20.csv` — same designated predictions.
- `submission_rescue_corrected_best.csv` — second final candidate.

Competition data are intentionally not bundled.

## Versions used

- Python 3.14.2
- NumPy 2.5.3
- pandas 3.0.6
- SciPy 1.18.1
- scikit-learn 1.9.1
- joblib 1.6.0
