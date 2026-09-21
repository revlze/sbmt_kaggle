"""RandomizedSearchCV без Marimo. Сетки скопированы из draft_marimo.py.

Запуск: uv run python -u random_search.py --n-iter 100 --jobs 2
Быстрая проверка: uv run python -u random_search.py --smoke-test
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from lightgbm import LGBMClassifier

SEED = 42
ROOT = Path(__file__).resolve().parent

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        max_features=0.5,
        random_state=SEED,
        n_jobs=-1,
    ),
    'GBDT': GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=SEED,
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=15,
        min_child_samples=30,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=5,
        metric='auc',
        random_state=SEED,
        n_jobs=1,  # Один OpenMP-поток для стабильной работы в этом окружении.
        verbose=-1,
    ),
}

param_grids = {
    'Random Forest': {
        'n_estimators': [300, 500, 800, 1000, 1100, 1200, 1300, 1400, 1500, 1700, 1800, 2000],
        'max_depth': [5, 10, 15],
        'min_samples_split': [5, 10, 20, 50],
        'min_samples_leaf': [5, 10, 20],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
    },

    'GBDT': {
        'n_estimators': [2000, 2200, 2500],
        'learning_rate': [0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [5],
        'min_samples_leaf': [10, 20, 30],
        'max_features': ['sqrt', 0.1, 0.3, 0.5],
        'subsample': [0.6, 0.8],
    },
    'LightGBM': {
        'n_estimators': [1000, 1500, 2000, 2500],
        'learning_rate': [0.02, 0.03, 0.05],
        'max_depth': [4, 5, 6, 7],
        'num_leaves': [7, 15, 16],
        'min_child_samples': [20, 30, 50, 80],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'reg_alpha': [0.0, 0.01, 0.1],
        'reg_lambda': [0.0, 0.01, 0.1, 1.0],
    },
}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Нужно положительное целое число')
    return value


def jobs_count(value):
    value = int(value)
    if value != -1 and value < 1:
        raise argparse.ArgumentTypeError('Нужно -1 (все ядра) или положительное целое число')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-iter', type=positive_int, default=100)
    parser.add_argument('--folds', type=positive_int, default=5)
    parser.add_argument('--jobs', type=jobs_count, default=2)
    parser.add_argument('--models', nargs='+', choices=list(models), default=list(models))
    parser.add_argument('--smoke-test', action='store_true', help='Все выбранные модели: 1 вариант, 2 фолда, 5 деревьев')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'logs')
    args = parser.parse_args()
    if args.folds < 2:
        parser.error('--folds должен быть >= 2')

    train_df = pd.read_csv(ROOT / 'data/train.csv')
    val_df = pd.read_csv(ROOT / 'data/validation.csv')
    # Как в активной ячейке ноутбука: исходные признаки + сумма по строке.
    feature_columns = [c for c in train_df if c not in ('row_id', 'target', 'total')]
    X_train = train_df[feature_columns].copy()
    X_val = val_df[feature_columns].copy()
    X_train['total'] = X_train.sum(axis=1)
    X_val['total'] = X_val.sum(axis=1)
    y_train, y_val = train_df['target'], val_df['target']
    if X_train.isna().any().any() or X_val.isna().any().any():
        raise ValueError('В признаках есть пропуски')

    n_iter, folds = (1, 2) if args.smoke_test else (args.n_iter, args.folds)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    started_at = datetime.now().astimezone()
    prefix = 'smoke_search' if args.smoke_test else 'script_search'
    run_dir = args.output_dir / f'{prefix}_{started_at:%Y-%m-%d_%H-%M-%S_%f}'
    run_dir.mkdir(parents=True, exist_ok=False)
    results = []
    predictions = val_df[['row_id', 'target']].copy()
    metadata = {
        'run_started_at': started_at.isoformat(),
        'seed': SEED, 'n_iter': n_iter, 'folds': folds, 'jobs': args.jobs,
        'smoke_test': args.smoke_test, 'feature_columns': list(X_train.columns),
        'results': results,
    }
    (run_dir / 'results.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'train={X_train.shape}, val={X_val.shape}', flush=True)
    print(f'{len(args.models)} моделей × {n_iter} вариантов × {folds} фолдов', flush=True)
    print(f'Результаты: {run_dir.resolve()}', flush=True)

    for index, name in enumerate(args.models, 1):
        model = models[name]
        # Параллельны только CV-процессы, внутри каждой модели один поток.
        if 'n_jobs' in model.get_params():
            model.set_params(n_jobs=1)
        grid = dict(param_grids[name])
        if args.smoke_test:
            grid['n_estimators'] = [5]
        print(f'\n[{index}/{len(args.models)}] {name}', flush=True)
        search = RandomizedSearchCV(
            model, grid, n_iter=n_iter, cv=cv, scoring='roc_auc',
            random_state=SEED, verbose=10, n_jobs=args.jobs,
            pre_dispatch='n_jobs', refit=True, error_score='raise',
        )
        start = time.perf_counter()
        with joblib.parallel_config(backend='loky', inner_max_num_threads=1):
            search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        best_model = search.best_estimator_
        model_file = f'{index:02d}_model.joblib'
        joblib.dump(best_model, run_dir / model_file, compress=3)
        pd.DataFrame(search.cv_results_).to_csv(run_dir / f'{index:02d}_cv_results.csv', index=False)
        val_probs = best_model.predict_proba(X_val)[:, 1]
        predictions[name] = val_probs
        result = {
            'Model': name, 'Best Params': search.best_params_,
            'CV Score': float(search.best_score_),
            'CV STD': float(search.cv_results_['std_test_score'][search.best_index_]),
            'Train AUC': float(roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])),
            'Val AUC': float(roc_auc_score(y_val, val_probs)),
            'Search + Refit Time (s)': round(elapsed, 3),
            'Model File': model_file,
        }
        results.append(result)
        (run_dir / 'results.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        predictions.to_csv(run_dir / 'validation_predictions.csv', index=False)
        print(f"{name}: CV AUC={result['CV Score']:.5f}, Val AUC={result['Val AUC']:.5f}, {elapsed:.1f} с", flush=True)
        print(f'Сохранено: {run_dir / model_file}', flush=True)

    print('\n' + pd.DataFrame(results)[['Model', 'CV Score', 'Val AUC', 'Search + Refit Time (s)']].to_string(index=False))


if __name__ == '__main__':
    main()
