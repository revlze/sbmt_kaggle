import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hidden Pairs: обучение, оценка и сохранение моделей
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Настройка
    """)
    return


@app.cell
def _():
    DOWNLOAD_DATA = False
    path = None
    if DOWNLOAD_DATA:
        import kagglehub

        # Download latest version
        path = kagglehub.competition_download('week-03-hidden-pairs')
        print("Path to competition files:", path)
    return DOWNLOAD_DATA, path


@app.cell
def _():
    import pandas as pd
    import random
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    from lightgbm import LGBMClassifier, early_stopping
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score
    from pathlib import Path
    import time
    import cleanlab
    import json
    from datetime import datetime

    import joblib
    from importlib.metadata import version

    PACKAGE_VERSIONS = {name: version(name) for name in [
        "numpy", "pandas", "scikit-learn", "lightgbm", "cleanlab", "joblib", "torch",
    ]}
    print(PACKAGE_VERSIONS)
    from dataclasses import replace

    return (
        CalibratedClassifierCV,
        FunctionTransformer,
        GradientBoostingClassifier,
        LGBMClassifier,
        PACKAGE_VERSIONS,
        Path,
        Pipeline,
        RandomForestClassifier,
        RandomizedSearchCV,
        SVC,
        StandardScaler,
        StratifiedKFold,
        cleanlab,
        clone,
        cross_val_score,
        datetime,
        early_stopping,
        joblib,
        json,
        np,
        pd,
        plt,
        random,
        replace,
        roc_auc_score,
        sns,
        time,
        torch,
    )


@app.cell
def _(np, random, torch):
    def set_seed(seed=42):
        """Фиксирует seed для random, NumPy и PyTorch, включая CUDA."""

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.use_deterministic_algorithms(True, warn_only=True)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f'Seed: {seed}')

    return (set_seed,)


@app.cell
def _(Path, set_seed):
    SEED = 42
    set_seed(SEED)

    LOCAL = not Path('/kaggle/input').exists()

    LOG_DIR = Path('./logs') if LOCAL else Path('/kaggle/working/logs')

    OUTPUT_DIR = Path('.') if LOCAL else Path('/kaggle/working')
    RUN_SEARCH = False
    RUN_CLEANLAB = False
    EARLY_STOPPING_ROUNDS = 500  # Длинное плато на validation возможно перед новым ростом.
    return (
        EARLY_STOPPING_ROUNDS,
        LOCAL,
        LOG_DIR,
        OUTPUT_DIR,
        RUN_CLEANLAB,
        RUN_SEARCH,
        SEED,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Данные
    Используются исходные признаки без агрегатов. Состав и порядок признаков при предсказании должны совпадать с обучением.
    """)
    return


@app.cell
def _(DOWNLOAD_DATA, LOCAL, Path, path, pd):
    DATA_DIR = Path(path) if DOWNLOAD_DATA else (
        Path('./data') if LOCAL else Path('/kaggle/input/competitions/week-03-hidden-pairs')
    )

    train_df: pd.DataFrame = pd.read_csv(DATA_DIR / 'train.csv')
    val_df: pd.DataFrame = pd.read_csv(DATA_DIR / 'validation.csv')
    test_df: pd.DataFrame = pd.read_csv(DATA_DIR / 'test.csv')
    sample_submission: pd.DataFrame = pd.read_csv(DATA_DIR / 'sample_submission.csv')

    print("train:", train_df.shape)
    print("val:", val_df.shape)
    print("test:", test_df.shape)
    print("sample submission:", sample_submission.shape)
    return sample_submission, test_df, train_df, val_df


@app.cell
def _(
    test_df: "pd.DataFrame",
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
):
    assert train_df.isna().sum().sum() == 0, "train_df contains NaN values"
    assert val_df.isna().sum().sum() == 0, "val_df contains NaN values"
    assert test_df.isna().sum().sum() == 0, "test_df contains NaN values"

    assert train_df['row_id'].is_unique, "train_df contains duplicate ids"
    assert val_df['row_id'].is_unique, "val_df contains duplicate ids"
    assert test_df['row_id'].is_unique, "test_df contains duplicate ids"
    return


@app.cell
def _(
    np,
    pd,
    test_df: "pd.DataFrame",
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
):
    feature_columns: list[str] = [
        col for col in train_df.columns if col not in ['row_id', 'target']
    ]

    assert feature_columns == [c for c in val_df if c not in ['row_id', 'target']]
    assert feature_columns == [c for c in test_df if c != 'row_id']
    for frame in [train_df, val_df, test_df]:
        assert np.isfinite(frame[feature_columns].to_numpy()).all(), 'Признаки содержат NaN/inf'

    X_train: pd.DataFrame = train_df[feature_columns]
    y_train: pd.Series = train_df['target']

    X_val: pd.DataFrame = val_df[feature_columns]
    y_val: pd.Series = val_df['target']

    X_test: pd.DataFrame = test_df[feature_columns]

    print('# of features:', len(feature_columns))
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_val shape:', X_val.shape)
    print('y_val shape:', y_val.shape)
    print('X_test shape:', X_test.shape)
    return X_test, X_train, X_val, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Engineering
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Поделить фичи пополам, перекинуть между собой, докинуть снизу в датасет
    """)
    return


@app.cell
def _(np, pd):
    def swap_features(df: pd.DataFrame, n_splits=2):
        """Меняет местами части признаков; порядок внутри частей сохраняет"""
        df_idx = np.arange(df.shape[1])
        splits = np.array_split(df_idx, n_splits)[::-1]
        swapped = df.iloc[:, np.concatenate(splits)].copy()
        swapped.columns = df.columns
        return swapped

    def add_swapped_features(df: pd.DataFrame, n_splits=2) -> pd.DataFrame:
        swapped = swap_features(df, n_splits=n_splits)
        return pd.concat([df, swapped], axis=0)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Докинуть еще фичей
    """)
    return


@app.cell
def _(np, pd):
    def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Заменяет признаки каждой строки статистиками, сохраняя индекс.

        row_id и target исключаются. total — сумма значений признаков.
        Пропуски игнорируются; std и var используют ddof=1.
        """
        vals = df.drop(columns=['row_id', 'target'], errors='ignore')
        mean = vals.mean(axis=1)
        total = vals.sum(axis=1, min_count=1)
        return pd.DataFrame({'total': total, 'count_zeros': vals.eq(0).sum(axis=1), 'relative_std': vals.std(axis=1) / mean.replace(0, np.nan), 'max_share': vals.max(axis=1) / total.replace(0, np.nan), 'q90_median_ratio': vals.quantile(0.9, axis=1) / vals.median(axis=1).replace(0, np.nan), 'max': vals.max(axis=1), 'log1p_sum': np.log1p(vals).sum(axis=1, min_count=1)}, index=df.index)

    def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет статистики к исходным признакам, исключая row_id и target."""
        vals = df.drop(columns=['row_id', 'target'], errors='ignore')
        aggregated = aggregate_features(vals)
        return pd.concat([vals, aggregated], axis=1)

    return (add_aggregate_features,)


@app.cell
def _(
    X_test: "pd.DataFrame",
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    add_aggregate_features,
    np,
):
    # only aggregation
    X_train_agg = add_aggregate_features(X_train)
    X_val_agg = add_aggregate_features(X_val)
    X_test_agg = add_aggregate_features(X_test)
    for _features in [X_train_agg, X_val_agg, X_test_agg]:
        assert _features.columns.is_unique, 'Повторяющиеся признаки'
        assert np.isfinite(_features.to_numpy()).all(), 'Агрегации содержат NaN/inf'
    X_train_agg.describe()
    return X_test_agg, X_train_agg, X_val_agg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Сохранение, загрузка и submission
    """)
    return


@app.cell
def _(Path, joblib, json, np, pd):
    def save_model(model, model_path):
        """Сохраняет обученную модель; возвращает путь к файлу."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path, compress=3)
        print(f'Обученная модель сохранена: {model_path}', flush=True)
        return model_path


    def load_model(json_path, model_name):
        """Загружает модель по JSON запуска и точному имени из поля Model."""
        json_path = Path(json_path)
        run_log = json.loads(json_path.read_text(encoding='utf-8'))
        matches = [row for row in run_log['results'] if row['Model'] == model_name]
        if len(matches) != 1:
            available = [row['Model'] for row in run_log['results']]
            raise ValueError(f'Нужна ровно одна модель {model_name!r}. Доступны: {available}')
        model_path = json_path.parent / matches[0]['Model File']
        print(f'Загружаю {model_name}: {model_path}', flush=True)
        return joblib.load(model_path)


    def save_submission(selected_model, X_test, test_ids, sample_submission, output_path):
        """Предсказывает без fit, сохраняет CSV в порядке sample_submission и возвращает его."""
        test_ids = pd.Series(test_ids).reset_index(drop=True)
        if len(X_test) != len(test_ids) or not test_ids.is_unique:
            raise ValueError('Нужен один уникальный row_id на каждую строку X_test')
        if not sample_submission['row_id'].is_unique:
            raise ValueError('В sample_submission есть повторяющиеся row_id')
        if set(test_ids) != set(sample_submission['row_id']):
            raise ValueError('Наборы row_id в test и sample_submission различаются')
        expected = getattr(selected_model, 'feature_names_in_', None)
        if expected is not None and list(X_test.columns) != list(expected):
            raise ValueError('Состав или порядок признаков X_test отличается от обучения')
        # Только предсказания: повторное обучение не запускается.
        print(f'Предсказываю для {len(X_test)} строк...', flush=True)
        test_probs = selected_model.predict_proba(X_test)[:, 1]
        if not np.isfinite(test_probs).all() or not ((test_probs >= 0) & (test_probs <= 1)).all():
            raise ValueError('Модель вернула некорректные вероятности')
        submission = sample_submission[['row_id']].merge(
            pd.DataFrame({'row_id': test_ids, 'target': test_probs}),
            on='row_id', how='left', sort=False, validate='one_to_one',
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        print(f'Сохранено: {output_path.resolve()} | {submission.shape}', flush=True)
        return submission

    return load_model, save_model, save_submission


@app.cell
def _(
    EARLY_STOPPING_ROUNDS,
    PACKAGE_VERSIONS,
    SEED,
    datetime,
    json,
    pd,
    save_model,
):
    def _save_trained_model(model, log_path, model_index, use_cleanlab):
        """Сохраняет обученный классификатор и возвращает путь относительно LOG_DIR.

        После joblib.load() модель готова к predict_proba() без повторного обучения.
        Для CleanLearning сохраняется его финальный обученный классификатор.
        """
        model_dir = log_path.with_suffix('')
        model_dir.mkdir(parents=True, exist_ok=True)
        variant = 'cleanlab' if use_cleanlab else 'base'
        model_path = model_dir / f'{model_index:02d}_{variant}.joblib'
        save_model(model, model_path)
        return str(model_path.relative_to(log_path.parent))

    def _save_pipeline_log(log_path, started_at, metrics_list):
        """Сохраняет время запуска, параметры и метрики уже обученных моделей в JSON."""
        log_data = {
            'run_started_at': started_at.isoformat(),
            'package_versions': PACKAGE_VERSIONS,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
            'updated_at': datetime.now().astimezone().isoformat(),
            'results': json.loads(pd.DataFrame(metrics_list).to_json(orient='records', double_precision=15)),
        }
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding='utf-8')

    def _record_result(result, run_name, cleanlab_enabled, log_path, model_index,
                       val_predictions, histories, metrics_list, extra_metrics=None):
        """Сохраняет модель и добавляет единообразную строку метрик."""
        model_file = _save_trained_model(result['model'], log_path, model_index, cleanlab_enabled)
        val_predictions[run_name] = result['val_probs']
        histories[run_name] = {
            'train_history': result['train_history'], 'val_history': result['val_history'],
            'best_iteration': result['best_iteration'],
        }
        metrics_list.append({
            'Model': run_name, 'Started At': result['started_at'],
            'Params': result['params'], 'Model File': model_file,
            'Cleanlab': cleanlab_enabled,
            'Seed': SEED,
            'Features': result['feature_columns'],
            'Best Iteration': result['best_iteration'],
            **(extra_metrics or {}),
            'Train AUC': result['train_auc'], 'Val AUC': result['val_auc'],
            'Fit Time (s)': round(result['fit_time'], 3),
        })

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Обучение и графики
    """)
    return


@app.cell
def _(
    EARLY_STOPPING_ROUNDS,
    GradientBoostingClassifier,
    LGBMClassifier,
    SEED,
    cleanlab,
    clone,
    datetime,
    early_stopping,
    replace,
    roc_auc_score,
    time,
):
    def _early_stopping_on_val(stopping_rounds):
        """Останавливает LightGBM только по val AUC, сохраняя обе кривые для графика."""
        stop = early_stopping(stopping_rounds, first_metric_only=True, verbose=True)

        def callback(env):
            val_results = [r for r in env.evaluation_result_list if r[0] == 'val']
            if not val_results:
                raise ValueError('Для early stopping нужен eval-набор с именем val')
            stop(replace(env, evaluation_result_list=val_results))

        callback.order = stop.order
        callback.before_iteration = stop.before_iteration
        return callback


    def _extract_auc_history(model, X_train, y_train, X_val, y_val):
        """Возвращает списки train/val ROC AUC по итерациям бустинга.

        Для дерева и случайного леса возвращает два пустых списка.
        """
        if isinstance(model, LGBMClassifier):
            history = model.evals_result_
            return history.get('train', {}).get('auc', []), history['val']['auc']

        if isinstance(model, GradientBoostingClassifier):
            train_scores = [
                roc_auc_score(y_train, probs[:, 1])
                for probs in model.staged_predict_proba(X_train)
            ]
            val_scores = [
                roc_auc_score(y_val, probs[:, 1])
                for probs in model.staged_predict_proba(X_val)
            ]
            return train_scores, val_scores

        return [], []

    def _fit_single_model(model, X_train, y_train, X_val, y_val, use_cleanlab=False):
        """Обучает копию модели и возвращает вероятности, AUC, время fit и кривые.

        Validation используется для оценки и early stopping LightGBM;
        use_cleanlab=True включает поиск ошибок разметки и обучение через CleanLearning.
        Train AUC считается на исходном train. Время fit включает работу CleanLearning,
        но не включает предсказания и вычисление истории AUC.
        """
        started_at = datetime.now().astimezone().isoformat()
        fitted_model = clone(model)
        print('  Обучение...', flush=True)
        start_time = time.perf_counter()

        fit_kwargs = {}
        if isinstance(fitted_model, LGBMClassifier):
            fitted_model.set_params(metric='auc')
            fit_kwargs = {
                # После Cleanlab исходный train уже не является обучающей выборкой.
                # Передаём только val, чтобы исходный train не влиял на early stopping.
                'eval_X': X_val if use_cleanlab else (X_train, X_val),
                'eval_y': y_val if use_cleanlab else (y_train, y_val),
                'eval_names': ['val'] if use_cleanlab else ['train', 'val'],
                'eval_metric': 'auc',
                'callbacks': [
                    _early_stopping_on_val(EARLY_STOPPING_ROUNDS),
                ],
            }

        if use_cleanlab:
            print('  CleanLearning: ищу ошибки разметки на train и обучаю модель...', flush=True)
            clean_model = cleanlab.classification.CleanLearning(
                clf=fitted_model, seed=SEED, verbose=True,
                find_label_issues_kwargs={'n_jobs': 1},
            )
            # Validation передаётся только финальному fit, не внутренним CV-моделям.
            clean_model.fit(X_train, y_train, clf_final_kwargs=fit_kwargs)
            fitted_model = clean_model.clf
            n_issues = int(clean_model.get_label_issues()['is_label_issue'].sum())
            print(f'  CleanLearning: исключено {n_issues} из {len(y_train)} строк.', flush=True)
        else:
            fitted_model.fit(X_train, y_train, **fit_kwargs)

        if isinstance(fitted_model, LGBMClassifier):
            print('Лучшая итерация:', fitted_model.best_iteration_)

        fit_time = time.perf_counter() - start_time
        print(f'  Fit завершён за {fit_time:.1f} с. Считаю предсказания и AUC...', flush=True)
        train_probs = fitted_model.predict_proba(X_train)[:, 1]
        val_probs = fitted_model.predict_proba(X_val)[:, 1]
        train_auc = roc_auc_score(y_train, train_probs)
        val_auc = roc_auc_score(y_val, val_probs)
        print(f'  Train AUC: {train_auc:.5f} | Val AUC: {val_auc:.5f}', flush=True)
        print('  Извлекаю историю AUC по итерациям...', flush=True)
        train_history, val_history = _extract_auc_history(
            fitted_model, X_train, y_train, X_val, y_val,
        )

        print('  Модель готова.', flush=True)
        return {
            'model': fitted_model,
            'started_at': started_at,
            'params': fitted_model.get_params(deep=False),
            'best_iteration': getattr(fitted_model, 'best_iteration_', None),
            'feature_columns': list(X_train.columns),
            'val_probs': val_probs,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'fit_time': fit_time,
            'train_history': train_history,
            'val_history': val_history,
        }

    return


@app.cell
def _(plt, sns):
    def _plot_learning_curves(histories, ncols=2):
        """Рисует train/val ROC AUC по итерациям для моделей с историей обучения."""
        histories = {name: result for name, result in histories.items() if result['val_history']}
        if not histories:
            return

        sns.set_theme(style='darkgrid')
        nrows = (len(histories) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False,
        )
        axes = axes.ravel()

        for ax, (name, result) in zip(axes, histories.items()):
            for key, label in [('train_history', 'Train AUC'), ('val_history', 'Val AUC')]:
                scores = result[key]
                if scores:
                    ax.plot(range(1, len(scores) + 1), scores, label=label)
            if result.get('best_iteration'):
                ax.axvline(result['best_iteration'], color='gray', linestyle='--', label='Best iteration')
            ax.set(title=name, xlabel='Iteration', ylabel='ROC AUC')
            ax.legend()

        # Удаление неиспользуемых ячеек сетки (если количество моделей нечетное)
        for ax in axes[len(histories):]:
            fig.delaxes(ax)

        fig.tight_layout()
        plt.show()

    return


@app.cell
def _(LOG_DIR, datetime, pd):
    def run_model_pipeline(
        models_dict, X_train, y_train, X_val, y_val, plot_curves=True, use_cleanlab=False,
    ):
        """Обучает модели и возвращает вероятности на val и таблицу AUC/времени.

        Строки вероятностей соответствуют X_val; метрики отсортированы по Val AUC.
        use_cleanlab=True запускает каждую модель дважды: без очистки, затем с ней.
        Версия с очисткой получает суффикс " + Cleanlab" в таблицах и графиках.
        Параметры, метрики и время запуска сохраняются в LOG_DIR после каждой модели.
        Обученные модели сохраняются рядом в .joblib; Model File — путь от LOG_DIR.
        """
        started_at = datetime.now().astimezone()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"pipeline_{started_at:%Y-%m-%d_%H-%M-%S_%f}.json"
        print(f'Лог запуска: {log_path.resolve()}', flush=True)
        val_predictions = {}
        metrics_list = []
        histories = {}

        print(f'Пайплайн: {len(models_dict)} моделей | train={X_train.shape}, val={X_val.shape}', flush=True)
        for i, (name, model) in enumerate(models_dict.items(), start=1):
            print(f'\n[{i}/{len(models_dict)}] {name}', flush=True)
            for cleanlab_enabled in ([False, True] if use_cleanlab else [False]):
                run_name = f'{name} + Cleanlab' if cleanlab_enabled else name
                print(f'  Вариант: {run_name}', flush=True)
                result = _fit_single_model(model, X_train, y_train, X_val, y_val, use_cleanlab=cleanlab_enabled)
                _record_result(
                    result, run_name, cleanlab_enabled, log_path, i,
                    val_predictions, histories, metrics_list,
                )
                _save_pipeline_log(log_path, started_at, metrics_list)
                print(f'  Результат сохранён: {log_path.name}', flush=True)

        preds_df = pd.DataFrame(val_predictions, index=X_val.index)
        metrics_df = pd.DataFrame(metrics_list).sort_values('Val AUC', ascending=False).reset_index(drop=True)
        if plot_curves:
            print('\nСтрою графики AUC...', flush=True)
            _plot_learning_curves(histories)
        print('Пайплайн завершён. Предсказания и таблица метрик готовы.', flush=True)
        return preds_df, metrics_df

    return (run_model_pipeline,)


@app.cell
def _(LOG_DIR, RandomizedSearchCV, SEED, clone, datetime, joblib, pd, time):
    def run_model_pipeline_search(
        models_dict, param_grids, X_train, y_train, X_val, y_val,
        n_iter=10, cv=5, scoring='roc_auc', verbose=10, n_jobs=-1, plot_curves=True,
        use_cleanlab=False,
    ):
        """Подбирает параметры на train для моделей, перечисленных в param_grids.

        Параметры подбираются один раз без CleanLearning. При use_cleanlab=True
        лучшая конфигурация обучается без очистки, затем с ней. CV Score/STD в обеих
        строках относятся к одному поиску без очистки.
        Возвращает вероятности и метрики: CV Score/STD относятся к scoring,
        Train/Val AUC — к финальной модели, Fit Time — к её обучению без поиска.
        Параметры, метрики и время запуска сохраняются в LOG_DIR после каждой модели.
        Обученные модели сохраняются рядом в .joblib; Model File — путь от LOG_DIR.
        """
        started_at = datetime.now().astimezone()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"search_{started_at:%Y-%m-%d_%H-%M-%S_%f}.json"
        print(f'Лог запуска: {log_path.resolve()}', flush=True)
        val_predictions = {}
        metrics_list = []
        histories = {}

        print(f'Поиск: {len(param_grids)} моделей | метрика={scoring}', flush=True)
        if use_cleanlab:
            print('CleanLearning включён после поиска. CV Score — без очистки.', flush=True)
        for i, (name, param_grid) in enumerate(param_grids.items(), start=1):
            print(f'\n[{i}/{len(param_grids)}] {name}: подбор параметров...', flush=True)
            if not param_grid:
                raise ValueError(f'Пустая сетка параметров: {name}')
            model = clone(models_dict[name])
            if 'n_jobs' in model.get_params():
                model.set_params(n_jobs=1)
            search = RandomizedSearchCV(
                model, param_distributions=param_grid, n_iter=n_iter,
                cv=cv, scoring=scoring, random_state=SEED,
                verbose=verbose, n_jobs=n_jobs, refit=False, pre_dispatch='n_jobs',
            )
            search_start = time.perf_counter()
            with joblib.parallel_config(backend='loky', inner_max_num_threads=1):
                search.fit(X_train, y_train)
            print(
                f'  Поиск завершён за {time.perf_counter() - search_start:.1f} с. '
                f'Лучший CV {scoring}: {search.best_score_:.5f}',
                flush=True,
            )
            print(f'  Параметры: {search.best_params_}', flush=True)
            print('  Обучаю лучшую конфигурацию на всём train.', flush=True)
            best_model = clone(model).set_params(**search.best_params_)
            for cleanlab_enabled in ([False, True] if use_cleanlab else [False]):
                run_name = f'{name} + Cleanlab' if cleanlab_enabled else name
                print(f'  Вариант: {run_name}', flush=True)
                result = _fit_single_model(best_model, X_train, y_train, X_val, y_val, use_cleanlab=cleanlab_enabled)

                _record_result(
                    result, run_name, cleanlab_enabled, log_path, i,
                    val_predictions, histories, metrics_list,
                    extra_metrics={
                        'Best Params': search.best_params_,
                        'CV Score': search.best_score_,
                        'CV STD': search.cv_results_['std_test_score'][search.best_index_],
                    },
                )
                _save_pipeline_log(log_path, started_at, metrics_list)
                print(f'  Результат сохранён: {log_path.name}', flush=True)

        preds_df = pd.DataFrame(val_predictions, index=X_val.index)
        metrics_df = pd.DataFrame(metrics_list).sort_values('Val AUC', ascending=False).reset_index(drop=True)
        if plot_curves:
            print('\nСтрою графики AUC...', flush=True)
            _plot_learning_curves(histories)
        print('Пайплайн завершён. Предсказания и таблица метрик готовы.', flush=True)
        return preds_df, metrics_df

    return (run_model_pipeline_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Сравнение моделей
    """)
    return


@app.cell
def _():
    lightgbm_params = {
      "subsample_freq": 1,
      "subsample": 0.6,
      "reg_lambda": 0.0,
      "reg_alpha": 0.0,
      "num_leaves": 16,
      "n_estimators": 2000,
      "min_child_samples": 20,
      "max_depth": 6,
      "learning_rate": 0.05,
      "colsample_bytree": 0.5
    }

    gbmt_params = {
      "subsample": 0.6,
      "n_estimators": 2000,
      "min_samples_split": 5,
      "min_samples_leaf": 30,
      "max_features": 0.1,
      "max_depth": 4,
      "learning_rate": 0.1
    }

    random_forest_params = {
      "n_estimators": 2000,
      "min_samples_split": 10,
      "min_samples_leaf": 5,
      "max_features": 0.5,
      "max_depth": 10
    }

    svc_params = {
      "C": 1.0,
      "kernel": "rbf",
      "gamma": "scale",
      "probability": True
    }
    return


@app.cell
def _():
    # models = {
    #     'Decision Tree': DecisionTreeClassifier(
    #         max_depth=5,
    #         min_samples_leaf=20,
    #         random_state=SEED,
    #     ),
    #     'Random Forest': RandomForestClassifier(
    #         **random_forest_params,
    #         random_state=SEED,
    #         n_jobs=-1,
    #     ),
    #     'GBDT': GradientBoostingClassifier(
    #        **gbmt_params,
    #        random_state=SEED,
    #     ),
    #     'LightGBM': LGBMClassifier(
    #         **lightgbm_params,
    #         random_state=SEED,
    #         n_jobs=1,
    #         verbose=-1,
    #     ),
    # }
    return


@app.cell
def _(
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    models,
    run_model_pipeline,
    y_train: "pd.Series",
    y_val: "pd.Series",
):
    preds_df, metrics_df = run_model_pipeline(
        models, X_train, y_train, X_val, y_val,
        use_cleanlab=False,  # True — сравнить обычную модель и CleanLearning
    )
    return (metrics_df,)


@app.cell
def _(metrics_df):
    metrics_df
    return


@app.cell
def _(
    X_train_extended,
    X_val_extended,
    models,
    run_model_pipeline,
    y_train: "pd.Series",
    y_val: "pd.Series",
):
    preds_df_extended, metrics_df_extended = run_model_pipeline(
        models, X_train_extended, y_train, X_val_extended, y_val,
        use_cleanlab=False,  # True — сравнить обычную модель и CleanLearning
    )
    return (metrics_df_extended,)


@app.cell
def _(metrics_df_extended):
    metrics_df_extended
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Пайплайн с
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Cell tags: svc_experiment
    mo.md(r"""
    ## Эксперимент: log1p → StandardScaler → веса агрегатов → RBF SVC

    Обучение и оценка выполняются на исходных строках без перестановки половин.
    SVC — самостоятельная модель, GBDT здесь не участвует.

    После стандартизации умножаем агрегированные признаки на 2. CV ROC-AUC считаем по `decision_function`. Все преобразования обучаются внутри фолдов. Затем калибруем вероятности на train для `predict_proba` и submission; validation в обучении и калибровке не участвует. Сохраняем весь конвейер, поэтому при загрузке передаём признаки с агрегатами **без ручного log1p, масштабирования и умножения на веса**.
    """)
    return


@app.cell
def _(
    FunctionTransformer,
    Pipeline,
    SEED,
    SVC,
    StandardScaler,
    StratifiedKFold,
    X_test_agg,
    X_train_agg,
    X_val_agg,
    np,
):
    from functools import partial

    aggregate_weight = 2.0
    n_aggregate_features = X_train_agg.shape[1] - 512
    if n_aggregate_features < 0:
        raise ValueError('Ожидалось не менее 512 исходных признаков')
    feature_weights = np.ones(X_train_agg.shape[1]) * 50
    feature_weights[512:] = aggregate_weight

    svc_pipeline = Pipeline([
        ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scaler', StandardScaler()),
        ('aggregate_weight', FunctionTransformer(
            partial(np.multiply, feature_weights), feature_names_out='one-to-one',
        )),
        ('svc', SVC(C=10, kernel='rbf', gamma='scale')),
    ])
    svc_pipeline_original = Pipeline([
        ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scaler', StandardScaler()),
        ('svc', SVC(C=10, kernel='rbf', gamma='scale')),
    ])
    svc_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, _features in [('train', X_train_agg), ('val', X_val_agg), ('test', X_test_agg)]:
        if not np.isfinite(_features.to_numpy()).all() or (_features < 0).any().any():
            raise ValueError(f'{name}: для этого эксперимента нужны конечные неотрицательные признаки')
        if list(_features.columns) != list(X_train_agg.columns):
            raise ValueError(f'{name}: состав или порядок признаков отличается от train')

    print(f'SVC: 512 исходных и {n_aggregate_features} агрегированных признаков')
    return svc_cv, svc_pipeline, svc_pipeline_original


@app.cell
def _(
    X_train_agg,
    cross_val_score,
    np,
    svc_cv,
    svc_pipeline,
    y_train: "pd.Series",
):
    svc_cv_scores_agg = cross_val_score(
        svc_pipeline, X_train_agg, y_train,
        cv=svc_cv, scoring='roc_auc', n_jobs=-1,
    )
    print('agg features:', X_train_agg.shape[1], 'size:', X_train_agg.shape[0])
    print('CV AUC по фолдам:', np.round(svc_cv_scores_agg, 5))
    print(f'CV AUC: {svc_cv_scores_agg.mean():.5f} ± {svc_cv_scores_agg.std():.5f}')
    return (svc_cv_scores_agg,)


@app.cell
def _(
    X_train: "pd.DataFrame",
    cross_val_score,
    np,
    svc_cv,
    svc_pipeline_original,
    y_train: "pd.Series",
):
    svc_cv_scores = cross_val_score(
        svc_pipeline_original, X_train, y_train,
        cv=svc_cv, scoring='roc_auc', n_jobs=-1,
    )
    print('original features:', X_train.shape[1])
    print('CV AUC по фолдам:', np.round(svc_cv_scores, 5))
    print(f'CV AUC: {svc_cv_scores.mean():.5f} ± {svc_cv_scores.std():.5f}')
    return


@app.cell
def _(
    CalibratedClassifierCV,
    datetime,
    json,
    pd,
    roc_auc_score,
    save_model,
    time,
):
    def run_svc_calibrated(
        pipeline, X_fit, y_fit, X_train_eval, y_train_eval, X_val, y_val,
        val_ids, cv_scores, calibration_cv, log_dir, seed,
    ):
        """Калибрует SVC на train, оценивает исходные строки и сохраняет запуск.

        Validation не участвует в обучении и калибровке.
        """
        # Калибруется весь Pipeline: scaler обучается заново внутри каждого фолда.
        # ensemble=False: итоговая SVC обучается на всём train.
        svc_model = CalibratedClassifierCV(
            estimator=pipeline, method='sigmoid', cv=calibration_cv,
            ensemble=False,
        )

        svc_started_at = datetime.now().astimezone()
        svc_start = time.perf_counter()
        svc_model.fit(X_fit, y_fit)
        svc_fit_time = time.perf_counter() - svc_start

        svc_train_probs = svc_model.predict_proba(X_train_eval)[:, 1]
        svc_val_probs = svc_model.predict_proba(X_val)[:, 1]
        svc_metrics = pd.DataFrame([{
            'Model': 'log1p + StandardScaler + RBF SVC (weighted aggregate features)',
            'CV AUC (decision_function)': cv_scores.mean(),
            'CV STD': cv_scores.std(),
            'Train AUC': roc_auc_score(y_train_eval, svc_train_probs),
            'Val AUC': roc_auc_score(y_val, svc_val_probs),
            'Fit + Calibration Time (s)': round(svc_fit_time, 3),
        }])

        svc_run_dir = log_dir / f'svc_{svc_started_at:%Y-%m-%d_%H-%M-%S_%f}'
        save_model(svc_model, svc_run_dir / 'model.joblib')
        svc_log = {
            'run_started_at': svc_started_at.isoformat(),
            'feature_columns': list(X_fit.columns),
            'params': pipeline.named_steps['svc'].get_params(),
            'preprocessing': ['log1p', 'StandardScaler', 'aggregate_weight'],
            'calibration': {'method': 'sigmoid', 'ensemble': False, 'folds': calibration_cv.get_n_splits()},
            'seed': seed,
            'cv_scores': cv_scores.tolist(),
            'results': [{
                **json.loads(svc_metrics.to_json(orient='records'))[0],
                'Model File': 'model.joblib',
            }],
        }
        (svc_run_dir / 'results.json').write_text(
            json.dumps(svc_log, ensure_ascii=False, indent=2), encoding='utf-8',
        )
        pd.DataFrame({
            'row_id': val_ids.to_numpy(),
            'target': y_val.to_numpy(),
            'prediction': svc_val_probs,
        }).to_csv(svc_run_dir / 'validation_predictions.csv', index=False)
        print(f'Результаты: {svc_run_dir.resolve()}')
        return svc_model, svc_metrics, svc_run_dir, svc_val_probs

    return (run_svc_calibrated,)


@app.cell
def _(
    LOG_DIR,
    SEED,
    X_train_agg,
    X_val_agg,
    run_svc_calibrated,
    svc_cv,
    svc_cv_scores_agg,
    svc_pipeline,
    val_df: "pd.DataFrame",
    y_train: "pd.Series",
    y_val: "pd.Series",
):
    svc_model, svc_metrics, svc_run_dir, svc_val_probs = run_svc_calibrated(
        svc_pipeline, X_train_agg, y_train, X_train_agg, y_train,
        X_val_agg, y_val, val_df['row_id'], svc_cv_scores_agg,
        svc_cv, LOG_DIR, SEED,
    )
    svc_metrics
    return


@app.cell
def _():
    # Выполни эти строки, когда понадобится файл для Kaggle:
    # submission_svc = save_submission(
    #     svc_model, X_test_agg, test_df['row_id'], sample_submission,
    #     OUTPUT_DIR / 'submission_svc_extended.csv',
    # )

    # После перезапуска ядра можно загрузить весь обученный конвейер:
    # svc_model = load_model(svc_run_dir / 'results.json', svc_log['results'][0]['Model'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Поиск параметров
    Включается флагом `RUN_SEARCH` в настройках.
    CV здесь — дополнительная диагностика. Основное сравнение моделей по условиям соревнования — на supplied validation, отделённом по скрытым сущностям.
    """)
    return


@app.cell
def _(SEED, StratifiedKFold):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
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
    return cv, param_grids


@app.cell
def _(
    RUN_SEARCH,
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    cv,
    display,
    models,
    param_grids,
    run_model_pipeline_search,
    y_train: "pd.Series",
    y_val: "pd.Series",
):
    preds_df_search = metrics_df_search = None
    if RUN_SEARCH:
        preds_df_search, metrics_df_search = run_model_pipeline_search(
            models, param_grids, X_train, y_train, X_val, y_val,
            n_iter=30, cv=cv, scoring='roc_auc', plot_curves=True,
            verbose=10, n_jobs=-1,
        )
        display(metrics_df_search)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Эксперименты с CleanLearning
    Включаются флагом `RUN_CLEANLAB` в настройках.
    """)
    return


@app.cell
def _(
    RUN_CLEANLAB,
    RandomForestClassifier,
    SEED,
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    cleanlab,
    clone,
    roc_auc_score,
    y_train: "pd.Series",
    y_val: "pd.Series",
):
    if RUN_CLEANLAB:
        model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=50, min_samples_leaf=20, random_state=SEED, n_jobs=-1)
        cl = cleanlab.classification.CleanLearning(model, seed=SEED, verbose=True)
        cl.fit(X_train, y_train)
        model = clone(model)
        model.fit(X_train, y_train)

        # issues = cl.get_label_issues()
        # print(issues['is_label_issue'].sum())
        print("Val AUC (cleaned):", roc_auc_score(y_val, cl.predict_proba(X_val)[:, 1]))
        print('Val AUC (model):', roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
        print('model params:', model.get_params())
    return


@app.cell
def _(
    RUN_CLEANLAB,
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    display,
    models,
    run_model_pipeline,
    y_train: "pd.Series",
    y_val: "pd.Series",
):
    # True — сравнить обычную модель и CleanLearning.
    if RUN_CLEANLAB:
        models_clean = {'LightGBM': models['LightGBM']}
        preds_df_clean, metrics_df_clean = run_model_pipeline(
            models_clean, X_train, y_train, X_val, y_val, use_cleanlab=True,
        )
        display(metrics_df_clean)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Предсказания сохранённой модели
    По умолчанию сохраняется submission GBDT из текущего extended-запуска. Для загрузки исторической модели укажи JSON_PATH. В этом режиме достаточно настройки, подготовки данных и функций сохранения/загрузки.
    """)
    return


@app.cell
def _(
    LOG_DIR,
    OUTPUT_DIR,
    X_test_extended,
    display,
    joblib,
    load_model,
    metrics_df_extended,
    sample_submission: "pd.DataFrame",
    save_submission,
    test_df: "pd.DataFrame",
):
    # По умолчанию используем GBDT из текущего запуска на extended-признаках.
    # Для отдельной загрузки старого запуска явно укажи JSON_PATH.
    JSON_PATH = None  # Например: LOG_DIR / 'pipeline_2026-09-19_20-51-16_279271.json'
    MODEL_NAME = 'GBDT'  # Например: 'Random Forest + Cleanlab'
    SUBMISSION_PATH = OUTPUT_DIR / 'submission_GBDT_extended.csv'

    if JSON_PATH is None:
        model_info = metrics_df_extended.loc[metrics_df_extended['Model'] == MODEL_NAME].iloc[0]
        selected_model = joblib.load(LOG_DIR / model_info['Model File'])
    else:
        selected_model = load_model(JSON_PATH, MODEL_NAME)
    submission = save_submission(
        selected_model, X_test_extended, test_df['row_id'], sample_submission, SUBMISSION_PATH,
    )
    display(submission.head())
    return


if __name__ == "__main__":
    app.run()
