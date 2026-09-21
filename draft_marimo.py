import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from pathlib import Path
    from utils import set_seed
    from tqdm.auto import tqdm
    import time
    import cleanlab
    import joblib
    import json
    from datetime import datetime
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    LOG_DIR = Path('logs')
    SEED = 42
    set_seed(SEED)
    return (
        CatBoostClassifier,
        GradientBoostingClassifier,
        LGBMClassifier,
        LOG_DIR,
        Path,
        RandomForestClassifier,
        RandomizedSearchCV,
        SEED,
        StratifiedKFold,
        cleanlab,
        clone,
        datetime,
        joblib,
        json,
        np,
        pd,
        plt,
        roc_auc_score,
        sns,
        time,
        tqdm,
    )


@app.cell
def _(Path, pd):
    DATA_DIR = Path('data')

    train_df: pd.DataFrame = pd.read_csv(DATA_DIR / 'train.csv')
    val_df: pd.DataFrame = pd.read_csv(DATA_DIR / 'validation.csv')
    test_df: pd.DataFrame = pd.read_csv(DATA_DIR / 'test.csv')
    sample_submission: pd.DataFrame = pd.read_csv(DATA_DIR / 'sample_submission.csv')

    print("train:", train_df.shape)
    print("val:", val_df.shape)
    print("test:", test_df.shape)
    print("sample submission:", sample_submission.shape)
    return test_df, train_df, val_df


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
def _(pd):
    def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Заменяет признаки каждой строки статистиками, сохраняя индекс.

        row_id и target исключаются. total — сумма значений признаков.
        Пропуски игнорируются; std и var используют ddof=1.
        """
        vals = df.drop(columns=['row_id', 'target'], errors='ignore')
        q1 = vals.quantile(0.25, axis=1)
        q2 = vals.median(axis=1)
        q3 = vals.quantile(0.75, axis=1)

        return pd.DataFrame({
            'mean': vals.mean(axis=1),
            'median': q2,
            'std': vals.std(axis=1),
            'var': vals.var(axis=1),
            'mad': vals.sub(q2, axis=0).abs().median(axis=1),
            'min': vals.min(axis=1),
            '25% (Q1)': q1,
            '50% (Q2)': q2,
            '75% (Q3)': q3,
            'max': vals.max(axis=1),
            'IQR': q3 - q1,
            'total': vals.sum(axis=1, min_count=1),
        }, index=df.index)

    return (aggregate_features,)


@app.cell
def _(aggregate_features, pd):
    def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет статистики к исходным признакам, исключая row_id и target."""
        vals = df.drop(columns=['row_id', 'target'], errors='ignore')
        aggregated = aggregate_features(vals)
        return pd.concat([vals, aggregated], axis=1)

    return (add_aggregate_features,)


@app.cell
def _(
    add_aggregate_features,
    aggregate_features,
    test_df: "pd.DataFrame",
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
):
    X_train_agg = aggregate_features(train_df)
    X_val_agg = aggregate_features(val_df)
    X_test_agg = aggregate_features(test_df)

    X_train_extended = add_aggregate_features(train_df)
    X_val_extended = add_aggregate_features(val_df)
    X_test_extended = add_aggregate_features(test_df)
    return


@app.cell
def _(
    test_df: "pd.DataFrame",
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
):
    def add_total(df):
        vals = df.drop(columns=['row_id', 'target', 'total'], errors='ignore')
        return df.assign(total=vals.sum(axis=1))

    train_with_total = add_total(train_df)
    val_with_total = add_total(val_df)
    test_with_total = add_total(test_df)
    return test_with_total, train_with_total, val_with_total


@app.cell
def _(train_with_total):
    train_with_total['total']
    return


@app.cell
def _(pd, test_with_total, train_with_total, val_with_total):
    feature_columns: list[str] = [
        col for col in train_with_total.columns if col not in ['row_id', 'target']
    ]

    X_train: pd.DataFrame = train_with_total[feature_columns]
    y_train: pd.Series = train_with_total['target']

    X_val: pd.DataFrame = val_with_total[feature_columns]
    y_val: pd.Series = val_with_total['target']

    X_test: pd.DataFrame = test_with_total[feature_columns]
    return X_train, X_val, y_train, y_val


@app.cell
def _(train_with_total):
    train_with_total['total'].describe()
    return


@app.cell
def _(sns, train_with_total):
    sns.histplot(data=train_with_total, x='total', hue='target', bins=50, kde=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## adversarial validation
    проверка насколько отличается test от train и остальные
    """)
    return


@app.cell
def _(
    RandomForestClassifier,
    SEED,
    StratifiedKFold,
    clone,
    np,
    pd,
    roc_auc_score,
    tqdm,
):
    def adv_val(df, test_df, model):
        df1: pd.DataFrame = df.copy()
        df2: pd.DataFrame = test_df.copy()

        df1['is_test'] = 0
        df2['is_test'] = 1

        adv_dataset: pd.DataFrame = pd.concat([df1, df2], ignore_index=True)
        adv_X: pd.DataFrame = adv_dataset.drop(columns='is_test')
        adv_y: pd.Series = adv_dataset['is_test']

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores: list = []
        feature_importances: np.ndarray = np.zeros(adv_X.shape[1])

        for train_idx, val_idx in tqdm(cv.split(adv_X, adv_y), total=5, leave=False):
            fold_X_train, fold_y_train = adv_X.iloc[train_idx], adv_y[train_idx]
            fold_X_val, fold_y_val = adv_X.iloc[val_idx], adv_y[val_idx]

            fold_model = clone(model)
            fold_model.fit(fold_X_train, fold_y_train)

            pred: np.ndarray = fold_model.predict_proba(fold_X_val)[:, 1]

            score: float = roc_auc_score(fold_y_val, pred)
            scores.append(score)

            if hasattr(fold_model, 'feature_importances_'):
                feature_importances += fold_model.feature_importances_ / cv.n_splits

        fi_series: pd.Series = (
            pd.Series(
                feature_importances, index=adv_X.columns
            )
            .sort_values(ascending=False)
        )

        return np.mean(scores), np.std(scores), scores, fi_series

    model = RandomForestClassifier(
        n_estimators=700, criterion='gini',
        max_depth=7, min_samples_leaf=10,
        random_state=SEED
    )
    return


@app.cell
def _():
    # train_vs_val: tuple = adv_val(X_train, X_val, model)
    # train_vs_test: tuple = adv_val(X_train, X_test, model)
    # test_vs_val: tuple = adv_val(X_test, X_val, model)

    # print(f'train vs val: {train_vs_val[0]} ± {train_vs_val[1]}')
    # print(f'train vs test: {train_vs_test[0]} ± {train_vs_test[1]}')
    # print(f'test vs val: {test_vs_val[0]} ± {test_vs_val[1]}')
    return


@app.cell
def _():
    # print(train_vs_val[2])
    # print(train_vs_test[2])
    # print(test_vs_val[2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Сравнение моделей
    """)
    return


@app.cell
def _(
    CatBoostClassifier,
    GradientBoostingClassifier,
    LGBMClassifier,
    LOG_DIR,
    RandomizedSearchCV,
    SEED,
    cleanlab,
    clone,
    datetime,
    joblib,
    json,
    pd,
    plt,
    roc_auc_score,
    sns,
    time,
):
    def _extract_auc_history(model, X_train, y_train, X_val, y_val):
        """Возвращает списки train/val ROC AUC по итерациям бустинга.

        Для дерева и случайного леса возвращает два пустых списка.
        """
        if isinstance(model, CatBoostClassifier):
            history = model.get_evals_result()
            # Два eval_set при обучении: сначала train, затем validation.
            return history['validation_0']['AUC'], history['validation_1']['AUC']

        if isinstance(model, LGBMClassifier):
            history = model.evals_result_
            return history['train']['auc'], history['val']['auc']

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

        Validation используется для оценки; CatBoost сохраняет все итерации.
        use_cleanlab=True включает поиск ошибок разметки и обучение через CleanLearning.
        Train AUC считается на исходном train. Время fit включает работу CleanLearning,
        но не включает предсказания и вычисление истории AUC.
        """
        started_at = datetime.now().astimezone().isoformat()
        fitted_model = clone(model)
        print('  Обучение...', flush=True)
        start_time = time.perf_counter()

        fit_kwargs = {}
        if isinstance(fitted_model, CatBoostClassifier):
            fitted_model.set_params(eval_metric='AUC:hints=skip_train~false', verbose=False)
            fit_kwargs = {
                'eval_set': [(X_train, y_train), (X_val, y_val)],
                'use_best_model': False,
                'verbose': False,
            }
        elif isinstance(fitted_model, LGBMClassifier):
            fit_kwargs = {
                'eval_set': [(X_train, y_train), (X_val, y_val)],
                'eval_names': ['train', 'val'],
                'eval_metric': 'auc',
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
            'val_probs': val_probs,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'fit_time': fit_time,
            'train_history': train_history,
            'val_history': val_history,
        }


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
                ax.plot(range(1, len(scores) + 1), scores, label=label)
            ax.set(title=name, xlabel='Iteration', ylabel='ROC AUC')
            ax.legend()

        # Удаление неиспользуемых ячеек сетки (если количество моделей нечетное)
        for ax in axes[len(histories):]:
            fig.delaxes(ax)

        fig.tight_layout()
        plt.show()


    def _save_trained_model(model, log_path, model_index, use_cleanlab):
        """Сохраняет обученный классификатор и возвращает путь относительно LOG_DIR.

        После joblib.load() модель готова к predict_proba() без повторного обучения.
        Для CleanLearning сохраняется его финальный обученный классификатор.
        """
        model_dir = log_path.with_suffix('')
        model_dir.mkdir(parents=True, exist_ok=True)
        variant = 'cleanlab' if use_cleanlab else 'base'
        model_path = model_dir / f'{model_index:02d}_{variant}.joblib'
        joblib.dump(model, model_path, compress=3)
        print(f'  Обученная модель сохранена: {model_path}', flush=True)
        return str(model_path.relative_to(log_path.parent))


    def _save_pipeline_log(log_path, started_at, metrics_list):
        """Сохраняет время запуска, параметры и метрики уже обученных моделей в JSON."""
        log_data = {
            'run_started_at': started_at.isoformat(),
            'updated_at': datetime.now().astimezone().isoformat(),
            'results': json.loads(pd.DataFrame(metrics_list).to_json(orient='records', double_precision=15)),
        }
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding='utf-8')


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
                model_file = _save_trained_model(
                    result.pop('model'), log_path, i, cleanlab_enabled,
                )
                val_predictions[run_name] = result['val_probs']
                histories[run_name] = result
                metrics_list.append({
                    'Model': run_name,
                    'Started At': result['started_at'],
                    'Params': result['params'],
                    'Model File': model_file,
                    'Cleanlab': cleanlab_enabled,
                    'Train AUC': result['train_auc'],
                    'Val AUC': result['val_auc'],
                    'Fit Time (s)': round(result['fit_time'], 3),
                })
                _save_pipeline_log(log_path, started_at, metrics_list)
                print(f'  Результат сохранён: {log_path.name}', flush=True)

        preds_df = pd.DataFrame(val_predictions, index=X_val.index)
        metrics_df = pd.DataFrame(metrics_list).sort_values('Val AUC', ascending=False).reset_index(drop=True)
        if plot_curves:
            print('\nСтрою графики AUC...', flush=True)
            _plot_learning_curves(histories)
        print('Пайплайн завершён. Предсказания и таблица метрик готовы.', flush=True)
        return preds_df, metrics_df


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
            model = models_dict[name]
            # LightGBM: параллелим фолды, ограничивая OpenMP внутри процессов.
            search_n_jobs = n_jobs
            if isinstance(model, LGBMClassifier):
                search_n_jobs = min(4, joblib.effective_n_jobs(n_jobs))
                print(f'  LightGBM: {search_n_jobs} CV-процесса, по одному потоку.', flush=True)
            search = RandomizedSearchCV(
                model, param_distributions=param_grid, n_iter=n_iter,
                cv=cv, scoring=scoring, random_state=SEED,
                verbose=verbose, n_jobs=search_n_jobs, refit=False,
            )
            search_start = time.perf_counter()
            if isinstance(model, LGBMClassifier):
                with joblib.parallel_config(backend='loky', inner_max_num_threads=1):
                    search.fit(X_train, y_train)
            else:
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

                model_file = _save_trained_model(
                    result.pop('model'), log_path, i, cleanlab_enabled,
                )
                val_predictions[run_name] = result['val_probs']
                histories[run_name] = result
                metrics_list.append({
                    'Model': run_name,
                    'Started At': result['started_at'],
                    'Params': result['params'],
                    'Model File': model_file,
                    'Cleanlab': cleanlab_enabled,
                    'Best Params': search.best_params_,
                    'CV Score': search.best_score_,
                    'CV STD': search.cv_results_['std_test_score'][search.best_index_],
                    'Train AUC': result['train_auc'],
                    'Val AUC': result['val_auc'],
                    'Fit Time (s)': round(result['fit_time'], 3),
                })
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


@app.cell
def _(
    GradientBoostingClassifier,
    LGBMClassifier,
    RandomForestClassifier,
    SEED,
):
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
    return (models,)


@app.cell
def _():
    # preds_df, metrics_df = run_model_pipeline(
    #     models, X_train, y_train, X_val, y_val, use_cleanlab=False,
    # )
    # metrics_df
    return


@app.cell
def _():
    # preds_df
    return


@app.cell
def _(SEED, StratifiedKFold):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    param_grids = {
        'Random Forest': {
            'n_estimators': [300, 500, 800, 1000, 1200, 1500],
            'max_depth': [5, 10, 15],
            'min_samples_split': [5, 10, 20, 50],
            'min_samples_leaf': [5, 10, 20],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
        },

        'GBDT': {
            'n_estimators': [300, 500, 800, 1000, 1200, 1500],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [1, 2, 3],
            'min_samples_split': [5, 10, 20, 50],
            'min_samples_leaf': [5, 10, 20],
            'max_features': ['sqrt', 'log2', 0.5],
            'subsample': [0.6, 0.8, 1.0],
        },
        'LightGBM': {
            'n_estimators': [300, 400, 500, 800, 1000, 1200, 1500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'max_depth': [1, 2, 3, 5],
            'min_child_samples': [10, 20, 50, 100],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'reg_alpha': [0.0, 0.01, 0.1, 1.0],
            'reg_lambda': [0.0, 0.01, 0.1, 1.0],
        },
    }
    return cv, param_grids


@app.cell
def _(
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    cv,
    models,
    param_grids,
    run_model_pipeline_search,
    y_train: "pd.Series",
    y_val: "pd.Series",
):
    preds_df_search, metrics_df_search = run_model_pipeline_search(
        models, param_grids, X_train, y_train, X_val, y_val,
        n_iter=100, cv=cv, scoring='roc_auc', plot_curves=True,
        verbose=10, n_jobs=-1, use_cleanlab=False,
    )
    return (metrics_df_search,)


@app.cell
def _(metrics_df_search):
    metrics_df_search
    return


@app.cell
def _(Path, json, pd):

    run_dir = Path("logs/script_search_2026-09-20_01-25-43_392077")
    results = pd.DataFrame(
        json.loads((run_dir / "results.json").read_text())["results"]
    ).sort_values("CV Score", ascending=False)

    results[["Model", "CV Score", "Val AUC",  "Best Params"]]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
