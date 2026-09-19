import marimo

__generated_with = "0.24.0"
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

    SEED = 42
    set_seed(SEED)
    return (
        DecisionTreeClassifier,
        GradientBoostingClassifier,
        Path,
        RandomForestClassifier,
        RandomizedSearchCV,
        SEED,
        StratifiedKFold,
        clone,
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
def _(
    pd,
    test_df: "pd.DataFrame",
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
):
    feature_columns: list[str] = [
        col for col in train_df.columns if col not in ['row_id', 'target']
    ]

    X_train: pd.DataFrame = train_df[feature_columns]
    y_train: pd.Series = train_df['target']

    X_val: pd.DataFrame = val_df[feature_columns]
    y_val: pd.Series = val_df['target']

    X_test: pd.DataFrame = test_df[feature_columns]
    return X_test, X_train, X_val, y_train, y_val


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
    return adv_val, model


@app.cell
def _(
    X_test: "pd.DataFrame",
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    adv_val,
    model,
):
    train_vs_val: tuple = adv_val(X_train, X_val, model)
    train_vs_test: tuple = adv_val(X_train, X_test, model)
    test_vs_val: tuple = adv_val(X_test, X_val, model)

    print(f'train vs val: {train_vs_val[0]} ± {train_vs_val[1]}')
    print(f'train vs test: {train_vs_test[0]} ± {train_vs_test[1]}')
    print(f'test vs val: {test_vs_val[0]} ± {test_vs_val[1]}')
    return test_vs_val, train_vs_test, train_vs_val


@app.cell
def _(test_vs_val: tuple, train_vs_test: tuple, train_vs_val: tuple):
    print(train_vs_val[2])
    print(train_vs_test[2])
    print(test_vs_val[2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Сравнение моделей
    """)
    return


@app.cell
def _(RandomizedSearchCV, SEED, clone, np, pd, plt, roc_auc_score, sns, time):
    def _extract_auc_history(model, X_train, y_train, X_val, y_val):
        """
        Извлекает историю ROC AUC по итерациям 
        """
        train_scores, val_scores = [], []

        def _find_auc_in_dict(d):
            for metric_name, values in d.items():
                if 'auc' in metric_name.lower():
                    return values
            return []

        # CatBoost
        if hasattr(model, 'get_evals_result'):
            evals = model.get_evals_result()
            for k, v in evals.items():
                auc_vals = _find_auc_in_dict(v)
                if auc_vals:
                    if any(term in k.lower() for term in ['learn', 'train']):
                        train_scores = auc_vals
                    else:
                        val_scores = auc_vals

        # LightGBM / XGBoost
        elif hasattr(model, 'evals_result_'):
            evals = model.evals_result_
            for k, v in evals.items():
                auc_vals = _find_auc_in_dict(v)
                if auc_vals:
                    if 'train' in k.lower() or k == 'valid_0':
                        if not train_scores:
                            train_scores = auc_vals
                        else:
                            val_scores = auc_vals
                    else:
                        val_scores = auc_vals

        # Sklearn GradientBoostingClassifier
        elif hasattr(model, 'staged_predict_proba'):
            for tr_pred, va_pred in zip(model.staged_predict_proba(X_train), model.staged_predict_proba(X_val)):
                train_scores.append(roc_auc_score(y_train, tr_pred[:, 1]))
                val_scores.append(roc_auc_score(y_val, va_pred[:, 1]))

        return train_scores, val_scores


    def _fit_single_model(model, X_train, y_train, X_val, y_val):
        fitted_model = clone(model)
        model_type = type(fitted_model).__name__.lower()

        fit_kwargs = {}
        if any(m in model_type for m in ['catboost', 'lgbm', 'lightgbm', 'xgb', 'xgboost']):
            fit_kwargs['eval_set'] = [(X_train, y_train), (X_val, y_val)]
            if 'catboost' in model_type:
                fit_kwargs['verbose'] = False

        start_time = time.time()
        try:
            fitted_model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            fitted_model.fit(X_train, y_train)
        fit_time = time.time() - start_time

        train_probs = fitted_model.predict_proba(X_train)[:, 1]
        val_probs = fitted_model.predict_proba(X_val)[:, 1]

        train_auc = roc_auc_score(y_train, train_probs)
        val_auc = roc_auc_score(y_val, val_probs)

        tr_hist, va_hist = _extract_auc_history(fitted_model, X_train, y_train, X_val, y_val)

        return {
            'val_probs': val_probs,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'fit_time': fit_time,
            'train_history': tr_hist,
            'val_history': va_hist
        }


    def _plot_learning_curves(histories, ncols=2):
        """
        Отрисовывает динамику ROC AUC в сетке по ncols графиков в ряду (по умолчанию 2).
        """
        iterative_models = {k: v for k, v in histories.items() if len(v['val_history']) > 0}
        if not iterative_models:
            return

        sns.set_theme(style="darkgrid")
        num_models = len(iterative_models)
        nrows = int(np.ceil(num_models / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False)

        for idx, (name, hist) in enumerate(iterative_models.items()):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            iterations = np.arange(1, len(hist['val_history']) + 1)
    
            if len(hist['train_history']) > 0:
                ax.plot(iterations, hist['train_history'], label='Train AUC', color='#5E93CF', linewidth=2)
            ax.plot(iterations, hist['val_history'], label='Val AUC', color='#F80012', linewidth=2)

            ax.set_title(name, fontsize=12, fontweight='bold', loc='left')
            ax.set_xlabel('Iteration / Epoch', fontweight='bold')
            ax.set_ylabel('ROC AUC', fontweight='bold')
            ax.legend(frameon=True)

        # Удаление неиспользуемых ячеек сетки (если количество моделей нечетное)
        for idx in range(num_models, nrows * ncols):
            row, col = divmod(idx, ncols)
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.show()


    def run_model_pipeline(models_dict, X_train, y_train, X_val, y_val, plot_curves=True):
        val_predictions = {}
        metrics_list = []
        histories = {}

        for name, model in models_dict.items():
            res = _fit_single_model(model, X_train, y_train, X_val, y_val)

            val_predictions[name] = res['val_probs']
            metrics_list.append({
                'Model': name,
                'Train AUC': res['train_auc'],
                'Val AUC': res['val_auc'],
                'Fit Time (s)': round(res['fit_time'], 3)
            })
            histories[name] = {
                'train_history': res['train_history'],
                'val_history': res['val_history']
            }

        preds_df = pd.DataFrame(val_predictions, index=X_val.index if hasattr(X_val, 'index') else None)
        metrics_df = pd.DataFrame(metrics_list).sort_values(by='Val AUC', ascending=False).reset_index(drop=True)

        if plot_curves:
            _plot_learning_curves(histories, ncols=2)

        return preds_df, metrics_df

    def run_model_pipeline_search(models_dict, param_grids, X_train, y_train, X_val, y_val, n_iter=10, cv=5, scoring='roc_auc', plot_curves=True):
        val_predictions = {}
        metrics_list = []
        histories = {}

        for name, model in models_dict.items():
            param_grid = param_grids.get(name, {})
            search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, cv=cv, scoring=scoring, random_state=SEED)
            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            res = _fit_single_model(best_model, X_train, y_train, X_val, y_val)

            val_predictions[name] = res['val_probs']
            metrics_list.append({
                'Model': name,
                'Best Params': search.best_params_,
                'Train AUC': res['train_auc'],
                'Val AUC': res['val_auc'],
                'Fit Time (s)': round(res['fit_time'], 3)
            })
            histories[name] = {
                'train_history': res['train_history'],
                'val_history': res['val_history']
            }

        preds_df = pd.DataFrame(val_predictions, index=X_val.index if hasattr(X_val, 'index') else None)
        metrics_df = pd.DataFrame(metrics_list).sort_values(by='Val AUC', ascending=False).reset_index(drop=True)

        if plot_curves:
            _plot_learning_curves(histories, ncols=2)

        return preds_df, metrics_df

    return run_model_pipeline, run_model_pipeline_search


@app.cell
def _(
    DecisionTreeClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    SEED,
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    run_model_pipeline,
    y_train: "pd.Series",
    y_val: "pd.Series",
):
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED),
        'GBDT 3': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=SEED),
        'GBDT 5': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=SEED),
        # 'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, metric='auc', random_state=SEED, verbose=-1),
        # 'CatBoost': CatBoostClassifier(iterations=100, learning_rate=0.05, depth=4, eval_metric='AUC', random_seed=SEED)
    }

    # Запуск пайплайна
    preds_df, metrics_df = run_model_pipeline(models, X_train, y_train, X_val, y_val)

    # Таблица с результатами
    metrics_df
    return models, preds_df


@app.cell
def _(preds_df):
    preds_df
    return


@app.cell
def _(
    SEED,
    StratifiedKFold,
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    models,
    run_model_pipeline_search,
    y_train: "pd.Series",
    y_val: "pd.Series",
):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    param_grids = {
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_leaf': [1, 2, 5, 10, 20],
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.5],
        },
        'GBDT': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'max_depth': [2, 3, 4, 5],
            'min_samples_leaf': [1, 5, 10, 20],
        },
    }
    preds_df_search, metrics_df_search = run_model_pipeline_search(models, param_grids, X_train, y_train, X_val, y_val, n_iter=10, cv=cv, scoring='roc_auc', plot_curves=True)
    return (metrics_df_search,)


@app.cell
def _(metrics_df_search):
    metrics_df_search
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
