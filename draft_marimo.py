import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from pathlib import Path
    from utils import set_seed
    from tqdm.auto import tqdm

    SEED = 42
    set_seed(SEED)
    return (
        Path,
        RandomForestClassifier,
        SEED,
        StratifiedKFold,
        clone,
        np,
        pd,
        roc_auc_score,
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
    return X_test, X_train, X_val


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
