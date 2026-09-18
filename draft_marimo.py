import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score
    from pathlib import Path
    from utils import set_seed
    set_seed()
    return DecisionTreeClassifier, Path, pd, roc_auc_score, train_test_split


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
    DecisionTreeClassifier,
    X_test: "pd.DataFrame",
    X_train: "pd.DataFrame",
    pd,
    roc_auc_score,
    train_test_split,
):
    def adv_val(df1, df2, model):
        X_train['is_test'] = 0
        X_test['is_test'] = 1

        adv_train_test = pd.concat([X_train, X_test])
        adv_X = adv_train_test.drop(columns="is_test").to_numpy()
        adv_y = adv_train_test["is_test"].to_numpy()

        adv_X_tr, adv_X_val, adv_y_tr, adv_y_val = train_test_split(adv_X, adv_y, test_size=0.2, stratify=adv_y)
        model.fit(adv_X_tr, adv_y_tr)

        pred =  model.predict_proba(adv_X_val)[:,1]
        score = roc_auc_score(adv_y_val, pred)
        return score

    model = DecisionTreeClassifier(max_depth=3, min_samples_split=20)
    return adv_val, model


@app.cell
def _(
    X_test: "pd.DataFrame",
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    adv_val,
    model,
):
    print(f'train vs val', adv_val(X_train, X_val, model))
    print(f'train vs test', adv_val(X_train, X_test, model))
    print(f'test vs val', adv_val(X_test, X_val, model))
    return


if __name__ == "__main__":
    app.run()
