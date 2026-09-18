import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Спец. настройки для графиков
    sns.set_theme(
        style='darkgrid',  # приятная фоновая сетка
        palette='bright'  # очень яркая палитра цветов
    )
    return Path, np, pd, plt, sns


@app.cell
def _(Path, pd):
    DATA_DIR = Path('data')

    train_df: pd.DataFrame = pd.read_csv(DATA_DIR / 'train.csv')
    val_df: pd.DataFrame = pd.read_csv(DATA_DIR / 'validation.csv')
    test_df: pd.DataFrame = pd.read_csv(DATA_DIR / 'test.csv')
    sample_submission: pd.DataFrame = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    return test_df, train_df, val_df


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
    return X_train, y_train


@app.cell
def _(plt, train_df: "pd.DataFrame", val_df: "pd.DataFrame"):
    target_distribution = (
        train_df['target']
        .map({0: 'Different', 1: 'Same'})
        .value_counts()
    )
    val_distribution = (
        val_df['target']
        .map({0: 'Different', 1: 'Same'})
        .value_counts()
    )

    fig_1, ax_1 = plt.subplots(1, 2,figsize=(10, 3))
    ax_1[0].bar(target_distribution.index, target_distribution.values)
    ax_1[0].set_title('Distribution of target classes in the training set')
    ax_1[1].bar(val_distribution.index, val_distribution.values)
    ax_1[1].set_title('Distribution of target classes in the validation set')
    return


@app.cell
def _(X_train: "pd.DataFrame", np, pd, plt):
    feature_ranges: pd.Series = X_train.max() - X_train.min()

    fig_2, ax_2 = plt.subplots(figsize=(14, 4))

    ax_2.plot(feature_ranges.values, color='#1f77b4', linewidth=1)
    ax_2.fill_between(np.arange(len(feature_ranges)), feature_ranges.values, color='#1f77b4', alpha=0.2)

    ax_2.set_title('Feature ranges in the training set')
    ax_2.set_ylabel('Range')
    ax_2.set_xlabel('Feature Index (0 to 511)')
    ax_2.set_xlim(0, len(feature_ranges))
    ax_2.grid(True, linestyle='--', alpha=0.5)

    # Раскомментируй, если разброс значений в разных масштабах (10^1, 10^5):
    # ax_2.set_yscale('log')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_train: "pd.DataFrame", np):
    corr_matrix = X_train.corr(method='spearman')

    upper_tri = (
        corr_matrix
        .where(
            np.triu(
                    np.ones(corr_matrix.shape), k=1
                )
                .astype(bool)
        )
    )

    corr_pairs = (
        upper_tri.stack()
        .reset_index()
        .rename(
            columns={
                'level_0': 'Feature_1', 
                'level_1': 'Feature_2', 
                0: 'Correlation'
            }
        )
    )

    threshold = 0.5
    high_corr = (
        corr_pairs[
            corr_pairs['Correlation'].abs() >= threshold
        ]
        .sort_values(
            by='Correlation', key=abs, ascending=False
        )
    )

    print(f"Всего пар с корреляцией Спирмана >= {threshold}: {len(high_corr)}")
    high_corr.head(10)
    return


@app.cell
def _(X_train: "pd.DataFrame", y_train: "pd.Series"):
    # Считаем корреляцию всех столбцов X_train с y_train
    target_corr = X_train.corrwith(y_train, method='spearman')

    target_corr_df = (
        target_corr
        .reset_index()
        .rename(
            columns={
                'index': 'Feature', 
                0: 'Target_Correlation'
            }
        )
    )

    threshold_1 = 0.3
    high_target_corr = (
        target_corr_df[
            target_corr_df['Target_Correlation'].abs() >= threshold_1
        ]
        .sort_values(
            by='Target_Correlation', key=abs, ascending=False
        )
    )

    print(f"Всего признаков с корреляцией Спирмана к таргету >= {threshold_1}: {len(high_target_corr)}")
    high_target_corr.head(15)
    return


@app.cell
def _(np, pd):
    def describe_feats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Функция для описания признаков в датафрейме.
    
        Параметры:
        df (pd.DataFrame): Входной датафрейм.
    
        Возвращает:
        pd.DataFrame: Датафрейм с описанием признаков.
        """
        stats = []
        for col in df.columns:
            vals = df[col]
            is_num = pd.api.types.is_numeric_dtype(vals)

            # Мода
            mode = vals.mode(dropna=True)
            top_val = mode.iloc[0] if not mode.empty else np.nan
            freq_top = (vals == top_val).sum() if not pd.isna(top_val) else np.nan

            stat = {
                # 'dtype': str(vals.dtype),
                # 'count': vals.count(),
                # 'num_miss': vals.isna().sum(),
                'num_unique': vals.nunique(dropna=True),
                # 'top_val': top_val,
                # 'freq_top': freq_top,
            }

            if is_num:
                mean = vals.mean()
                q1 = vals.quantile(0.25)
                q2 = vals.median()
                q3 = vals.quantile(0.75)

                # Выбросы по правилу Тьюки
                lower_bound = q1 - 1.5 * (q3 - q1)
                upper_bound = q3 + 1.5 * (q3 - q1)
                outliers_mask = (vals < lower_bound) | (vals > upper_bound)
                outliers_count = outliers_mask.sum()
                # Считаем процент от валидных (не пропущенных) значений
                outliers_pct = (
                    (outliers_count / vals.count()) * 100 if vals.count() > 0 else 0.0
                )

                stat.update({
                    'mean': mean,
                    'median': q2,
                    'std': vals.std(),
                    'var': vals.var(),
                    'mad': (vals - q2).abs().median(),
                    'min': vals.min(),
                    '25% (Q1)': q1,
                    '50% (Q2)': q2,
                    '75% (Q3)': q3,
                    'max': vals.max(),
                    'IQR': q3 - q1,
                    "tukey_outliers_cnt": outliers_count,
                    "tukey_outliers_pct (%)": round(outliers_pct, 2),
                    'skew': vals.skew(),  # асимметрия
                    'kurtosis': vals.kurtosis(),  # эксцесс
                })
            else:
                for key in ['mean', 'median', 'std',
                            'var', 'mad', 'min', 
                            '25% (Q1)', '50% (Q2)', '75% (Q3)', 
                            'max', 'IQR', 'tukey_outliers_cnt', 'tukey_outliers_pct (%)', 'skew', 'kurtosis']:
                    stat[key] = np.nan
            stats.append(stat)

        return pd.DataFrame(stats, index=df.columns)

    return (describe_feats,)


@app.cell
def _(X_train: "pd.DataFrame", describe_feats):
    describe_feats(X_train)
    return


@app.cell
def _(X_train: "pd.DataFrame", pd, plt, sns, y_train: "pd.Series"):
    import base64
    from io import BytesIO

    def generate_sparkline(data, y=None, plot_type='dist'):
        """Генерирует маленькую картинку графика в формате base64."""
        color_dict = {0: '#5E93CF', 1: '#F80012'}

        fig, ax = plt.subplots(figsize=(2, 0.6))

        if plot_type == 'dist':
            # Распределение значения фичи
            sns.kdeplot(data, ax=ax, fill=True, color='#5E93CF', lw=1)
        
        elif plot_type == 'target' and y is not None:
            # Зависимость от таргета
            sns.boxplot(x=y, y=data, ax=ax, hue=y, palette=color_dict, legend=False, orient='v')

        ax.axis('off')  # Убираем оси для компактности
        plt.tight_layout(pad=0)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        plt.close(fig)

        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f'<img src="data:image/png;base64,{encoded}"/>'


    # Сбор метрик по каждому признаку
    report_data = []

    for col in X_train.columns:
        feature_vec = X_train[col]

        # Генерируем мини-картинки
        dist_img = generate_sparkline(feature_vec, plot_type='dist')
        target_img = generate_sparkline(
            feature_vec, y=y_train, plot_type='target'
        )

        report_data.append({
            'Feature': col,
            'Mean': feature_vec.mean(),
            'Std': feature_vec.std(),
            'Skewness': feature_vec.skew(),
            'Distribution': dist_img,
            'vs Target (<span style="color:#5E93CF">■ 0</span> / <span style="color:#F80012">■ 1</span>)': target_img,
        })

    df_report = pd.DataFrame(report_data)

    # Экспорт в интерактивный HTML-файл с поиском и сортировкой через DataTables
    html_content = df_report.to_html(
        escape=False, index=False, classes='display compact'
    )

    html_document = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
        <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.xjs"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
            table {{ width: 100%; font-size: 13px; }}
            th {{ background-color: #f1f3f5; }}
        </style>
    </head>
    <body>
        <h2>Feature EDA Overview (512 Features)</h2>
        {html_content}
        <script>
            $(document).ready(function() {{
                $('table').DataTable({{
                    "pageLength": 25,
                    "lengthMenu": [10, 25, 50, 100, 500]
                }});
            }});
        </script>
    </body>
    </html>
    """

    # Сохраняем и открываем в браузере
    with open('features_overview.html', 'w', encoding='utf-8') as f:
        f.write(html_document)

    print('Отчет сохранен в файл features_overview.html')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
