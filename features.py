import marimo

__generated_with = "0.24.2"
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
    return X_train, X_val, y_train, y_val


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
    target_corr = X_train.corrwith(y_train, method='pearson')

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Кручу-верчу фичи
    """)
    return


@app.cell
def _(X_train: "pd.DataFrame", np, pd, sns, y_train: "pd.Series"):
    from sklearn.decomposition import PCA

    _pca = PCA(n_components=1, random_state=42)
    X_pca_1 = _pca.fit_transform(X_train)

    # Применяем строгий стиль с темной сеткой
    sns.set_style("darkgrid")
    palette = {0: '#5E93CF', 1: '#F80012'}

    # 1. Генерируем новые признаки (агрегации по строкам)
    X_eng = pd.DataFrame(index=X_train.index)

    # X_eng['row_sum'] = X_train.sum(axis=1)   хуже лог-суммы                           # Аналог L1-расстояния (Манхэттенское)
    # X_eng['row_l2'] = np.sqrt((X_train ** 2).sum(axis=1))  хуже лог-суммы             # Аналог L2-расстояния (Евклидово)
    # X_eng['row_std'] = X_train.std(axis=1)          хуже PCA                    # Разброс значений внутри вектора
    X_eng['row_max'] = X_train.max(axis=1)                                # Максимальный "выброс" в различиях
    X_eng['row_zeros'] = (X_train == 0).sum(axis=1)                       # Количество точных совпадений (нулей)
    X_eng['row_log1p_sum'] = np.log1p(X_train).sum(axis=1)                # Сумма с подавлением тяжелых хвостов
    X_eng['row_q95'] = np.quantile(X_train, 0.95, axis=1)
    X_eng['row_near_zero_05'] = (X_train < 0.5).sum(axis=1)
    # X_eng['row_pca1'] = X_pca_1.flatten()            хуже лог-суммы             # Первая главная компонента (PCA)

    # Считаем корреляцию новых признаков с таргетом
    eng_corr = X_eng.corrwith(y_train, method='spearman').sort_values(key=abs, ascending=False)
    print("Корреляция новых признаков с таргетом:\n", eng_corr)
    return PCA, X_eng, palette


@app.cell
def _(X_eng, palette, plt, sns, y_train: "pd.Series"):
    # 2. Визуализация распределений по классам
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col_ in enumerate(X_eng.columns):
        sns.kdeplot(
            data=X_eng, 
            x=col_, 
            hue=y_train, 
            palette=palette, 
            fill=True, 
            alpha=0.5, 
            ax=axes[i],
            common_norm=False
        )
        axes[i].set_title(f'Distribution of {col_}', fontweight='bold', fontsize=12)
        axes[i].set_xlabel(col_, fontweight='bold')
        axes[i].set_ylabel('Density', fontweight='bold')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_eng, palette, plt, sns, y_train: "pd.Series"):
    from itertools import combinations

    # Объединяем признаки и таргет в один DataFrame
    df_plot = X_eng.copy()
    df_plot['target'] = y_train

    # Получаем все уникальные пары из 6 признаков (всего 15 пар)
    feature_pairs = list(combinations(X_eng.columns, 2))


    # Создаем сетку 5x3
    fig_3, axes_3 = plt.subplots(5, 3, figsize=(13, 18))
    axes_3 = axes_3.flatten()

    for i_, (x_col, y_col) in enumerate(feature_pairs):
        sns.scatterplot(
            data=df_plot,
            x=x_col,
            y=y_col,
            hue='target',
            palette=palette,
            alpha=0.6,
            s=10,                   # Размер точек (увеличен, так как объектов мало)
            edgecolor='black',      # Обводка точек для четкости
            linewidth=0.1,
            ax=axes_3[i_]
        )
        axes_3[i_].set_title(f'{x_col} vs {y_col}', fontweight='bold', fontsize=11)
        axes_3[i_].set_xlabel(x_col, fontweight='bold', fontsize=9)
        axes_3[i_].set_ylabel(y_col, fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PCA - слабый результат
    """)
    return


@app.cell
def _(PCA, X_train: "pd.DataFrame", pd, y_train: "pd.Series"):
    # Выделяем первые 2 компоненты для графика
    pca = PCA(n_components=1, random_state=42)
    X_pca_2 = pca.fit_transform(X_train)

    pca_df = pd.DataFrame(X_pca_2, columns=['PC1'])
    pca_df['Target'] = y_train.values

    # Доля объясненной дисперсии по каждой компоненте
    var_ratio = pca.explained_variance_ratio_

    print(f"PC1 объясняет: {var_ratio[0]:.2%}")
    # Для моделирования стоит взять больше компонент (например, объясняющих 80% дисперсии)
    # pca_full = PCA(n_components=0.80, random_state=42)
    # X_pca_full = pca_full.fit_transform(X_train)
    # print(f"Оставлено компонент для 80% дисперсии: {X_pca_full.shape[1]}")
    return


@app.cell
def _(PCA, X_train: "pd.DataFrame", np, pd, sns, y_train: "pd.Series"):
    sns.set_style("darkgrid")

    # 1. Обучаем PCA на всех компонентах и трансформируем данные
    pca_full = PCA(random_state=42)
    X_pca_all = pca_full.fit_transform(X_train)

    # Создаем DataFrame с названиями компонент PC1, PC2...
    pc_names = [f'PC{i+1}' for i in range(X_pca_all.shape[1])]
    df_pca = pd.DataFrame(X_pca_all, columns=pc_names, index=X_train.index)

    # 2. Считаем корреляцию Спирмена каждой компоненты с таргетом
    pca_spearman = df_pca.corrwith(y_train, method='spearman')

    # 3. Расчет кумулятивной дисперсии
    var_ratio_ = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio_)

    n_80 = np.argmax(cum_var >= 0.80) + 1
    n_90 = np.argmax(cum_var >= 0.90) + 1

    # 4. Вывод текстовой информации по первыми компонентам
    print("=== Доля дисперсии и Корреляция Спирмена с таргетом ===")
    pca_summary = pd.DataFrame({
        'Explained Var Ratio': var_ratio_,
        'Cumulative Var Ratio': cum_var,
        'Spearman Corr with Target': pca_spearman
    }, index=pc_names)

    print(pca_summary.head(10).round(4))
    print("\n" + "="*50)
    print(f"Компонент для 80% дисперсии: {n_80}")
    print(f"Компонент для 90% дисперсии: {n_90}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Слияние
    """)
    return


@app.cell
def _(
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    y_train: "pd.Series",
    y_val: "pd.Series",
):
    X_tr, y_tr = X_train.copy(), y_train.copy()
    X_valid, y_valid = X_val.copy(), y_val.copy()
    return X_tr, X_valid, y_tr, y_valid


@app.cell
def _():
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.preprocessing import StandardScaler
    from sklearn.random_projection import SparseRandomProjection

    return FeatureAgglomeration, SparseRandomProjection, StandardScaler


@app.cell
def _():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.base import clone

    def evaluate_lgbm(X_tr, y_tr, X_valid, y_valid, base_model, exp_name="Experiment"):
        model = clone(base_model)
    
        model.fit(
            X_tr,
            y_tr,
            eval_X=(X_tr, X_valid), eval_y=(y_tr, y_valid),
            eval_names=["train", "val"],
            eval_metric="auc"
        )

        y_tr_pred = model.predict_proba(X_tr)[:, 1]
        y_val_pred = model.predict_proba(X_valid)[:, 1]

        train_auc = roc_auc_score(y_tr, y_tr_pred)
        val_auc = roc_auc_score(y_valid, y_val_pred)

        print(f"=== {exp_name} ===")
        print(f"Features count:     {X_tr.shape[1]}")
        print(f"Train ROC-AUC:      {train_auc:.5f}")
        print(f"Validation ROC-AUC: {val_auc:.5f}")
        print(f"Overfit (Train-Val):{train_auc - val_auc:.5f}\n")
    
        return val_auc, model

    # Базовая конфигурация LightGBM
    base_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=-1,
        random_state=42
    )
    return base_model, evaluate_lgbm


@app.cell
def _(X_tr, X_valid, base_model, evaluate_lgbm, y_tr, y_valid):
    base_val_auc, fitted_base_model = evaluate_lgbm(
        X_tr, y_tr, X_valid, y_valid, base_model, exp_name="0. Baseline (512 Features)"
    )
    return (fitted_base_model,)


@app.cell
def _(
    FeatureAgglomeration,
    StandardScaler,
    X_tr,
    X_valid,
    base_model,
    evaluate_lgbm,
    y_tr,
    y_valid,
):
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_valid)

    agg = FeatureAgglomeration(n_clusters=50)
    X_tr_agg = agg.fit_transform(X_tr_scaled)
    X_val_agg = agg.transform(X_val_scaled)

    evaluate_lgbm(
        X_tr_agg, y_tr, X_val_agg, y_valid, base_model, exp_name="1. Feature Agglomeration (50 Meta-Features)"
    );
    return X_tr_agg, X_tr_scaled, X_val_agg, X_val_scaled


@app.cell
def _(
    SparseRandomProjection,
    X_tr_scaled,
    X_val_scaled,
    base_model,
    evaluate_lgbm,
    y_tr,
    y_valid,
):
    srp = SparseRandomProjection(n_components=32, random_state=42)
    X_tr_srp = srp.fit_transform(X_tr_scaled)
    X_val_srp = srp.transform(X_val_scaled)

    evaluate_lgbm(
        X_tr_srp, y_tr, X_val_srp, y_valid, base_model, exp_name="2. Sparse Random Projection (32 Features)"
    );
    return


@app.cell
def _(
    X_tr,
    X_tr_agg,
    X_val_agg,
    X_valid,
    base_model,
    evaluate_lgbm,
    fitted_base_model,
    np,
    pd,
    y_tr,
    y_valid,
):
    # Берем важность признаков из базовой обученной модели
    feature_importances = fitted_base_model.feature_importances_
    top_30_indices = np.argsort(feature_importances)[::-1][:30]

    # Выделяем Top-30 фичей
    X_tr_top = X_tr.iloc[:, top_30_indices].values if isinstance(X_tr, pd.DataFrame) else X_tr[:, top_30_indices]
    X_val_top = X_valid.iloc[:, top_30_indices].values if isinstance(X_valid, pd.DataFrame) else X_valid[:, top_30_indices]

    # Склеиваем Top-30 и 32 агломерированные мета-фичи
    X_tr_hybrid = np.hstack([X_tr_top, X_tr_agg])
    X_val_hybrid = np.hstack([X_val_top, X_val_agg])

    evaluate_lgbm(
        X_tr_hybrid, y_tr, X_val_hybrid, y_valid, base_model, exp_name="3. Hybrid (Top-30 + 32 Agglomerated)"
    );
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Гипотеза провалилась
    """)
    return


@app.cell
def _(X_train: "pd.DataFrame", plt, y_train: "pd.Series"):
    # Берем самый первый объект из X_train
    sample_obj = X_train.iloc[0].values
    sample_target = y_train.iloc[0]

    # Разбиваем на гипотетические форматы
    img_16x16_part1 = sample_obj[:256].reshape(16, 16)
    img_16x16_part2 = sample_obj[256:].reshape(16, 16)
    img_16x32 = sample_obj.reshape(16, 32)

    fig_6, axes_6 = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Первая половина (16x16)
    im0 = axes_6[0].imshow(img_16x16_part1, cmap='magma')
    axes_6[0].set_title('Первые 256 фичей (16x16)', fontweight='bold')
    axes_6[0].axis('off')
    plt.colorbar(im0, ax=axes_6[0], fraction=0.046, pad=0.04)

    # 2. Вторая половина (16x16)
    im1 = axes_6[1].imshow(img_16x16_part2, cmap='magma')
    axes_6[1].set_title('Вторые 256 фичей (16x16)', fontweight='bold')
    axes_6[1].axis('off')
    plt.colorbar(im1, ax=axes_6[1], fraction=0.046, pad=0.04)

    # 3. Полная матрица (16x32)
    im2 = axes_6[2].imshow(img_16x32, cmap='magma')
    axes_6[2].set_title('Все 512 фичей (16x32)', fontweight='bold')
    axes_6[2].axis('off')
    plt.colorbar(im2, ax=axes_6[2], fraction=0.046, pad=0.04)

    plt.suptitle(f'Visualizing Object #0 (Target = {sample_target})', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_train: "pd.DataFrame", plt, y_train: "pd.Series"):
    # Считаем средний вектор для каждого класса
    mean_class_0 = X_train[y_train == 0].mean(axis=0).values
    mean_class_1 = X_train[y_train == 1].mean(axis=0).values

    fig_7, axes_7 = plt.subplots(2, 3, figsize=(14, 8))

    # Класс 0
    axes_7[0, 0].imshow(mean_class_0[:256].reshape(16, 16), cmap='viridis')
    axes_7[0, 0].set_title('Class 0: Part 1 (16x16)', fontweight='bold', color='#5E93CF')
    axes_7[0, 0].axis('off')

    axes_7[0, 1].imshow(mean_class_0[256:].reshape(16, 16), cmap='viridis')
    axes_7[0, 1].set_title('Class 0: Part 2 (16x16)', fontweight='bold', color='#5E93CF')
    axes_7[0, 1].axis('off')

    axes_7[0, 2].imshow(mean_class_0.reshape(16, 32), cmap='viridis')
    axes_7[0, 2].set_title('Class 0: Full (16x32)', fontweight='bold', color='#5E93CF')
    axes_7[0, 2].axis('off')

    # Класс 1
    axes_7[1, 0].imshow(mean_class_1[:256].reshape(16, 16), cmap='viridis')
    axes_7[1, 0].set_title('Class 1: Part 1 (16x16)', fontweight='bold', color='#F80012')
    axes_7[1, 0].axis('off')

    axes_7[1, 1].imshow(mean_class_1[256:].reshape(16, 16), cmap='viridis')
    axes_7[1, 1].set_title('Class 1: Part 2 (16x16)', fontweight='bold', color='#F80012')
    axes_7[1, 1].axis('off')

    axes_7[1, 2].imshow(mean_class_1.reshape(16, 32), cmap='viridis')
    axes_7[1, 2].set_title('Class 1: Full (16x32)', fontweight='bold', color='#F80012')
    axes_7[1, 2].axis('off')

    plt.suptitle('Mean Feature Heatmaps by Class', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_train: "pd.DataFrame", np, plt, y_train: "pd.Series"):
    # 1. Расчет средних векторов по классам
    mean_c0 = X_train[y_train == 0].mean(axis=0).values
    mean_c1 = X_train[y_train == 1].mean(axis=0).values

    # 2. Вычисление разностей каналов (Part 1 - Part 2)
    diff_parts_c0 = mean_c0[:256] - mean_c0[256:]
    diff_parts_c1 = mean_c1[:256] - mean_c1[256:]

    # 3. Вычисление разностей между классами (Class 1 - Class 0)
    diff_class_part1 = mean_c1[:256] - mean_c0[:256]
    diff_class_part2 = mean_c1[256:] - mean_c0[256:]
    diff_class_full  = mean_c1 - mean_c0

    # -------------------------------------------------------------------------
    # Отрисовка сетки разностей
    # -------------------------------------------------------------------------
    fig_8, axes_8 = plt.subplots(2, 3, figsize=(15, 9))

    # --- РЯД 1: Внутриклассовые разности (Part 1 − Part 2) ---
    # Класс 0: Part 1 - Part 2
    lim_0 = np.abs(diff_parts_c0).max()
    _im0 = axes_8[0, 0].imshow(diff_parts_c0.reshape(16, 16), cmap='bwr', vmin=-lim_0, vmax=lim_0)
    axes_8[0, 0].set_title('Class 0: (Part 1 − Part 2)', fontweight='bold', color='#5E93CF')
    axes_8[0, 0].axis('off')
    plt.colorbar(_im0, ax=axes_8[0, 0], fraction=0.046, pad=0.04)

    # Класс 1: Part 1 - Part 2
    lim_1 = np.abs(diff_parts_c1).max()
    _im1 = axes_8[0, 1].imshow(diff_parts_c1.reshape(16, 16), cmap='bwr', vmin=-lim_1, vmax=lim_1)
    axes_8[0, 1].set_title('Class 1: (Part 1 − Part 2)', fontweight='bold', color='#F80012')
    axes_8[0, 1].axis('off')
    plt.colorbar(_im1, ax=axes_8[0, 1], fraction=0.046, pad=0.04)

    # Разность разностей: (Class 1 Diff) − (Class 0 Diff)
    diff_of_diffs = diff_parts_c1 - diff_parts_c0
    lim_dd = np.abs(diff_of_diffs).max()
    im_dd = axes_8[0, 2].imshow(diff_of_diffs.reshape(16, 16), cmap='bwr', vmin=-lim_dd, vmax=lim_dd)
    axes_8[0, 2].set_title('Δ(Part 1−Part 2): Class 1 vs Class 0', fontweight='bold')
    axes_8[0, 2].axis('off')
    plt.colorbar(im_dd, ax=axes_8[0, 2], fraction=0.046, pad=0.04)


    # --- РЯД 2: Межклассовые разности (Class 1 − Class 0) ---
    # Part 1: Class 1 - Class 0
    lim_p1 = np.abs(diff_class_part1).max()
    _im2 = axes_8[1, 0].imshow(diff_class_part1.reshape(16, 16), cmap='bwr', vmin=-lim_p1, vmax=lim_p1)
    axes_8[1, 0].set_title('Part 1: (Class 1 − Class 0)', fontweight='bold')
    axes_8[1, 0].axis('off')
    plt.colorbar(_im2, ax=axes_8[1, 0], fraction=0.046, pad=0.04)

    # Part 2: Class 1 - Class 0
    lim_p2 = np.abs(diff_class_part2).max()
    im3 = axes_8[1, 1].imshow(diff_class_part2.reshape(16, 16), cmap='bwr', vmin=-lim_p2, vmax=lim_p2)
    axes_8[1, 1].set_title('Part 2: (Class 1 − Class 0)', fontweight='bold')
    axes_8[1, 1].axis('off')
    plt.colorbar(im3, ax=axes_8[1, 1], fraction=0.046, pad=0.04)

    # Full: Class 1 - Class 0 (16x32)
    lim_full = np.abs(diff_class_full).max()
    im4 = axes_8[1, 2].imshow(diff_class_full.reshape(16, 32), cmap='bwr', vmin=-lim_full, vmax=lim_full)
    axes_8[1, 2].set_title('Full (16x32): (Class 1 − Class 0)', fontweight='bold')
    axes_8[1, 2].axis('off')
    plt.colorbar(im4, ax=axes_8[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle('Differences Heatmaps Analysis', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_train: "pd.DataFrame", np, plt, y_train: "pd.Series"):
    from scipy.cluster.hierarchy import linkage, leaves_list

    # 1. Считаем матрицу корреляций между фичами
    _corr_matrix = X_train.corr(method='spearman').fillna(0).values

    # 2. Иерархическая кластеризация признаков
    link = linkage(_corr_matrix, method='ward')
    reordered_indices = leaves_list(link)

    # 3. Переупорядочиваем колонки X_train
    X_train_reordered = X_train.iloc[:, reordered_indices]

    # 4. Отрисовываем усредненные объекты заново (уже в сгруппированном виде)
    mean_c0_reordered = X_train_reordered[y_train == 0].mean(axis=0).values
    mean_c1_reordered = X_train_reordered[y_train == 1].mean(axis=0).values

    fig_9, axes_9 = plt.subplots(1, 3, figsize=(15, 5))

    # Класс 0 (16x32)
    _im0 = axes_9[0].imshow(mean_c0_reordered.reshape(16, 32), cmap='viridis')
    axes_9[0].set_title('Reordered Class 0 (16x32)', fontweight='bold')
    axes_9[0].axis('off')

    # Класс 1 (16x32)
    _im1 = axes_9[1].imshow(mean_c1_reordered.reshape(16, 32), cmap='viridis')
    axes_9[1].set_title('Reordered Class 1 (16x32)', fontweight='bold')
    axes_9[1].axis('off')

    # Разность классов (16x32)
    diff_reordered = mean_c1_reordered - mean_c0_reordered
    lim = np.abs(diff_reordered).max()
    _im2 = axes_9[2].imshow(diff_reordered.reshape(16, 32), cmap='bwr', vmin=-lim, vmax=lim)
    axes_9[2].set_title('Reordered Diff (Class 1 − Class 0)', fontweight='bold')
    axes_9[2].axis('off')

    plt.suptitle('Features Reordered by Hierarchical Clustering', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_train: "pd.DataFrame", np, plt, y_train: "pd.Series"):
    # 1. Считаем корреляцию каждого признака с таргетом
    _target_corr = X_train.corrwith(y_train, method='spearman')

    # 2. Сортируем индексы признаков от -1 до +1
    sorted_by_target = _target_corr.sort_values().index
    X_train_sorted_target = X_train[sorted_by_target]

    # 3. Визуализируем разность средних
    diff_target_sorted = (X_train_sorted_target[y_train == 1].mean() - 
                          X_train_sorted_target[y_train == 0].mean()).values

    plt.figure(figsize=(10, 5))
    _lim = np.abs(diff_target_sorted).max()
    plt.imshow(diff_target_sorted.reshape(16, 32), cmap='bwr', vmin=-_lim, vmax=_lim)
    plt.colorbar(label='Mean Difference (Class 1 − Class 0)')
    plt.title('Features Sorted by Correlation with Target (16x32)', fontweight='bold')
    plt.axis('off')
    plt.show()
    return


@app.cell
def _():
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
def _(
    aggregate_features,
    test_df: "pd.DataFrame",
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
):
    X_train_agg = aggregate_features(train_df)
    X_val_agg = aggregate_features(val_df)
    X_test_agg = aggregate_features(test_df)
    return


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
    test_df: "pd.DataFrame",
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
):
    X_train_extended = add_aggregate_features(train_df)
    X_val_extended = add_aggregate_features(val_df)
    X_test_extended = add_aggregate_features(test_df)
    return


if __name__ == "__main__":
    app.run()
