# Краткая история экспериментов SBMTeam

Ниже перечислены основные направления, которые команда проверяла до фиксации финальных моделей. Это не полный журнал всех сеток параметров, а краткое описание разных идей и причин, по которым они были полезны для сравнения.

1. **Обычные деревья на 512 исходных признаках.** Проверили неглубокое дерево решений, Random Forest, sklearn GBDT и LightGBM. Дерево и лес дали понятные нижние baselines; бустинги были заметно сильнее, но уступали итоговым SVC. В сохранённом запуске validation ROC-AUC был примерно 0.788 / 0.885 / 0.917 / 0.917 соответственно.

2. **Деревья после компактного feature engineering.** К исходным координатам добавляли геометрические и агрегатные признаки. На расширенном наборе особенно выросли GBDT и LightGBM (примерно до 0.930 и 0.933 validation ROC-AUC), что подтвердило полезность дополнительной структуры, даже если деревья не стали финальным решением.

3. **Простой RBF SVC.** Базовая цепочка `log1p → StandardScaler → RBF SVC` использовалась как главный контраст: сильная модель без специальных блоков, custom kernels и prediction blending.

4. **Weighted/selected RBF SVC.** Пробовали univariate-веса признаков по AUC или separation statistic, `top_k`-отбор, разные `C` и `gamma`, включая компактный Optuna-поиск. Идея была полезна для диагностики, но большого устойчивого выигрыша над хорошим raw RBF не дала.

5. **Leakage-safe supervised feature blocks.** Строили geometry, density, coordinate likelihood, kNN и aggregates строго внутри train-части каждого outer fold. Отдельно проверяли PCA/Factor Analysis и качество каждого блока.

6. **Структурные и custom-kernel эксперименты.** Исследовали coordinate evidence, magnitude/direction decomposition, product и additive kernels, learned RBF metric, low-rank/diagonal weighting, permutation и block-sorting diagnostics.Эти опыты помогли отказаться от нестабильных направлений и оставить additivemulti-view kernel.

7. **Bagging, augmentation и label cleaning.** Проверяли bagged SVC, генерацию/аугментацию, CleanLearning/Cleanlab и варианты stacking. Они не дали достаточно стабильного прироста для усложнения финального offline-пакета.

8. **Fine6 multi-kernel SVC.** Смешали raw, geometry, density, aggregates, kNN и coordinate kernels с фиксированными gamma. Fine6 стал сильным историческим anchor и основой дальнейших парных сравнений.

9. **Whitening и complementary auxiliary SVC.** Для малых engineered-блоков сравнивали original, PCA-whitening и family representations. Отдельная модель `0.75·PCA128 + 0.25·aggregates_white4` оказалась слабее основной сама по себе, но полезной благодаря отличающемуся ranking.

10. **Rescue K0/T6.** Исправили выбор представлений и raw gamma, затем сравнили ограниченный набор K0–K9. Финальный Rescue T6 — rank blend `0.40·Fine6 + 0.40·corrected K0 + 0.20·clean_aux`.

11. **Sqrt-raw last push.** Добавили отдельную метрику расстояния на стандартизованном `sqrt(raw)`. S100 заменил log-raw часть на sqrt-raw, а NEW5 объединил `0.80·S100 + 0.20·clean_aux`. Эта модель выбрана designated submission.

## Что оставлено в исполняемом сравнении

Чтобы итоговый notebook оставался коротким и укладывался в лимиты, он запускает ровно десять репрезентативных конфигураций. Три древесные модели получают общий расширенный leakage-safe набор из raw, geometry, density, aggregates, coordinate, kNN и PCA128 признаков; raw-only вариант дерева отдельно не добавляется. Только простой RBF SVC намеренно остаётся на исходных 512 признаках как понятная точка отсчёта:

1. дерево решений на расширенных признаках;
2. случайный лес на расширенных признаках;
3. GBDT на расширенных признаках;
4. простой raw RBF SVC;
5. Fine6 SVC;
6. corrected K0 SVC без blending;
7. S100 SVC без blending;
8. вспомогательный clean auxiliary SVC;
9. Rescue T6;
10. финальный NEW5.

LightGBM, Optuna, Cleanlab, augmentation и широкие экспериментальные grids в финальный executable notebook намеренно не перенесены: они отражены здесь как проведённые исследования, но потребовали бы лишних зависимостей или времени.
