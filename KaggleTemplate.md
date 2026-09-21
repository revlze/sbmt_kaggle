### To-do
- SVC поперебирать параметры
- Сделать веса побольше на аггрегиванные фичи


### Notes
- ~10% объектов в train, val и public с ошибочными метками
- в ответе нужны вероятности в [0, 1], а не классы
- row_id не фича
- обязательно надо сравнить 3 модели: decision tree, random forest, gradient-boosting
- стандартизация, нормализация данных (?)
- данные в train, test, val распределены одинково, как показал adversarial validation, roc auc score выдает ≈0.5 во всех vs

### Questions
- pass

### Solution

| Approach | CV  | CV STD | LB  | Date |
| -------- | --- | ------ | --- | ---- |
| aggregate_features + log1p + StandardScaler + SVC(C=10, kernel='rbf')  |   0.94689  |   ...   |   0.93654  |   Sun Sep 20 2026 11:50:15 |

### Annotations

| №   | Annotation |
| --- | ---------- |
|     |            |

### Final Ensemble

| Model | CV  | Public LB | Private LB |
| ----- | --- | --------- | ---------- |
|       |     |           |            |

### Processed ideas
#### Good
- SVC
- aggregating fetures

#### Neutral
- Поделить фичи пополам, перекинуть между собой, докинуть снизу в датасет

#### Bad
- pass
#### Some experimental ideas I didn't implemented
- pass
