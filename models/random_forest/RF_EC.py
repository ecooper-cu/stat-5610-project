import os
import pathlib
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as skm

data = pd.read_csv(os.path.join(pathlib.Path.home(), "stat-5610-project", "data", "train.csv"))
x_data = np.array(data[data.columns.drop("Y")].values)
y_data = data["Y"].values

# Train/ test split
idx = list(range(len(y_data)))
train_idx, test_idx = skm.train_test_split(idx, random_state=1)
x_train, y_train = x_data[train_idx, :], y_data[train_idx]
x_test, y_test = x_data[test_idx, :], y_data[test_idx]

clf = RandomForestClassifier(random_state=1)
parameters = {"criterion" :["gini", "entropy"],
              "max_depth" : list(range(5, 65, 10)), # 6
              "min_samples_split" : list(range(2, 30, 4)), # 7
              "min_samples_leaf" : list(range(2, 30, 4)), # 7
              "min_impurity_decrease" : list(np.arange(0, 0.1, 0.02)), # 5
              'max_samples': list(np.arange(0.3, 1, 0.1)), # 7
              "n_estimators" : list(range(100, 600, 100)), # 5
              "max_features" :["sqrt", "log2"]}

scorer = make_scorer(f1_score)
grid_search = skm.GridSearchCV(clf, parameters, scoring=scorer, n_jobs=-1)
results = grid_search.fit(x_train, y_train)

joblib.dump(grid_search, '/projects/emco4286/data/stats/try5/grid_search_model.joblib')
joblib.dump(results, '/projects/emco4286/data/stats/try5/results.joblib')