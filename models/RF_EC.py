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
train_idx, test_idx = skm.train_test_split(idx)
x_train, y_train = x_data[train_idx, :], y_data[train_idx]
x_test, y_test = x_data[test_idx, :], y_data[test_idx]

clf = RandomForestClassifier(random_state=0)
parameters = {"criterion" :["gini", "entropy"],
              "max_depth" : list(range(15, 45, 10)),
              "min_samples_split" : list(range(2, 17, 5)),
              "min_samples_leaf" : list(range(2, 17, 5)),
              "n_estimators" : list(range(100, 300, 100)),
              "max_features" :["sqrt", "log2"]}

scorer = make_scorer(f1_score)
grid_search = skm.GridSearchCV(clf, parameters, scoring=scorer, n_jobs=-1)
results = grid_search.fit(x_train, y_train)

joblib.dump(grid_search, '/projects/emco4286/data/stats/grid_search_model.joblib')
joblib.dump(results, '/projects/emco4286/data/stats/results.joblib')