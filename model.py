import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("data/SPARCS-2015-TRAIN.csv")
dataset_val = pd.read_csv("data/SPARCS-2015-VAL.csv")

x = dataset.iloc[:, :11]
y = dataset.iloc[:, -1]

x_val = dataset_val.iloc[:, :11]
y_val = dataset_val.iloc[:, :-1]

xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=12, min_child_weight=1, n_estimators=100, n_jobs=1, nthread=None, objective='binary:logistic', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=None, subsample=1, verbosity=1)
xgb.fit(x.values, y.values)

yt_pred = xgb.predict(x.values)
y_pred = xgb.predict(x_val.values)
print("== Training")
print(classification_report(y.values, yt_pred))
print("== Validation")
print(classification_report(y_val.values, y_pred))

# Saving model to disk
pickle.dump(xgb, open('model/model.pkl','wb'))
