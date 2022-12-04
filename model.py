import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("data/SPARCS-2015-TRAIN.csv")

x = dataset.iloc[:, :11]
y = dataset.iloc[:, -1]

xgb = XGBClassifier()
xgb.fit(x.values, y.values)

y_pred = xgb.predict(x)

print(classification_report(y, y_pred))

pickle.dump(xgb, open('model/model.pkl','wb'))
