import pandas as pd
import numpy as np
import re

data = pd.read_csv(r'C:\Users\Sunshine\Downloads\datasets\Placement_Data_Full_Class.csv')

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
data["gender"] = label.fit_transform(data["gender"])
data["ssc_b"] = label.fit_transform(data["ssc_b"])
data["hsc_b"] = label.fit_transform(data["hsc_b"])
data["hsc_s"] = label.fit_transform(data["hsc_s"])
data["degree_t"] = label.fit_transform(data["degree_t"])
data["workex"] = label.fit_transform(data["workex"])
data["specialisation"] = label.fit_transform(data["specialisation"])
data["status"] = label.fit_transform(data["status"])

data["salary"] = data["salary"].fillna(0)

X_new = data.iloc[:, [2, 4, 7, 10, 12]].values
y = data.iloc[:, 13:15].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2)
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

import pickle

pickle.dump(clf, open('clf.pkl', 'wb'))

model = pickle.load(open('clf.pkl','rb'))