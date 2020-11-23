import pandas as pd
data = pd.read_csv("Admission_Prediction.csv")
data = data.drop(columns=["Serial No."])

for col in data.columns:
    if data[col].isnull().sum()!=0:
        data[col] = data[col].fillna(data[col].mean())

X = data.iloc[:, :-1]
y = data.iloc[:,-1]

# from statsmodels.stats.outliers_influence import variance_inflation_factor
# vif = pd.DataFrame()
# vif["VIF"] = [ variance_inflation_factor(X, X.columns[i]) for i in range(X.shape[1])]
# vif["Features"] = X.columns

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

import pickle
filename = "my_lr.pickle"
pickle.dump(lr, open(filename, "wb"))