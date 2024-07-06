from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv("pythonwork/regression/고용노동부 연도별 최저임금.csv", encoding='cp949')
#dataset = pd.read_excel("pythonwork/regression/test.xlsx", engine='openpyxl')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
print(X)
print(Y)

poly_reg = PolynomialFeatures(degree=20)
X_poly = poly_reg.fit_transform(X)
X_poly = X_poly.tolist()

lin_reg = LinearRegression()
lin_reg.fit(X_poly, Y)

X_range = np.arange(min(X), max(X)+0.1, 0.1).reshape(-1, 1)

y_pred = lin_reg.predict(poly_reg.fit_transform(X_range).tolist())
plt.scatter(X, Y, color='blue')
plt.plot(X_range, y_pred, color='green')
plt.title('Polynomial')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

x = int(input("X : "))
x_poly = poly_reg.fit_transform([[x]])
y = lin_reg.predict(x_poly)
print("예측값 :", *y)
