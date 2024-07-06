from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("pythonwork/regression/고용노동부 연도별 최저임금.csv", encoding='cp949') #csv 파일 열기 
#dataset = pd.read_excel("pythonwork/regression/test.xlsx", engine='openpyxl') #excel 파일 열기
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

reg = LinearRegression()
reg.fit(X, Y)

y_pred = reg.predict(X)

plt.scatter(X, Y, color='blue')
plt.plot(X, y_pred, color='green')
plt.title('Linear')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

x = int(input("X : "))
y = reg.predict([[x]])
print("예측값 :", *y)
