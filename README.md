# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Developed by:HEMAPRIYA K 
RegisterNumber: 212223040066
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)

print(df.head())
print(df.tail())

x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## DATASET
![416961070-086152da-3dd8-4966-b11d-0b75cfe21f03](https://github.com/user-attachments/assets/4536e1c7-54c2-49d2-99e0-3258fc3ce31e)

## HEAD VALUES
![416961128-622b828b-eba6-4aef-a6aa-22e274a4f704](https://github.com/user-attachments/assets/acc49f7c-0f72-4999-9210-7ada0552d25c)

## TAIL VALUES
![416961175-72767627-4499-4524-88e3-090b1051a565](https://github.com/user-attachments/assets/523a21b6-fdf3-4946-8b7a-308301bdf40d)

## X and Y VALUES
![416961263-b49303b5-0a6f-4897-b8b9-4583ac7eb5e7](https://github.com/user-attachments/assets/e7a5cc02-22ba-493a-9bc8-78d79259fc2d)


## PREDICTION VALUES OF X AND Y


![416961333-e6a1246a-7836-4dd9-8de1-8fe3595efce1](https://github.com/user-attachments/assets/1e38a4a8-8377-4dda-8d04-6b33ecd654d8)

## TRAINING SET
![416961391-9407cc8d-ca0e-4e20-ba2b-b605a5f703cc](https://github.com/user-attachments/assets/030aed6c-db52-4b96-8366-d09cb51023f6)

## TESTING SET AND MSE,MAE and RMSE
![416961453-b4335ae9-f9b0-4884-ab62-2fbe37e4e54a](https://github.com/user-attachments/assets/e38b0f3e-d833-487d-9501-f976ba11c6c8)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
