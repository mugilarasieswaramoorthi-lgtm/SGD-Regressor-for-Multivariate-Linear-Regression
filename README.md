# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start and Import Required Libraries

2.Load and Prepare the Dataset

3.Split the Dataset into Training and Testing Sets

4.Scale the Features and Train the Model using SGDRegressor

5.Predict, Evaluate, and Display the Model Performance 


## Program:
```

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Mugilarasi E
RegisterNumber:25017644


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Assignments_Completed': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'Sleep_Hours': [8, 7, 6, 7, 6, 5, 5, 6, 4, 4],
    'Scores': [50, 55, 60, 65, 70, 72, 78, 82, 85, 88]
}

df = pd.DataFrame(data)
print(df)

X = df[['Hours_Studied', 'Assignments_Completed', 'Sleep_Hours']].values
Y = df['Scores'].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd = SGDRegressor(
    max_iter=1000,  
    learning_rate='invscaling',
    eta0=0.01, 
    penalty='l2', 
    random_state=0
)

sgd.fit(X_train_scaled, Y_train)

Y_pred = sgd.predict(X_test_scaled)

print("\n Coefficients:", sgd.coef_)
print(" Intercept:", sgd.intercept_)
print(" Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
print(" R2 Score:", r2_score(Y_test, Y_pred))  

```

## Output:
<img width="693" height="376" alt="Screenshot 2025-10-06 201827" src="https://github.com/user-attachments/assets/633aaa5d-b741-43a1-90d8-f1ea2c44064f" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
