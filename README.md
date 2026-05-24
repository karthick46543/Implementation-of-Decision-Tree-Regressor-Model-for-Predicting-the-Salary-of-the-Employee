# Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load and preprocess the salary dataset by encoding categorical values into numerical format.
2.Split the dataset into training and testing sets for model evaluation.
3.Train a Decision Tree Regressor model using employee position and level features.
4.Predict salaries and evaluate model performance using the R² score.
```


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: RISHI BALA KARTHICK K
RegisterNumber: 212225040337
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
 Load dataset
csv_path = r"C:\Users\acer\Downloads\Salary.csv"   # <-- Change path if needed
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at: {csv_path}. Update the path.")

print("Dataset Loaded Successfully!\n")
 Data exploration
print("Shape:", data.shape)
display(data.head())
print("\nInfo:")
display(data.info())
print("\nMissing Values:\n", data.isnull().sum())

 Encode categorical column 'Position'
if "Position" in data.columns:
    le = LabelEncoder()
    data["Position"] = le.fit_transform(data["Position"])
    print("\nLabel Encoding Mapping (Position):")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

Select features and target
X = data[["Position", "Level"]]
y = data["Salary"]

print("\nFeature Sample:")
display(X.head())
print("\nTarget Sample:")
display(y.head())

Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
print(f"\nTrain Size: {X_train.shape}, Test Size: {X_test.shape}")

 Initialize and train Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=10)
dt.fit(X_train, Y_train)
print("\nModel Training Completed!")

 Predict on test data
y_pred = dt.predict(X_test)
print("\nPredicted Salaries:", y_pred)

 Evaluate model using R² Score
r2 = metrics.r2_score(Y_test, y_pred)
print(f"\nR² Score: {r2:.4f}")

Visualize Decision Tree
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=["Position", "Level"], filled=True)
plt.title("Decision Tree Regressor for Salary Prediction")
plt.show()

Feature Importances
importances = pd.Series(dt.feature_importances_, index=["Position", "Level"])
print("\nFeature Importances:")
display(importances)
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="1167" height="822" alt="image" src="https://github.com/user-attachments/assets/b2ed3fa8-4c1b-43df-aea8-540de03d4b79" />
<img width="637" height="577" alt="image" src="https://github.com/user-attachments/assets/a2aa377d-494a-41e3-9815-03daa8b24c9c" />
<img width="1197" height="847" alt="image" src="https://github.com/user-attachments/assets/cf0b4e94-cfa9-4268-8654-a3116abeea68" />





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
