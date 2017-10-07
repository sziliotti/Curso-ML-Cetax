#Data preprocessing

#Importando as libs necessárias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importando o dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Tratando missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

"""
#Tratando missing data - Pandas
X = dataset.iloc[:,:-1] # Deve ser um Dataframe
y = dataset.iloc[:,-1]
X['Age'].fillna(X['Age'].mean(), inplace=True)
X['Salary'].fillna(X['Salary'].mean(), inplace=True)
"""

#Tratando categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

"""
#Tratando categorical data - Pandas
X = dataset.iloc[:,:-1] # Deve ser um Dataframe
y = dataset.iloc[:,-1]
dummies_X = pd.get_dummies(X['Country'])
X.drop(['Country'], axis=1, inplace=True)
X = X.join(dummies_X)
y = y.map({'Yes': 1, 'No': 0})
"""

#Dividindo em connjunto de treino e conjunto de teste
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)