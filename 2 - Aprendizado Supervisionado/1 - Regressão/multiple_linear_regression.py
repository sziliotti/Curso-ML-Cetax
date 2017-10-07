#Multiple Linear Regression

# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importando o dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Tratando dados categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Evitando a Dummy Variable Trap
X = X[:, 1:]

# Dividindo em conjunto de treino e testes
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Criando o modelo de regressão linear múltipla
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predizendo os resultados no test set
y_pred = regressor.predict(X_test)

# Backward Elimination para o modelo ótimo

# Passo 1 - Selecionando o Significance Level de 0.05
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) # Adicionando uma coluna de 1's
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # Passo 2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Passo 2
regressor_OLS.summary() # Gera uma tabela no console, procurar valores P

# Repetindo o procedimento sem o índice de maior valor P
X_opt = X[:, [0, 1, 3, 4, 5]] # Passo 4 - Removendo o preditor
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Passo 5 - Treinando o novo modelo
regressor_OLS.summary() # Gera uma tabela no console, procurar valores P

X_opt = X[:, [0, 3, 4, 5]] # Passo 4 - Removendo o preditor
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Passo 5 - Treinando o novo modelo
regressor_OLS.summary() # Gera uma tabela no console, procurar valores P

X_opt = X[:, [0, 3, 5]] # Passo 4 - Removendo o preditor
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Passo 5 - Treinando o novo modelo
regressor_OLS.summary() # Gera uma tabela no console, procurar valores P

X_opt = X[:, [0, 3]] # Passo 4 - Removendo o preditor
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Passo 5 - Treinando o novo modelo
regressor_OLS.summary() # Gera uma tabela no console, procurar valores P