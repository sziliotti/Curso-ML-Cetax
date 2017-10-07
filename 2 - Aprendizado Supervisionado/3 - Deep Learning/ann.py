# Artificial Neural Network

# Instalando Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalando Tensorflow
# Instale a partir do site: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Instalando Keras
# pip install --upgrade keras

# Parte 1 - Data Preprocessing

# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando o dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # Encoding country
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # Encoding gender
onehotencoder = OneHotEncoder(categorical_features = [1]) # Criando Dummy variable
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Removendo uma das Dummies para evitar Trap

# Dividindo em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Parte 2 - Criando a Rede Neural

# Importando as bibliotecas e classes
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializando a rede
classifier = Sequential()

# Adicionando a primeira camada escondida
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) # 6 neurônios, é a segunda camada escondida

# Adicionando a segunda camada escondida
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) # não é necessário informar a camada de saída

# Adicionando a camada de saída
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) # Uma saída, utilizando ativação sigmoid para probabilidades

# Compilando a rede
# adam = Um tipo de descida de gradiente estocástico
# metrics = accuracy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Treinando a rede
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Parte 3 - Fazendo predições e conferindo métricas

# Prevendo os resultados
y_pred = classifier.predict(X_test) # Retorna probabilidades
y_pred = (y_pred > 0.5) # Retorna 1 para maior que 0,5 e 0 caso contrário

# Criando a matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

