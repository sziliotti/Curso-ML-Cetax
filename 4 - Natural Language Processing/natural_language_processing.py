# Natural Language Processing

# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando o dataset - quoting 3 = ignorando aspas duplas
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Limpando o texto
import re
import nltk
nltk.download('stopwords') # stopwords são preposições, 'this', 'that',...
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #loving = love
corpus = [] # inicializando como uma lista varia
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #substituindo tudo que não for letras por espaços em branco
    review = review.lower() # transformando em letras minúsculas
    review = review.split() # criando um vetor de palavras
    ps = PorterStemmer() # inicializando a classe stem
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # iterando pelo vetor de palavras e excluindo as stopwords
    review = ' '.join(review) # juntando novamente em string
    corpus.append(review) # append no corpus

# Criando o modelo de Bag of Words (matriz esparsa) e tokenizando as palavras chaves
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Dividindo em conjunto de treino e teste
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Criando o modelo de classificação
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Prevendo no conjunto de testes
y_pred = classifier.predict(X_test)

# Criando a matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

(55+91)/(55+91+12+42)