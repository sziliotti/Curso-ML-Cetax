# Convolutional Neural Network

# Instalando Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalando Tensorflow
# Instalando do site: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Instalando Keras
# pip install --upgrade keras

# Parte 1 - Criando a CNN

# Importando as bibliotecas e Classes
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializando a Rede
classifier = Sequential()

# Passo 1 - Convolution
# 32, 3, 3 - Número de filtros, Dimensão dos filtros (3x3)
# input shape = 3 camadas (RGB), 64x64 pixels em cada
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Passo 2 - Pooling
# 2x2 - Tamanho da forma de pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adicionando uma segunda camada de convolução e maxpooling
classifier.add(Convolution2D(32, 3, 3, activation = 'relu')) # não é necessário informar input_shape
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - Full Connection
# Output_dim = Número de neurônios na camada escondida = 128
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) # output layer, sigmoid function, binary outcome

# Compilando a CNN
# adam - um tipo de descida de gradiente estocástico
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Parte 2 - Treinando a rede nas imagens
# Image augmentation - Usado para evitar overfitting
# Image augmentation cria um novo batch de imagens com pequenas mudanças, 
# de forma a aumentar virtualmente o número de imagens
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, # Cria imagens com tamanhos diferentes
                                   shear_range=0.2, # Transformações geométricas
                                   zoom_range=0.2, # Zoom randômico 
                                   horizontal_flip=True) # Flipping

test_datagen = ImageDataGenerator(rescale=1./255) # É apenas necessário fazer rescale

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64), # Dimensão esperada pela CNN
                                                 batch_size=32, # Número de imagens simultâneas na CNN
                                                 class_mode='binary') # 2 classes

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

# Treinando a rede
classifier.fit_generator(training_set,
                         steps_per_epoch=8000/32, # samples_per_epoch/batch_size
                         epochs=1,
                         validation_data=test_set,
                         validation_steps=2000/32)
