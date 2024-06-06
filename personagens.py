# Libs
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras.api
import tensorflow as tf
import numpy as np
import seaborn as sns
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras._tf_keras.keras.preprocessing.image import load_img, ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix

#load_img(r'./dados/training_set/bart/bart100.bmp')

# Criando as bases de dados baseadas nas imagens
gerador_treinamento = ImageDataGenerator(rescale=1. / 255,
                                         rotation_range=7,
                                         horizontal_flip=True,
                                         zoom_range=0.2)

base_treinamento = gerador_treinamento.flow_from_directory('./dados/training_set',
                                                           target_size=(64, 64),
                                                           batch_size=8,
                                                           class_mode='categorical')


gerador_teste = ImageDataGenerator(rescale=1. / 255)

base_teste = gerador_teste.flow_from_directory('./dados/test_set',
                                                target_size=(64, 64),
                                                batch_size=8,
                                                class_mode='categorical',
                                                shuffle=False)

# Construção e treinamento da rede neural
rede_neural = Sequential()
# Criando camadas de convolução e pooling
rede_neural.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation='relu'))
rede_neural.add(MaxPooling2D(pool_size=(2,2)))
rede_neural.add(Conv2D(32, (3,3), activation='relu'))
rede_neural.add(MaxPooling2D(pool_size=(2,2)))
# Adicionando camada de Flatenning (Vetoriza a Matriz)
rede_neural.add(Flatten())
# Criação da rede neural tradicional (2 camadas ocultas com 4 neuronios cada e dois na saída)
rede_neural.add(Dense(units=4, activation='relu'))
rede_neural.add(Dense(units=4, activation='relu'))
rede_neural.add(Dense(units=2, activation='softmax'))
# Compilar a rede
rede_neural.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Realizar treinamento
rede_neural.fit(base_treinamento, epochs=20, validation_data=base_teste)

# Avaliar a rede neural
prev = rede_neural.predict(base_teste)
previsoes = np.argmax(prev, axis=1)
print(accuracy_score(previsoes, base_teste.classes))
cm = confusion_matrix(previsoes, base_teste.classes)
sns.heatmap(cm, annot=True)