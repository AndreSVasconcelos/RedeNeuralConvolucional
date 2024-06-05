# Libs
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras.api
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras._tf_keras.keras.preprocessing.image import load_img, ImageDataGenerator

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
                                                class_mode='categorical')