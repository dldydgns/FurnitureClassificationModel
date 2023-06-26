import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import VGG16
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

batchsize = 20

img_width, img_height = 150, 150

train_dir = 'img/train/'
val_dir = 'img/val'

num_train_images = num_val_images = 0

#데이터 갯수 저장
for dirname, _, filenames in os.walk('img'):
    for filename in filenames:
        if dirname[4:].split('\\')[0] == 'train':
            num_train_images += 1
        if dirname[4:].split('\\')[0] == 'val':
            num_val_images += 1
            
#학습세트
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip  = True)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_height,img_width),
                                                    batch_size = batchsize)

#검증세트
val_datagen = ImageDataGenerator(rescale=1./255,
                                 horizontal_flip = True)
val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(img_height,img_width),
                                                batch_size = batchsize)

base_model = VGG16(weights='imagenet',
              include_top=False,
              input_shape = (img_height, img_width, 3)
              )

print(len(base_model.layers))

def build_model(base_model, dropout, fc_layers, num_classes):
  for each_layer in base_model.layers:
    each_layer.trainable = True
  
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)

  # Fine-tune from this layer onwards
  fine_tune_at = 100

  ## freeze the bafore layers of fine tune at number in base model
  for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

  for fc in fc_layers:
    x = Dense(units=fc, activation='relu')(x)
    x = Dropout(dropout)(x)

  ## output layer with softmax activation function
  predictions = Dense(num_classes, activation='softmax')(x)

  final_model = Model(inputs = base_model.input, outputs = predictions)
  return final_model


class_list = ['bed', 'chair', 'sofa', 'swivelchair', 'table']
## units for fully connected layers 
FC_LAYERS = [1024, 1024]
dropout = 0.3

model = build_model(base_model, dropout, FC_LAYERS, len(class_list))

## compilation
adam = Adam(learning_rate=0.00001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

## train model
history = model.fit(train_generator, 
                    epochs=10,
                    steps_per_epoch= num_train_images // batchsize, 
                    validation_data=val_generator,
                    validation_steps = num_val_images // batchsize)

model.save('model.h5')

