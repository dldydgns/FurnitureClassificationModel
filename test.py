import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input

class_list = ['bed', 'chair', 'sofa', 'swivelchair', 'table']

## load model
load_model = tf.keras.models.load_model('model.h5')

## save image by cam
cap = cv2.VideoCapture(0)
cap.set(3,150)
cap.set(4,150)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
           cv2.imwrite('photo.jpg', frame)
           break
    else:
        break

## load image
img = tf.keras.utils.load_img('photo.jpg', target_size = (150,150))
img_tensor = tf.keras.utils.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor = preprocess_input(img_tensor)

#predict
predict = load_model.predict(img_tensor)
index = np.argmax(predict)
print(class_list[index])
plt.imshow(img)
plt.show()

