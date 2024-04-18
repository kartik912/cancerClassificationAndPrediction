import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.applications import ResNet50
from keras.models import Model
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import matplotlib.pyplot as plt

class predict_class :

    def pred(self,image_path):
        pred = ""
        image_data = image_path.read()
        image_path.seek(0)
        image = Image.open(BytesIO(image_data))
        image = image.convert('RGB')
        

        image_array = np.array(image)
        image_copy1 = image_array.copy()
        image_copy = image_array.copy()
        image_copy = tf.cast(image_copy, tf.float32)
        image_copy = tf.image.resize(image_copy, [224,224])
        image_copy = np.array(image_copy) / 255.0
        image_copy = np.expand_dims(image_copy, axis=0)


# classify
        
        model_path='imageclassification/models21/UC.h5'
        model=load_model(model_path)
        prediction = model.predict(image_copy)
        predicted_classes = np.argmax(prediction)
        

#INPUT CLASS predictor
        if(predicted_classes==1):
            model_path='imageclassification/models21/organClassifier.h5'
            model=load_model(model_path)
            predictions = model.predict(image_copy)
            predicted_class = np.argmax(predictions)
        
            if predicted_class == 0:
                model_path='imageclassification/models21/best_model_2.h5'
                incept_model=load_model(model_path)
                IMAGE_SHAPE=(224,224)
                print(image_copy1.shape)
                resized_image = cv2.resize(image_copy1, (224, 224))
                img_array = resized_image.reshape((224,224,3))
                classes=['benign','malignant','normal']

                img1 = tf.keras.applications.efficientnet.preprocess_input(img_array)
                res = incept_model.predict(np.expand_dims(img1, axis = 0))
                pred = classes[np.argmax(res)]

                
            else:
                def prepare_image(file):
                    grayscale_image = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
                    resized_image = cv2.resize(grayscale_image, (150, 150))
                    img_array = resized_image.reshape((150,150,1))
                    print(img_array.shape)
                    return img_array
                classes=['normal','benign','malignant']
                model_path='imageclassification\models21\my_gall.h5'
                model=load_model(model_path)
                img2 = prepare_image(image_copy1)
                res = model.predict(np.expand_dims(img2, axis = 0))
                pred = classes[np.argmax(res)]
        return pred