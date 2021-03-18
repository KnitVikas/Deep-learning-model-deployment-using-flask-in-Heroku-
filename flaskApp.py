import keras
from keras.layers import Input, GlobalAveragePooling2D, Lambda
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from flask import Flask, render_template, request, jsonify
import os
import io
import numpy as np
import cv2
import base64
import pandas as pd
app = Flask(__name__)


@app.route('/upload', methods=['GET', 'POST'])
def get_image():
    #data = request.json["image"]
    if request.method == "GET":
        return jsonify({"error": "Hi,"})
    else:
        b64_string = request.json["image"]
        with open("imageToSave.jpg", "wb") as fh:
            fh.write(base64.decodebytes(b64_string.encode('utf-8')))
        img_test = cv2.imread(os.getcwd()+"/" + "imageToSave.jpg")
        resized_image = cv2.resize(img_test, (331, 331))
        pred_breed, max_prob = get_prediction(
            trained_model, cnn_model, resized_image, labels)
        return jsonify({"breed": pred_breed, "prob": round(max_prob[0] * 100, 2)})


def get_features(MODEL, cnn_model, X):

    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)
    features = cnn_model.predict(X, batch_size=64, verbose=1)
    return features


def get_prediction(Model, cnn_model, image, labels):

    classes = list(sorted(set(labels["breed"])))
    width = 331
    test = np.zeros((1, width, width, 3), dtype=np.uint8)
    test[0] = image
    test_feature = get_features(Model, cnn_model, test)
    y_pred = Model.predict(test_feature, batch_size=128)

    max_prob = sorted(y_pred[0], reverse=True)[:5]
    pred_breed = classes[list(y_pred[0]).index(max_prob[0])]
    return pred_breed, max_prob


if __name__ == '__main__':
    print("----loading model----")
    width = 331
    cnn_model = ResNet50(include_top=False, input_shape=(
        width, width, 3), weights='imagenet')
    model_name = "model1.h5"
    trained_model = keras.models.load_model(os.getcwd() + "/" + model_name)
    print("model loaded---")
    labels = pd.read_csv("labels.csv")

    app.run(port=3030, debug=True)
