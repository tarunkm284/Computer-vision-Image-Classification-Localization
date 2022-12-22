import io
import pickle
from base64 import b64encode

import cv2
import numpy as np  # linear algebra
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw
from flask import Flask, render_template, request
from tensorflow import keras

model = keras.models.load_model('models/ResNet_CIFAR10_Model_Best.h5')
rmodel = tf.keras.models.load_model("models/image_localization_model.h5", compile=False)

app = Flask(__name__)


with open('label_binary.pkl', 'rb') as file:
    encoder = pickle.load(file)

labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split()


def predictDrawbox(model, image, le):
    img = tf.cast(np.expand_dims(image, axis=0), tf.float32)
    # prediction
    predict = model.predict(img)
    pred_box = predict[..., 0:4] * 228
    x = pred_box[0][0]
    y = pred_box[0][1]
    w = pred_box[0][2]
    h = pred_box[0][3]
    # get class name
    trans = le.inverse_transform(predict[..., 4:])
    file_object = io.BytesIO()
    img= Image.fromarray(image.astype('uint8'))
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, w, h], outline='red')
    img.save(file_object, 'PNG')
    base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('ascii')
    return base64img,trans[0]


@app.route("/")
def home():
    return render_template("home.html")

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        f = request.files['file']
        path = "TestDataset/TestImageClassification/"+f.filename
        imgdata = cv2.resize(cv2.imread(path), (32, 32))
        n = np.array(imgdata)/255
        print(n.shape)
        p = n.reshape(1, 32, 32, 3)
        try:
            predicted_label = labels[model.predict(p).argmax()]
        except:
            return "Not Supported try another image"
        finally:
            print("predicted label is {}".format(predicted_label))
            return "predicted label is {}".format(predicted_label)

@app.route('/imageLocalUploader', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        path = "TestDataset/TestImageLocalization/"+f.filename
        imgdata = cv2.resize(cv2.imread(path), (228, 228))
        rimg = np.array(imgdata)
        finalimg = predictDrawbox(rmodel,rimg,encoder)
        return render_template("plot.html",displayimage=finalimg[0],label=finalimg[1])


if __name__ == "__main__":
    app.run(debug=True)



