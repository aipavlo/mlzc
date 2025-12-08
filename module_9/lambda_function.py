import onnxruntime as ort
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

session = ort.InferenceSession('hair_classifier_empty.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess(img):
    x = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0).astype(np.float32)
    return x


def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))
    x = preprocess(img)
    output = session.run([output_name], {input_name: x})[0]
    return float(output[0][0])


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return {'prediction': result}