from flask import Flask, request, jsonify
from fastai.vision import (
    ImageDataBunch,
    ConvLearner,
    open_image,
    get_transforms,
    models,
)
from flask_cors import CORS,cross_origin
app = Flask(__name__)
CORS(app, support_credentials=True)

# load the learner
learn = load_learner(path='./models', file='trained_model.pkl')
classes = learn.data.classes


def predict_single(img_file):
    'function to take image and return prediction'
    prediction = learn.predict(open_image(img_file))
    probs_list = prediction[2].numpy()
    return {
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    }

def predict():
    return jsonify(predict_single(request.files['image']))

# route for prediction
@app.route('/predict', methods=['POST'])



if __name__ == '__main__':
    app.run()
