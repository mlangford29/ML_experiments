# import necessary libraries
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# define flask app
app = Flask(__name__)

# define model path
MODEL_PATH = 'food_classification_model.h5'

# define food categories
food_categories = ['pizza', 'hamburger', 'sushi', 'ramen']

# load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# define function to preprocess image
def preprocess_image(image):
    # resize image
    image = image.resize((224, 224))
    # convert to numpy array
    image_array = np.array(image)
    # normalize image
    image_array = image_array / 255.0
    # expand dimensions to match model input
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# define function to predict food category from image
def predict_food_category(image):
    # preprocess image
    image_array = preprocess_image(image)
    # make prediction
    prediction = model.predict(image_array)
    # get predicted class index
    class_index = np.argmax(prediction)
    # get predicted class label
    class_label = food_categories[class_index]
    return class_label

# define route to handle image upload and prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # check if image was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'})
    # read uploaded image
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    # predict food category
    prediction = predict_food_category(image)
    # return prediction result
    return jsonify({'prediction': prediction})

# run flask app
if __name__ == '__main__':
    app.run(debug=True)
