"""
ref: https://github.com/tensorflow/tensorflow/issues/14356

if shit don't work because 'Tensor is not an element of this graph', do this:

    def load_model():
        global model
        model = ResNet50(weights="imagenet")
                # this is key : save the graph after loading the model
        global graph
        graph = tf.get_default_graph()

While predicting, use the same graph

    with graph.as_default():
    preds = model.predict(image)
    #... etc

"""

# flask stuff
from flask import Flask, request, jsonify
from flask_cors import CORS
# swagger stuff
from flasgger import Swagger
# ml stuff
from src.functions.app import segment_image
# other stuff
import ujson as json
from src.python_utils.input_preprocessing import preprocess_json_input

# CORS for connecting with the front
allowed_domains = [
    r'*'
]

application = Flask(__name__)
swagger = Swagger(application)

CORS(application,
     origins=allowed_domains,
     resources=r'/v1/*',
     supports_credentials=True)


@application.route('/', methods=['GET'])
def main():
    return 'welcome to pspnet!'


@application.route('/v1/segment/', methods=['POST'])
def segment():
    """
    segments images using PSPNet

    Takes a json input in this form:
    {
        "image_url": "https://i.imgur.com/zxTNS3e.jpg",
        "filter_class": "sky"
    }

    or in this form:
    {
        "image_base64": "base64_img",
        "image_name": "image_name",
        "filter_class": "sky"
    }

    Where "image_base64" is an encoded image in base64 with UTF-8 encoding.

    Eg to encode with python:
        with file as image_file:
            base64_bytes = base64.b64encode(image_file.read())
            base64_string = base64_bytes.decode('utf-8')

        json = {"image_base64": base64_string, "image_name": "my_image"}

    Args:
        input (dict): The parsed json

    Returns:
        dict: A dict in the form :
            {"sr_image_url": url, "sr_image_uri": uri,
            "original_image_url": url, "original_image_uri": uri,
            "upscale_factor": upscale_factor}
    ---
    consumes:
        application/json
    produces:
        application/json
    parameters:
        - name: body
          in: body
          description: base image input object
          required: true
          content:
            application/json:
                schema:
                    type: object
                    required:
                        - image_url
                        - filter_class
                    properties:
                        image_url:
                            type: string
                        filter_class:
                            type: string
    responses:
        200:
            description: json with base64 images
            schema:
                id: base64File
                properties:
                    results:
                      type: json
                      default: {'image': "0101010"}
                    status:
                      type: number
                      default: 200
        """

    if request.method == 'POST':
        input_json = request.get_json('input')

    preprocessed_input = preprocess_json_input(input_json)

    image_base64 = preprocessed_input['image_b64']
    image_name = preprocessed_input['name']
    filter_class = preprocessed_input['filter_class']

    # up to here, the code can return an image object or a base64 object
    input_b64 = image_base64
    response = segment_image(input_b64, filter_class_in=filter_class)[0]
    # todo: do image filtering by class

    output_dict = {
        "name": image_name,
        "segmented": response[0],
        "alpha_blended": response[1],
        "filtered_image": response[2]
    }

    output_json = json.dumps(output_dict, indent=4, ensure_ascii=False)

    return output_json


if __name__ == '__main__':
    # this needs to be places in __main__ because when apache is
    # reading this file, if this is run, apache will never be able to
    # grab the application file for use in the production server.
    application.run(host='0.0.0.0', port=5000, debug=False)
    print('a flask app is initiated at {0}'.format(application.instance_path))
