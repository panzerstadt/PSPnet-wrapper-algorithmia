"""
ref: https://tuatini.me/part-2-creating-an-api-endpoint-for-your-model/
"""

import Algorithmia

from src.functions.app import segment_image
import ujson as json
from src.python_utils.input_preprocessing import preprocess_json_input


# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages
def apply(input):
    """
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
        Ex to encode with python:
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
    """
    preprocessed_input = preprocess_json_input(input)

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
    """
    CODE ABOVE IS MEANT FOR ONLINE DEVELOPMENT
    """
    # query = input('input image or press ENTER to run default: ')
    # if query:
    #     r = apply(query)
    # else:
    #     from src.functions.get_b64 import get_dummy_base64
    #     b64_input = get_dummy_base64()
    #
    #     r = apply(b64_input)
    #
    # print(r[:1000])
