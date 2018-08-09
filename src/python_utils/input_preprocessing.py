import requests
from PIL import Image
from pathlib import Path
from io import BytesIO
import base64

from src.python_utils.utils import image_obj_to_base64


class AlgorithmError(Exception):
    """Define error handling class."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value).replace("\\n", "\n")


def preprocess_json_input(input):
    if "image_url" in input and ("image_base64" in input or "image_name" in input):
        raise AlgorithmError("You provided both an image_url and image_base64. Please choose only one.")
    elif "image_url" in input:
        valid_json = True
        image_url = input["image_url"]
        class_to_filter = input["filter_class"]
        valid_json = True
        image_response = requests.get(image_url)
        # Retrieve input information
        image = Image.open(BytesIO(image_response.content))
        image_base64 = image_obj_to_base64(image)
        image_name = Path(image_url).name
    elif "image_base64" in input and "image_name" in input:
        valid_json = True
        image_name = input["image_name"]
        image_base64 = input["image_base64"]
        class_to_filter = input["filter_class"]
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
    else:
        image_response = requests.get("https://i.imgur.com/zxTNS3e.jpg")
        # Retrieve input information
        image = Image.open(BytesIO(image_response.content))
        image_base64 = image_obj_to_base64(image)
        image_name = Path("https://i.imgur.com/zxTNS3e.jpg").name
        class_to_filter = 'sky'

    output_dict = {
        "name": image_name,
        "image_b64": image_base64,
        "filter_class": class_to_filter
    }

    return output_dict