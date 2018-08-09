from __future__ import print_function
import colorsys
import numpy as np
from scipy import misc
from keras.models import Model
from src.labels.cityscapes_labels import trainId2label
from src.labels.ade20k_labels import ade20k_id2label
from src.labels.pascal_voc_labels import voc_id2label
import base64
from PIL import Image
import io, os


def class_image_to_image(class_id_image, class_id_to_rgb_map):
    """Map the class image to a rgb-color image."""
    colored_image = np.zeros(
        (class_id_image.shape[0], class_id_image.shape[1], 3), np.uint8)
    for row in range(class_id_image.shape[0]):
        for col in range(class_id_image.shape[1]):
            try:
                colored_image[row, col, :] = class_id_to_rgb_map[
                    int(class_id_image[row, col])].color
            except KeyError as key_error:
                print("Warning: could not resolve classid %s" % key_error)
    return colored_image


def color_class_image(class_image, model_name):
    """Color classed depending on the model used."""
    if 'cityscapes' in model_name:
        colored_image = class_image_to_image(class_image, trainId2label)
    elif 'voc' in model_name:
        colored_image = class_image_to_image(class_image, voc_id2label)
    elif 'ade20k' in model_name:
        colored_image = class_image_to_image(class_image, ade20k_id2label)
    else:
        colored_image = add_color(class_image)
    return colored_image


def add_color(img, num_classes=32):
    h, w = img.shape
    img_color = np.zeros((h, w, 3))
    for i in range(1, 151):
        img_color[img == i] = to_color(i)
    img_color[img == num_classes] = (1.0, 1.0, 1.0)
    return img_color


def to_color(category):
    """Map each category color a good distance away from each other on the HSV color space."""
    v = (category - 1) * (137.5 / 360)
    return colorsys.hsv_to_rgb(v, 1, 1)


def debug(model, data):
    """Debug model by printing the activations in each layer."""
    names = [layer.name for layer in model.layers]
    for name in names[:]:
        print_activation(model, name, data)


def print_activation(model, layer_name, data):
    """Print the activations in each layer."""
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    io = intermediate_layer_model.predict(data)
    print(layer_name, array_to_str(io))


def array_to_str(a):
    return "{} {} {} {} {}".format(a.dtype, a.shape, np.min(a),
                                   np.max(a), np.mean(a))


# def base64_to_image_np(input_b64):
#     b64_content = base64.decodebytes(input_b64)
#     q = np.frombuffer(b64_content, dtype=np.float64)
#     print(q)
#     print(q.shape)
#
#     im_content = misc.imread(b64_content)
#     return im_content
#
#
def image_np_to_base64(input_image_np):
    """
    OUTPUT conversion (API)
    for algorithm output into API

    if not using numpy's save, the array is normalized to 0, 1. so you have to remap it

    ref: https://github.com/python-pillow/Pillow/issues/985
    ref2: https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image
    formats: http://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html#jpeg
    :param input_image:
    :return: base64 encoded PNG image
    """
    if input_image_np.shape[-1] < 3:
        print('this converter has only been tried on RGB images. result may be wrong.')

    # check if max is 1 or 255
    if np.argmax(input_image_np) <= 1.5:
        im_content = Image.fromarray(np.uint8(np.multiply(input_image_np, 255)))
    elif np.argmax(input_image_np) <= 256:
        im_content = Image.fromarray(np.uint8(input_image_np))
    else:
        im_content = Image.fromarray(np.uint8(input_image_np))
        im_content.show()
        print(input_image_np)
        print('possible uncaught error: array max range is larger than 255: {}'.format(np.argmax(input_image_np)))

    b = io.BytesIO()
    im_content.save(b, 'PNG')
    image_bytes = b.getvalue()

    # this image object needs to turn into bytes
    b64_content = base64.b64encode(image_bytes)
    return b64_content


def image_obj_to_base64(input_pil_Image):
    im_content = input_pil_Image

    b = io.BytesIO()
    im_content.save(b, 'PNG')
    image_bytes = b.getvalue()

    # this image object needs to turn into bytes
    b64_content = base64.b64encode(image_bytes)
    return b64_content


def base64_to_image(input_b64):
    """
    INPUT conversion (API)
    converts b64 to image OBJECT (to be passed into scipy.misc.imread() to convert into an np array
    :param input_b64:
    :return: similar to open(fp).read()
    """
    decoded_img = base64.b64decode(input_b64)
    output_img_obj = io.BytesIO(decoded_img)
    return output_img_obj


def image_to_base64(input_filepath):
    """
    before INPUT (client)
    used for conversion BEFORE sending to API
    :param input_filepath:
    :return:
    """
    with open(input_filepath, 'rb') as im_content:
        original_string = im_content.read()
    b64_content = base64.b64encode(original_string)
    return b64_content


def show_base64_image(base64_str):
    im_bytes = base64_to_image(base64_str)
    im_content = Image.open(im_bytes)
    im_content.show()


def get_filepath(filepath_with_extension, debug=False):
    test = filepath_with_extension[0]
    if not test == '/':
        filepath_with_extension = '/' + filepath_with_extension
    if test == '.':
        filepath_with_extension = filepath_with_extension[2:]
    if debug: print('starting filepath', filepath_with_extension)

    try:
        with open(filepath_with_extension): pass
        return filepath_with_extension
    except FileNotFoundError:
        try:
            fp0 = '.' + filepath_with_extension
            if debug: print('fp0', fp0)
            with open(fp0): pass
            return fp0
        except FileNotFoundError:
            try:
                fp1 = '..' + filepath_with_extension
                if debug: print('fp1', fp1)
                with open(fp1): pass
                return fp1
            except FileNotFoundError:
                try:
                    fp2 = '../..' + filepath_with_extension
                    if debug: print('fp2', fp2)
                    with open(fp2): pass
                    return fp2
                except FileNotFoundError:
                    try:
                        fp3 = '../../..' + filepath_with_extension
                        if debug: print('fp3', fp3)
                        with open(fp3): pass
                        return fp3
                    except:
                        print('current workingdir: ', os.getcwd())
                        raise SystemError("file not found by traversing backwards 3 folders")



if __name__ == '__main__':
    b64_str = "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"



    t = show_base64_image(b64_str)