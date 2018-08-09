import tensorflow as tf
import os
import numpy as np
import argparse
from scipy import misc
from os.path import splitext, join, basename, isdir
import glob

from keras import backend as K

from src.functions.pspnet import PSPNet50, PSPNet101
from src.python_utils import utils
from src.python_utils.postprocessing import mask_transparency, flip_image


dummy_file = 'dummy/01.jpg'
dummy_output = 'dummy/01_result.jpg'

# settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parser_with_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet50_ade20k',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('-i', '--input_path', type=str, default=dummy_file,
                        help='Path the input image')
    parser.add_argument('-g', '--glob_path', type=str, default=None,
                        help='Glob path for multiple images')
    parser.add_argument('-o', '--output_path', type=str, default=dummy_output,
                        help='Path to output')
    parser.add_argument('--id', default="0")
    parser.add_argument('--input_size', type=int, default=500)
    parser.add_argument('-f', '--flip', type=bool, default=True,
                        help="Whether the network should predict on both image and flipped image.")

    return parser

# defaults
parser = parser_with_arguments()
args = parser.parse_args()

if not args.weights:
    if "pspnet50" in args.model:
        pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                          weights=args.model)
    elif "pspnet101" in args.model:
        if "cityscapes" in args.model:
            pspnet = PSPNet101(nb_classes=19, input_shape=(713, 713),
                               weights=args.model)
        if "voc2012" in args.model:
            pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                               weights=args.model)

    else:
        print("Network architecture not implemented.")
else:
    pspnet = PSPNet50(nb_classes=2, input_shape=(
        768, 480), weights=args.weights)

graph = tf.get_default_graph()


def segment_image(input_b64=None, filter_class_in=None):

    def predict(img, save=True):
        # predict
        with graph.as_default():
            probs = pspnet.predict(img, args.flip)

            cm = np.argmax(probs, axis=2)
            pm = np.max(probs, axis=2)

            color_cm = utils.add_color(cm)
            # color cm is [0.0-1.0] img is [0-255]
            alpha_blended = 0.5 * color_cm * 255 + 0.5 * img

            # filtered png with transparency
            full_color_cm = np.multiply(color_cm, 255)
            full_color_cm = full_color_cm[:, :, ::-1].copy()  # PIL (RGB) -> OPENCV (BGR)
            masked_output = mask_transparency(img, full_color_cm, save=False, output_type='array')


            if args.glob_path:
                input_filename, ext = splitext(basename(img_path))
                filename = join(args.output_path, input_filename)
            else:
                filename, ext = splitext(args.output_path)

            ext = '.png'

            if save:
                misc.imsave(filename + "_seg_read" + ext, cm)
                misc.imsave(filename + "_seg" + ext, color_cm)
                misc.imsave(filename + "_probs" + ext, pm)
                misc.imsave(filename + "_seg_blended" + ext, alpha_blended)
                misc.imsave(filename + "_masked" + ext, masked_output)

                # flip original and segmented
                ori_fp = img_path
                masked_fp = filename + "_masked" + ext

                flip_image(ori_fp)
                flip_image(masked_fp)

            # return color_cm and alpha_blended as base64 images
            output_color_cm = utils.image_np_to_base64(color_cm)
            output_alpha_blended = utils.image_np_to_base64(alpha_blended)
            output_masked = utils.image_np_to_base64(masked_output)

            return output_color_cm, output_alpha_blended, output_masked

    if input_b64:
        img = utils.base64_to_image(input_b64)
        img = misc.imread(img, mode='RGB')

        result = predict(img, save=False)

        return [result]
    else:
        images = glob(args.glob_path) if args.glob_path else [args.input_path, ]
        if args.glob_path:
            fn, ext = splitext(args.output_path)
            if ext:
                parser.error("output_path should be a folder for multiple file input")
            if not isdir(args.output_path):
                os.mkdir(args.output_path)

        output = []
        for i, img_path in enumerate(images):
            print("Processing image {} / {}".format(i + 1, len(images)))
            img = misc.imread(img_path, mode='RGB')
            cimg = misc.imresize(img, (args.input_size, args.input_size))

            results = predict(img)
            output.append(results)

        return output


if __name__ == '__main__':
    from src.functions.get_b64 import get_dummy_base64
    from src.python_utils.utils import show_base64_image
    import os

    t = input('input base64 string, or ENTER for dummy test: ')

    if not t:
        print('using dummy for prediction')
        t = get_dummy_base64()
    r = segment_image(t)

    show_base64_image(r[0][0])
    show_base64_image(r[0][1])

    print(r[0][0][:100])