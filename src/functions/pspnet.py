#!/usr/bin/env python
from __future__ import print_function
import os
from os.path import splitext, join, isfile, isdir, basename
import argparse
import numpy as np
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json, load_model
import tensorflow as tf
import src.functions.layers_builder as layers
from glob import glob
from src.python_utils import utils
from src.python_utils.preprocessing import preprocess_img
from keras.utils.generic_utils import CustomObjectScope
from src.python_utils.utils import base64_to_image, image_to_base64, image_np_to_base64
from src.python_utils.postprocessing import mask_transparency
from src.python_utils.utils import get_filepath

# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape, weights):
        self.input_shape = input_shape
        wd = os.getcwd()

        json_path = join(wd, "src", "weights", "keras", weights + ".json")
        h5_path = join(wd, "src", "weights", "keras", weights + ".h5")

        json_path_local = join("weights", "keras", weights + ".json")
        h5_path_local = join("weights", "keras", weights + ".h5")

        if not os.path.isfile(json_path):
                # local pspnet.py
                json_path = get_filepath(json_path_local)
                h5_path = get_filepath(h5_path_local)

        if 'pspnet' in weights:
            json_path_chk = json_path
            h5_path_chk = h5_path
            cwd = wd
            print('debug:')
            print(json_path_chk)
            print(h5_path_chk)
            print(cwd)
            print(os.listdir(cwd))
            print('end of debug.')
            if os.path.isfile(json_path) and os.path.isfile(h5_path):
                print("Keras model & weights found, loading...")
                with CustomObjectScope({'Interp': layers.Interp}):
                    with open(json_path, 'r') as file_handle:
                        self.model = model_from_json(file_handle.read())
                self.model.load_weights(h5_path)
            else:
                try:
                    print("No Keras model & weights found, import from npy weights.")
                    self.model = layers.build_pspnet(nb_classes=nb_classes,
                                                     resnet_layers=resnet_layers,
                                                     input_shape=self.input_shape)
                    self.set_npy_weights(weights)
                except:
                    print("couldn't find anything. the current work directory is:")
                    print(os.getcwd())

        else:
            print('Load pre-trained weights')
            self.model = load_model(weights)

    def predict(self, img, flip_evaluation=False):
        """
        Predict segementation for an image.

        Arguments:
            img: must be rowsxcolsx3
        """
        h_ori, w_ori = img.shape[:2]

        # Preprocess
        img = misc.imresize(img, self.input_shape)

        img = img - DATA_MEAN
        img = img[:, :, ::-1]  # RGB => BGR
        img = img.astype('float32')
        print("Predicting...")

        probs = self.feed_forward(img, flip_evaluation)

        if img.shape[0:1] != self.input_shape:  # upscale prediction if necessary
            h, w = probs.shape[:2]
            probs = ndimage.zoom(probs, (1. * h_ori / h, 1. * w_ori / w, 1.),
                                 order=1, prefilter=False)

        print("Finished prediction...")

        return probs

    def feed_forward(self, data, flip_evaluation=False):
        assert data.shape == (self.input_shape[0], self.input_shape[1], 3)

        if flip_evaluation:
            print("Predict flipped")
            input_with_flipped = np.array(
                [data, np.flip(data, axis=1)])
            prediction_with_flipped = self.model.predict(input_with_flipped)
            prediction = (prediction_with_flipped[
                          0] + np.fliplr(prediction_with_flipped[1])) / 2.0
        else:
            prediction = self.model.predict(np.expand_dims(data, 0))[0]
        return prediction

    def set_npy_weights(self, weights_path):
        npy_weights_path = join("weights", "npy", weights_path + ".npy")
        json_path = join("weights", "keras", weights_path + ".json")
        h5_path = join("weights", "keras", weights_path + ".h5")

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path, encoding='bytes').item()
        for layer in self.model.layers:
            print(layer.name)
            if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
                mean = weights[layer.name.encode()][
                    'mean'.encode()].reshape(-1)
                variance = weights[layer.name.encode()][
                    'variance'.encode()].reshape(-1)
                scale = weights[layer.name.encode()][
                    'scale'.encode()].reshape(-1)
                offset = weights[layer.name.encode()][
                    'offset'.encode()].reshape(-1)

                self.model.get_layer(layer.name).set_weights(
                    [scale, offset, mean, variance])

            elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
                try:
                    weight = weights[layer.name.encode()]['weights'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight])
                except Exception as err:
                    biases = weights[layer.name.encode()]['biases'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight,
                                                                  biases])
        print('Finished importing weights.')

        print("Writing keras model & weights")
        json_string = self.model.to_json()
        with open(json_path, 'w') as file_handle:
            file_handle.write(json_string)
        self.model.save_weights(h5_path)
        print("Finished writing Keras model & weights")


class PSPNet50(PSPNet):
    """Build a PSPNet based on a 50-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        print('debug: building pspnet50 with')
        print('nb_classes: {}'.format(nb_classes))
        print('input shape: {}'.format(input_shape))
        print('weights: {}'.format(weights))
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=50,
                        input_shape=input_shape, weights=weights)


class PSPNet101(PSPNet):
    """Build a PSPNet based on a 101-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=101,
                        input_shape=input_shape, weights=weights)


if __name__ == "__main__":
    """
    CODE HERE IS MEANT FOR LOCAL DEVELOPMENT
    """
    debug = False

    dummy_file = '../dummy/14.jpg'
    dummy_output = '../dummy/14_result.jpg'

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

    args = parser.parse_args()

    # Handle input and output args
    # two functions: from dummy folder or from base64 strings
    images = glob(args.glob_path) if args.glob_path else [args.input_path,]
    if args.glob_path:
        fn, ext = splitext(args.output_path)
        if ext:
            parser.error("output_path should be a folder for multiple file input")
        if not isdir(args.output_path):
            os.mkdir(args.output_path)

    # Predict
    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        print(args)
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

        for i, img_path in enumerate(images):
            print("Processing image {} / {}".format(i+1,len(images)))
            print('current wd')
            img = misc.imread(img_path, mode='RGB')
            cimg = misc.imresize(img, (args.input_size, args.input_size))

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

            misc.imsave(filename + "_seg_read" + ext, cm)
            misc.imsave(filename + "_seg" + ext, color_cm)
            misc.imsave(filename + "_probs" + ext, pm)
            misc.imsave(filename + "_seg_blended" + ext, alpha_blended)
            misc.imsave(filename + "_masked" + ext, masked_output)

            # flip original and segmented
            from src.python_utils.postprocessing import flip_image
            ori_fp = img_path
            masked_fp = filename + "_masked" + ext

            flip_image(ori_fp)
            flip_image(masked_fp)

            if debug:
                from PIL import Image

                print('color cm:')
                print(color_cm.shape)
                print(color_cm)

                print('cm:')
                print(cm.shape)
                print(cm)

                print('alpha blended:')
                print(alpha_blended.shape)
                print(alpha_blended)

                arr2im = Image.fromarray(np.uint8(np.multiply(color_cm, 255)))
                arr2im.show()

                # color segmentation
                t = image_np_to_base64(color_cm)

                # check if the base64 actually works properly
                m = base64_to_image(t)
                m_out = misc.imread(m)
                misc.imsave(filename + "_test.png", m_out)

                # alpha blended
                t = image_np_to_base64(alpha_blended)

                # check if the base64 actually works properly
                m = base64_to_image(t)
                m_out = misc.imread(m)
                misc.imsave(filename + "_test_blended.png", m_out)




