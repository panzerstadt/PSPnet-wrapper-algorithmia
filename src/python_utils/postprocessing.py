import cv2
import numpy as np
from PIL import Image
import os


def mask_transparency(original_image, colour_map, sky_colour_rgb=(0, 255, 74), sensitivity=2, save=True, output_type='array'):
    try:
        # Read original panaroma image
        foreground = cv2.imread(original_image)
        print('reading colour map from image')
    except:
        foreground = np.array(original_image)
        print('reading colour map from array')

    try:
        # Read segnet output image
        colour_map = cv2.imread(colour_map)
        print('reading colour map from image')
    except:
        colour_map = np.uint8(np.array(colour_map))
        print('reading colour map from array')

    print(type(colour_map))
    print(colour_map)

    r, g, b = sky_colour_rgb

    lr = max(r - sensitivity, 0)
    lg = max(g - sensitivity, 0)  # green never changes because we are looking for green
    lb = max(b - sensitivity, 0)

    ur = min(r + sensitivity, 255)
    ug = min(g + sensitivity, 255)
    ub = min(b + sensitivity, 255)

    # Select color range to remove for sky (order in BGR)
    def _skycolor_range(b, g, r):
        lower = np.array([b, g, r])
        return lower

    # Create mask
    mask = cv2.inRange(colour_map, _skycolor_range(lb, lg, lr), _skycolor_range(ub, ug, ur))

    # Reverse the mask (change sky in black and rest in white)
    mask = 255 - mask

    # Combine mask and panaroma image
    b, g, r = cv2.split(foreground)
    rgba = [b, g, r, mask]

    output_np = cv2.merge(rgba, 4)
    output_img = Image.fromarray(output_np)

    if save:
        # Save png image with no sky
        cv2.imwrite('output.png', output_np)

    if output_type == 'array':
        return output_np
    else:
        return output_img


def flip_image(input_fp, output_fp=None, type='horizontal'):
    img = cv2.imread(input_fp, -1)

    if type == 'horizontal':
        out_img = cv2.flip(img, 1)
    if type == 'vertical':
        out_img = cv2.flip(img, 0)
    if type == 'both':
        out_img = cv2.flip(img, -1)

    if not output_fp:
        pieces = os.path.splitext(input_fp)
        name = os.path.splitext(os.path.basename(input_fp))[0]
        output_fp = os.path.join(os.path.dirname(input_fp), (name + '_flipped' + pieces[-1]))

    cv2.imwrite(output_fp, out_img)



if __name__ == '__main__':
    def check_mask_transparency():
        original = 'ori.jpg'
        colourmap = 'seg.png'

        o = cv2.imread(original)
        c = cv2.imread(colourmap)

        t = mask_transparency(original_image=o, colour_map=c, output_type='image')
        print(t.show())

    def check_flip_image():
        input_paths = ['../dummy/01.jpg', '../dummy/01_result_masked.png']

        for p in input_paths:
            flip_image(input_fp=p)

    check_flip_image()