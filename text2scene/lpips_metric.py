import lpips
import torch
import numpy as np


loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def normalize_image(rgb_img):
    """
    Here it says that the normalization should be w.r.t all images
    https://gist.github.com/liuhh02/2bd3f5b1ced9142d728d207f7828043e?permalink_comment_id=4107352#gistcomment-4107352
    """
    img = np.array(rgb_img)
    img_min = img.min()
    img_max = img.max()
    img_range = img_max - img_min
    scaled = np.array((img-img_min) / float(img_range), dtype='f')
    return -1 + (scaled * 2)


def calculate_lpips_between_two_images(normalized_img1, normalized_img2):
    """
    IMPORTANT: input must be normalized to [-1,1]
    """
    return loss_fn_vgg(normalized_img1, normalized_img2)
