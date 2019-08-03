# -*- coding: utf-8 -*

# -------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: utils.py
# Time: 7/29/19 4:55 PM
# Description:
# -------------------------------------------------------------------------------

from __future__ import division
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def rotate(img, angle, resample=False, expand=False, center=None):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.rotate(angle, resample, expand, center)
