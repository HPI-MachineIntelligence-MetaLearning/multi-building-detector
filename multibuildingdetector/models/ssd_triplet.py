import warnings

import chainer
import numpy as np
from chainercv.links import SSD300
from chainercv.links.model.ssd import VGG16Extractor300
from chainercv.utils import download_model

from multibuildingdetector.multiboxes.tripletmultibox import TripletMultibox

_imagenet_mean = np.array((123, 117, 104)).reshape((-1, 1, 1))
try:
    import cv2  # NOQA
    _available = True
except ImportError:
    _available = False


class SSDTriplet(SSD300):
    def __init__(self, n_fg_class=None, pretrained_model=None):
        n_fg_class, path = _check_pretrained_model(
            n_fg_class, pretrained_model, self._models)

        super(SSD300, self).__init__(
            extractor=VGG16Extractor300(),
            multibox=TripletMultibox(
                n_class=n_fg_class + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=(8, 16, 32, 64, 100, 300),
            sizes=(30, 60, 111, 162, 213, 264, 315),
            mean=_imagenet_mean)

        if path:
            _load_npz(path, self)

    def predict(self, imgs):
        """Detect objects from images.
        This method predicts objects for each image.
        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.
        Returns:
           tuple of lists:
           This method returns a tuple of two lists,
           :obj:`(bboxes, labels, scores)`.
           * **mb_locs**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               (Default bounding box space, so somthing like 8700) \
               Each bounding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **mb_confs** : A list of integer arrays of shape
                            :math:`(R,<feature vector size>)`. \
               Each value indicates the feature vector of the bounding box. \
        """

        x = list()
        sizes = list()
        for img in imgs:
            _, H, W = img.shape
            img = self._prepare(img)
            x.append(self.xp.array(img))
            sizes.append((H, W))

        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            x = chainer.Variable(self.xp.stack(x))
            mb_locs, mb_confs = self(x)
        mb_locs, mb_confs = mb_locs.array, mb_confs.array
        return mb_locs, mb_confs


def _check_pretrained_model(n_fg_class, pretrained_model, models):
    if pretrained_model in models:
        model = models[pretrained_model]
        if n_fg_class:
            if model['n_fg_class'] and not n_fg_class == model['n_fg_class']:
                raise ValueError(
                    'n_fg_class should be {:d}'.format(model['n_fg_class']))
        else:
            if not model['n_fg_class']:
                raise ValueError('n_fg_class must be specified')
            n_fg_class = model['n_fg_class']

        path = download_model(model['url'])

        if not _available:
            warnings.warn(
                'cv2 is not installed on your environment. '
                'Pretrained models are trained with cv2. '
                'The performace may change with Pillow backend.',
                RuntimeWarning)
    elif pretrained_model:
        path = pretrained_model
    else:
        path = None

    return n_fg_class, path


def _load_npz(filename, obj):
    with np.load(filename) as f:
        d = chainer.serializers.NpzDeserializer(f, strict=False)
        d.load(obj)
