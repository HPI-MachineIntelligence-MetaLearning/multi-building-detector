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
    """Single Shot Multibox Detector with 300x300 inputs.
    This is a model of Single Shot Multibox Detector [#]_.
    This model uses :class:`~chainercv.links.model.ssd.VGG16Extractor300` as
    its feature extractor.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
       n_fg_class (int): The number of classes excluding the background.
       pretrained_model (str): The weight file to be loaded.
           This can take :obj:`'voc0712'`, `filepath` or :obj:`None`.
           The default value is :obj:`None`.
            * :obj:`'voc0712'`: Load weights trained on trainval split of \
                PASCAL VOC 2007 and 2012. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`20` or :obj:`None`. \
                These weights were converted from the Caffe model provided by \
                `the original implementation \
                <https://github.com/weiliu89/caffe/tree/ssd>`_. \
                The conversion code is `chainercv/examples/ssd/caffe2npz.py`.
            * :obj:`'imagenet'`: Load weights of VGG-16 trained on ImageNet. \
                The weight file is downloaded and cached automatically. \
                This option initializes weights partially and the rests are \
                initialized randomly. In this case, :obj:`n_fg_class` \
                can be set to any number.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.
    """

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
