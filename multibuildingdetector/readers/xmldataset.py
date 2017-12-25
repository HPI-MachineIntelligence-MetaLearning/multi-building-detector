import numpy as np
import xml.etree.ElementTree as ET
import chainer
from chainercv.utils import read_image

class XMLDataset(chainer.dataset.DatasetMixin):

    def __init__(self, annotations):
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def get_example(self, i):
        img_file, bbox, label = self.annotations[i]
        img = read_image(img_file, color=True)
        return img, bbox, label
