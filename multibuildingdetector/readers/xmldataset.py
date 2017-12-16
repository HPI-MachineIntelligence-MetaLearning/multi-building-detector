import numpy as np
import xml.etree.ElementTree as ET
import chainer

from chainercv.utils import read_image
from os import listdir
from os.path import isfile, join

LABEL_NAMES = ('other',
               'berlinerdom',
               'brandenburgertor',
               'fernsehturm',
               'funkturm',
               'reichstag',
               'rotesrathaus',
               'siegessaeule',
               'none')


class XMLDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir, split='train'):
        self._img_ids = [join(data_dir, ''.join(f.split('.')[:-1])) for f in
                         listdir(data_dir) if isfile(join(data_dir, f))]
        self._split = split

    def __len__(self):
        return int(len(self._img_ids) / 2)

    def get_example(self, i):
        """Returns the i-th example.
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.
        Args:
            i (int): The index of the example.
        Returns:
            tuple of an image and bounding boxes and label
        """
        id_ = self._img_ids[i]
        bbox = list()
        label = list()
        try:
            anno = ET.parse(id_ + '.xml')
            for obj in anno.findall('object'):
                bndbox_anno = obj.find('bndbox')
                # subtract 1 to make pixel indexes 0-based
                bbox.append([
                    int(bndbox_anno.find(tag).text) - 1
                    for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
                name = obj.find('name').text.lower().strip()
                label.append(LABEL_NAMES.index(name))
        except FileNotFoundError:
            # No buildings present in this image,
            # but dimensions have to be consistent
            bbox.append([0, 0, 0, 0])
            label.append(LABEL_NAMES.index('none'))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # Load a image
        img_file = id_ + '.jpg'
        img = read_image(img_file, color=True)
        return img, bbox, label
