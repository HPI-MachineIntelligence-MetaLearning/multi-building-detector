import xml.etree.ElementTree as ET
import numpy as np

from os.path import isfile, join
from os import listdir

LABEL_NAMES = ('other',
               'berlinerdom',
               'brandenburgertor',
               'fernsehturm',
               'funkturm',
               'reichstag',
               'rotesrathaus',
               'siegessaeule')


def get_annotations(data_dir):
    img_paths = [join(data_dir, ''.join(f.split('.')[:-1])) for f in
                 listdir(data_dir) if isfile(join(data_dir, f)) and
                 f.split('.')[-1] == 'jpg']

    return [_parse_annotations(img_path) for img_path in img_paths]


def _parse_annotations(img_path):
    bbox = list()
    label = list()
    anno = ET.parse(img_path + '.xml')
    for obj in anno.findall('object'):
        bndbox_anno = obj.find('bndbox')
        # subtract 1 to make pixel indexes 0-based
        bbox.append([
            int(bndbox_anno.find(tag).text) - 1
            for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
        name = obj.find('name').text.lower().strip()
        label.append(LABEL_NAMES.index(name))
    bbox = np.stack(bbox).astype(np.float32)
    label = np.stack(label).astype(np.int32)

    img_file = img_path + '.jpg'
    return img_file, bbox, label
