from os import listdir
import numpy as np
from os.path import isfile, join
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from chainercv.utils import read_image

from .xmldataset import XMLDataset


LABEL_NAMES = ('other',
               'berlinerdom',
               'brandenburgertor',
               'fernsehturm',
               'funkturm',
               'reichstag',
               'rotesrathaus',
               'siegessaeule',
               'none')

def load_train_test_set(data_dir, train_size=0.8, img_type='jpg'):
    img_paths = [join(data_dir, ''.join(f.split('.')[:-1])) for f in
                 listdir(data_dir) if isfile(join(data_dir, f)) and
                 f.split('.')[-1] == img_type]

    # includes image path, bounding box and label
    # possibly use id as key to retrieve data
    annotations = list(map(lambda img_path: parse_annotations(img_path), img_paths))

    train, test = train_test_split(annotations, train_size=train_size)
    print("Split: {0}, Train size: {1}, Test size: {2}"
          .format(train_size, len(train), len(test)))
    return XMLDataset(train), XMLDataset(test)


def parse_annotations(img_path):
        bbox = list()
        label = list()
        try:
            anno = ET.parse(img_path + '.xml')
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

        img_file = img_path + '.jpg'
        return img_file, bbox, label