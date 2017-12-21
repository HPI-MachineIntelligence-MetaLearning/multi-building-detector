from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

from .xmldataset import XMLDataset


def load_train_test_set(data_dir, train_size=0.8, img_type='jpg'):
    img_paths = [join(data_dir, ''.join(f.split('.')[:-1])) for f in
                 listdir(data_dir) if isfile(join(data_dir, f)) and
                 f.split('.')[-1] == img_type]
    train, test = train_test_split(img_paths, train_size=train_size)
    print("Split: {0}, Train size: {1}, Test size: {2}"
          .format(train_size, len(train), len(test)))
    return XMLDataset(train), XMLDataset(test)
