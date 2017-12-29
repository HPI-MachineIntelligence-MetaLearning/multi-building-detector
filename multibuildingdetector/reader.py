from multibuildingdetector.datasets.ImageBoundingBoxDataset import ImageBoundingBoxDataset
from sklearn.model_selection import train_test_split


def load_train_test_set(data_dir, train_size, parser):
    annotations = parser.get_annotations(data_dir)
    train, test = train_test_split(annotations, train_size=train_size)
    return ImageBoundingBoxDataset(train), ImageBoundingBoxDataset(test)
