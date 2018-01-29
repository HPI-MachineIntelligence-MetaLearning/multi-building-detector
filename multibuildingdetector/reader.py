from multibuildingdetector.datasets.ImageBoundingBoxDataset import ImageBoundingBoxDataset


def load_train_test_set(data_dir, test_dir, train_size, parser):
    annotations_train = parser.get_annotations(data_dir)
    annotations_test = parser.get_annotations(test_dir)
    return ImageBoundingBoxDataset(annotations_train), \
        ImageBoundingBoxDataset(annotations_test)
