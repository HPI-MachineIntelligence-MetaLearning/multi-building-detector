import copy
from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt

from chainer import reporter
import chainer.training.extensions
from multibuildingdetector.loss.ssdtripletloss import SSDTripletLoss
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from chainercv.utils import apply_prediction_to_iterator


class TripletEvaluator(chainer.training.extensions.Evaluator):

    """An extension that evaluates a triplet loss model by reporting the
    average distance between the individual feature vectors.
    This extension reports the following values with keys.
    * :obj:`'avg_dist/<label_names[l]>'`: Average distance for class \
        :obj:`label_names[l]`, where :math:`l` is the index of the class. \
    Args:
        iterator (chainer.Iterator): An iterator. Each sample should be
            following tuple :obj:`img, bbox, label`
            :obj:`img` is an image, :obj:`bbox` is coordinates of bounding
            boxes, :obj:`label` is labels of the bounding boxes
            (encoded in default bbox space!).
        target (chainer.Link): A detection link. This link must have
            :meth:`predict` method that takes a list of images and returns
            following tuple :obj:`multibox_locs, multibox_triplets`.
            :obj:`multibox_locs` is a vector containing the bbox locations
            encoded in the default bbox space,
            :obj:`multibox_triplets` is a vector containing the triplet loss
            feature vectors encoded in the default bbox space.
        label_names (iterable of strings): An iterable of names of classes.
    """

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, iterator, target, label_names=None,
            save_plt=False, save_path='result'):
        super(TripletEvaluator, self).__init__(
            iterator, target)
        self.label_names = label_names
        self._save = save_plt
        self._save_path = save_path

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            target.predict, it)
        # delete unused iterator explicitly
        del imgs

        _, mb_confs = pred_values

        _, gt_labels = gt_values

        report = {}

        label_groups = defaultdict(list)

        for labels, confs in zip(gt_labels, mb_confs):
            label_groups.update(SSDTripletLoss._get_label_groups(
                zip(labels, confs)))
        for label, feat_v in label_groups.items():
            if label != 0:
                distances = pdist(feat_v)
                avg_dist = mean(distances)
                label_name = self.label_names[label - 1]
                report['avg_dist/{}'.format(label_name)] = avg_dist
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(feat_v)
                plt.scatter([x[0] for x in pca_data],
                            [x[1] for x in pca_data],
                            label=label_name)
        plt.legend()
        if self._save:
            plt.savefig(self._save_path + '/triplet_scatter.jpg')
        print(report)
        plt.clf()

        observation = dict()
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
