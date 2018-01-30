import copy
from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

import chainer
import chainer.functions as F
from chainer import reporter
import chainer.training.extensions
from multibuildingdetector.loss.ssdtripletloss import SSDTripletLoss
from scipy.spatial.distance import pdist, cdist, euclidean
from sklearn.decomposition import PCA
from sklearn import metrics

from chainercv.utils import apply_prediction_to_iterator, \
    non_maximum_suppression


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

        mb_boxs, mb_confs = pred_values

        _, gt_labels = gt_values

        report = {}

        label_groups = defaultdict(list)

        for labels, confs in zip(gt_labels, mb_confs):
            label_groups.update(SSDTripletLoss._get_label_groups(
                self.filter_overlapping_bboxs(mb_boxs, mb_confs,
                                              gt_labels)))
        del label_groups[0]

        if self._save:
            self.plot_roc_curves(label_groups)

        for label, feat_v in label_groups.items():
            avg_dist = -1
            feat_v = np.array([x.data for x in feat_v])
            if feat_v.shape[0] > 1:
                distances = pdist(feat_v)
                avg_dist = mean(distances)
            label_name = self.label_names[label]
            report['main/avg_dist/{}'.format(label_name)] = avg_dist
            if self._save:
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(feat_v)
                plt.scatter([x[0] for x in pca_data],
                            [x[1] for x in pca_data],
                            label=label_name)
        if self._save:
            plt.legend()
            plt.savefig(self._save_path + '/triplet_scatter.jpg')
            plt.clf()
        print(report)

        observation = dict()
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation

    def filter_overlapping_bboxs(self, mb_boxs, mb_confs, gt_labels):
        confs = []
        labels = []
        for box, conf, label in zip(mb_boxs, mb_confs, gt_labels):
            indices = non_maximum_suppression(box, 0.5)
            # Add more beatiful version of thi nms-thresh

            confs.append(conf[indices])
            if chainer.cuda.available:
                labels.append(label[indices].get())
            else:
                labels.append(label[indices])
        confs = F.concat(confs, axis=0)
        labels = np.concatenate(labels)
        return zip(labels, confs)

    def center_point(self, points):
        center = []
        for i in range(points.shape[1]):
            center.append(sum(points[:, i]) / points.shape[1])
        return np.array(center)

    def plot_roc_curves(self, label_groups):
        for label_test in label_groups.keys():
            test_label = self.label_names[label_test]
            # Note: we don't need to subtract 1 here, we already
            # deleted the background label
            for label_center in [k for k in label_groups.keys()
                                 if k != label_test]:
                # label_center is the class the centroid is built for,
                # for label_test the ROC curve will be created
                center_label = self.label_names[label_center]
                feat_v = np.array([x.data for x in label_groups[label_center]])
                ctroid = self.center_point(feat_v)
                ctroid_arr = np.full(feat_v.shape, ctroid)
                mean_dist = mean([x[0] for x in cdist(feat_v, ctroid_arr)])
                predicted = []
                for feat_center in label_groups[label_center]:
                    dist = euclidean(ctroid, feat_center.data)
                    predicted.append(int(dist <= mean_dist))
                for feat_test in label_groups[label_test]:
                    dist = euclidean(ctroid, feat_test.data)
                    predicted.append(int(dist > mean_dist))
                actual = np.append(np.zeros(len(label_groups[label_center])),
                                   np.ones(len(label_groups[label_test])))
                fpr, tpr, _ = metrics.roc_curve(actual, predicted)
                plt.plot(fpr, tpr, label='ROC curve compared to {}'
                         .format(center_label))
                plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic {}'
                          .format(test_label))
                plt.legend(loc="lower right")
            plt.savefig(self._save_path + '/{}_ROC.jpg'.format(test_label))
            plt.clf()
