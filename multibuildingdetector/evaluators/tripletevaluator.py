import copy
import numpy as np
from collections import defaultdict
from statistics import mean

from chainer import reporter
import chainer.training.extensions
from multibuildingdetector.loss.ssdtripletloss import SSDTripletLoss
from scipy.spatial.distance import pdist


from chainercv.utils import apply_prediction_to_iterator


class TripletEvaluator(chainer.training.extensions.Evaluator):

    """An extension that evaluates a detection model by PASCAL VOC metric.
    This extension iterates over an iterator and evaluates the prediction
    results by average precisions (APs) and mean of them
    (mean Average Precision, mAP).
    This extension reports the following values with keys.
    Please note that :obj:`'ap/<label_names[l]>'` is reported only if
    :obj:`label_names` is specified.
    * :obj:`'map'`: Mean of average precisions (mAP).
    * :obj:`'ap/<label_names[l]>'`: Average precision for class \
        :obj:`label_names[l]`, where :math:`l` is the index of the class. \
        For example, this evaluator reports :obj:`'ap/aeroplane'`, \
        :obj:`'ap/bicycle'`, etc. if :obj:`label_names` is \
        :obj:`~chainercv.datasets.voc_bbox_label_names`. \
        If there is no bounding box assigned to class :obj:`label_names[l]` \
        in either ground truth or prediction, it reports :obj:`numpy.nan` as \
        its average precision. \
        In this case, mAP is computed without this class.
    Args:
        iterator (chainer.Iterator): An iterator. Each sample should be
            following tuple :obj:`img, bbox, label` or
            :obj:`img, bbox, label, difficult`.
            :obj:`img` is an image, :obj:`bbox` is coordinates of bounding
            boxes, :obj:`label` is labels of the bounding boxes and
            :obj:`difficult` is whether the bounding boxes are difficult or
            not. If :obj:`difficult` is returned, difficult ground truth
            will be ignored from evaluation.
        target (chainer.Link): A detection link. This link must have
            :meth:`predict` method that takes a list of images and returns
            :obj:`bboxes`, :obj:`labels` and :obj:`scores`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
        label_names (iterable of strings): An iterable of names of classes.
            If this value is specified, average precision for each class is
            also reported with the key :obj:`'ap/<label_names[l]>'`.
    """

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, iterator, target, label_names=None):
        super(TripletEvaluator, self).__init__(
            iterator, target)
        self.label_names = label_names

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
                print(label, avg_dist)
                report[label - 1] = avg_dist

        observation = dict()
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
