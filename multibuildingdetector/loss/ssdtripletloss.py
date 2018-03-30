import chainer.functions as F
import chainer

import itertools

import numpy as np
from chainercv import utils
from collections import defaultdict


class SSDTripletLoss:

    def __init__(self, gt_mb_locs, gt_mb_labels, coder,
                 nms_thresh=0.2):
        self.gt_mb_locs = gt_mb_locs
        self.gt_mb_labels = gt_mb_labels
        self.coder = coder
        self.nms_thresh = nms_thresh
        self.xp = coder.xp

    def __call__(self, mb_locs, mb_confs):
        positive = self.gt_mb_labels > 0
        n_positive = positive.sum()
        if n_positive == 0:
            z = chainer.Variable(self.xp.zeros((), dtype=np.float32))
            return z, z
        loc_loss = self._compute_loc_loss(mb_locs, n_positive, positive)
        triplet_batch_size, triplet_loss = self._compute_triplet_loss(mb_locs,
                                                                      mb_confs)
        return triplet_batch_size, loc_loss, triplet_loss

    def _decode_bbox(self, mb_loc):
        mb_bbox = self.coder._default_bbox.copy()
        mb_bbox[:, :2] += mb_loc[:, :2] * self.coder._variance[0] \
                                        * self.coder._default_bbox[:, 2:]
        mb_bbox[:, 2:] *= self.xp.exp(mb_loc[:, 2:] *
                                      self.coder._variance[1])

        # (center_y, center_x, height, width) -> (y_min, x_min, height, width)
        mb_bbox[:, :2] -= mb_bbox[:, 2:] / 2
        # (center_y, center_x, height, width) -> (y_min, x_min, y_max, x_max)
        mb_bbox[:, 2:] += mb_bbox[:, :2]
        return mb_bbox

    def _filter_overlapping_bboxs(self, mb_boxs, mb_confs):
        confs = []
        labels = []
        for box, conf, label in zip(mb_boxs, mb_confs, self.gt_mb_labels):
            indices = utils.non_maximum_suppression(box, self.nms_thresh)

            confs.append(conf[indices])
            if chainer.cuda.available:
                labels.append(label[indices].get())
            else:
                labels.append(label[indices])
        confs = F.concat(confs, axis=0)
        labels = np.concatenate(labels)
        return zip(labels, confs)

    def _compute_triplet_loss(self, mb_locs, mb_confs):
        mb_boxs = [self._decode_bbox(mb_loc) for mb_loc in mb_locs.array]
        labeled_features = self._filter_overlapping_bboxs(mb_boxs, mb_confs)
        anchors, positives, negatives = self._build_triplets(labeled_features)
        batch_size = len(anchors)
        if not anchors:
            return 0, chainer.Variable(self.xp.zeros((), dtype=np.float32))
        anchors = F.stack(anchors)
        positives = F.stack(positives)
        negatives = F.stack(negatives)
        return batch_size, F.triplet(anchors, positives, negatives)

    def _build_triplets(self, labeled_features):
        triplets = []

        label_groups = self._get_label_groups(labeled_features)

        if len(label_groups) < 2:
            return [], [], []
        for label, group in label_groups.items():

            # Do not choose background label as anchor
            if len(group) < 2 or label == 0:
                continue
            positives = itertools.combinations(group, 2)
            built_triplets = self._add_negatives(positives, label_groups, label)
            triplets = itertools.chain(triplets, built_triplets)
        anchors = []
        positives = []
        negatives = []
        for anchor, positive, negative in triplets:
            anchors.append(anchor)
            negatives.append(negative)
            positives.append(positive)
        return anchors, positives, negatives

    @staticmethod
    def _get_label_groups(labeled_features):
        label_groups = defaultdict(list)
        for label, feature in labeled_features:
            label_groups[label].append(feature)
        return label_groups

    @staticmethod
    def _add_negatives(positives, label_groups, positive_label):
        for label, group in label_groups.items():
            if label == positive_label or label == 0 or not group:
                continue
            for negative in group:
                for positive in positives:
                    pos_copy = positive[:]
                    pos_copy += (negative,)
                    yield pos_copy

    def _compute_loc_loss(self, mb_locs, n_positive, positive):
        loc_loss = F.huber_loss(mb_locs, self.gt_mb_locs, 1, reduce='no')
        loc_loss = F.sum(loc_loss, axis=-1)
        loc_loss *= positive.astype(loc_loss.dtype)
        loc_loss = F.sum(loc_loss) / n_positive
        return loc_loss
