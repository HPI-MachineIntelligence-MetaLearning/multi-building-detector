import chainer
import numpy as np
import itertools

from chainer import functions as F
from collections import defaultdict


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        coder = self.model.coder
        mb_locs, mb_confs = self.model(imgs)
        print('Img size: ', len(imgs), 'Conf size: ', len(mb_confs), 'Label size: ', len(gt_mb_labels))
        print('matrix size: ', mb_confs.shape, 'Label shape', gt_mb_labels.shape)
        xp = chainer.cuda.get_array_module(gt_mb_labels)
        loc_loss, conf_loss = compute_loss(mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, xp)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


def compute_loc_loss(mb_locs, gt_mb_locs, gt_mb_labels, xp):
    positive = gt_mb_labels.array > 0
    n_positive = positive.sum()
    if n_positive == 0:
        z = chainer.Variable(xp.zeros((), dtype=np.float32))
        return z, z

    loc_loss = F.huber_loss(mb_locs, gt_mb_locs, 1, reduce='no')
    loc_loss = F.sum(loc_loss, axis=-1)
    loc_loss *= positive.astype(loc_loss.dtype)
    loc_loss = F.sum(loc_loss) / n_positive
    return loc_loss


def compute_loss(mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, xp):
    mb_locs = chainer.as_variable(mb_locs)
    gt_mb_locs = chainer.as_variable(gt_mb_locs)
    gt_mb_labels = chainer.as_variable(gt_mb_labels)

    loc = compute_loc_loss(mb_locs, gt_mb_locs, gt_mb_labels, xp)
    triplet = compute_triplet_loss(mb_confs, gt_mb_labels, xp)
    return loc, triplet


def compute_triplet_loss(mb_confs, gt_mb_labels, xp):
    aspect_ratio_imgs = [defaultdict(list) for _ in range(mb_confs.shape[2])]
    triplet_loss = 0
    for confs, encoded_label in zip(mb_confs, gt_mb_labels):
        confs = F.transpose(confs)
        for i, conf in enumerate(confs):
            # HACK
            label = str(encoded_label)
            print(label)
            aspect_ratio_imgs[i][label].append(conf)
    # DEBUG
    # for i in aspect_ratio_imgs:
    #     for key, value in i.items():
    #         print("aspect ratios: ", key[:5], len(value))

    aspect_ratio_triplets = [build_triplets(group) for group in aspect_ratio_imgs]
    for ratio_triplets in aspect_ratio_triplets:
        positives, anchors, negatives = zip(*ratio_triplets)
        triplet_loss += F.triplet(xp.stack(positives),
                                  xp.stack(anchors), xp.stack(negatives))
    return triplet_loss


def build_triplets(label_groups):
    triplets = []
    if len(label_groups) < 3:
        return []
    for label, group in label_groups.items():
        if len(group) < 2:
            continue
        positives = itertools.combinations(group, 2)
        triplets = itertools.chain(triplets,
                                   add_negatives(positives,
                                                 label_groups, label))
    return triplets


def add_negatives(positives, label_groups, positive_label):
    for label, group in label_groups.items():
        if label == positive_label or not group:
            continue
        for negative in group:
            for positive in positives:
                pos_copy = positive[:]
                pos_copy.append(negative)
                yield pos_copy
