import chainer

from multibuildingdetector.loss.ssdtripletloss import SSDTripletLoss


class TripletTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1):
        super().__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        coder = self.model.coder
        mb_locs, mb_confs = self.model(imgs)
        sdd_triplet_loss = SSDTripletLoss(gt_mb_locs, gt_mb_labels, coder)
        loc_loss, triplet_loss = sdd_triplet_loss(mb_locs, mb_confs)
        loss = loc_loss * self.alpha + triplet_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/triplet': triplet_loss},
            self)

        return loss
