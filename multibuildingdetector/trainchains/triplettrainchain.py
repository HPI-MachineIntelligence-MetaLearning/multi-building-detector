import chainer

from multibuildingdetector.loss.ssdtripletloss import SSDTripletLoss


class TripletTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1):
        super().__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.loss_labels = ['loss', 'loss/loc', 'loss/triplet']

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        coder = self.model.coder
        mb_locs, mb_confs = self.model(imgs)
        sdd_triplet_loss = SSDTripletLoss(gt_mb_locs, gt_mb_labels, coder)
        loc_loss, triplet_loss = sdd_triplet_loss(mb_locs, mb_confs)
        loss = loc_loss * self.alpha + triplet_loss

        chainer.reporter.report({
            self.loss_labels[0]: loss,
            self.loss_labels[1]: loc_loss,
            self.loss_labels[2]: triplet_loss}, self)

        return loss
