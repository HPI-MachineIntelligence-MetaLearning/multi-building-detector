import chainer

from multibuildingdetector.loss.ssdtripletloss import SSDTripletLoss


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
        sddtripletloss = SSDTripletLoss(gt_mb_locs, gt_mb_labels, coder)
        loc_loss, conf_loss = sddtripletloss(mb_locs, mb_confs)
        print('Loc loss: {}, Conf Loss: {}'.format(loc_loss, conf_loss))
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss
