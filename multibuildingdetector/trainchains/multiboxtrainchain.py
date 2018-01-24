import chainer

from chainercv.links.model.ssd import multibox_loss


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super().__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k
        self.loss_labels = ['loss', 'loss/loc', 'loss/conf']

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report({
            self.loss_labels[0]: loss,
            self.loss_labels[1]: loc_loss,
            self.loss_labels[2]: conf_loss}, self)

        return loss
