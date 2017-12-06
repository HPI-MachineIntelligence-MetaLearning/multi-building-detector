import chainer

from chainer.optimizer import WeightDecay
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links import SSD300
from chainercv.links.model.ssd import GradientScaling
from multibuildingdetector.transformers.augmentation import ImageAugmentation
from .ssdtrainchain import MultiboxTrainChain
from .readers import xmldataset
from chainer.datasets import TransformDataset
from chainer.training import extensions


def main():
    model = SSD300(n_fg_class=len(xmldataset.LABEL_NAMES),
                   pretrained_model='imagenet')
    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)

    # chainer.cuda.get_device_from_id(1).use()
    # model.to_gpu()

    data_path = '../labeled_images/'
    OUTPUT = 'result'
    BATCH_SIZE = 32

    train = TransformDataset(
        xmldataset.XMLDataset(data_path),
        ImageAugmentation(model.coder, model.insize, model.mean))
    train_iter = chainer.iterators.MultiprocessIterator(train, BATCH_SIZE)

    test = xmldataset.XMLDataset(data_path, split='test')
    test_iter = chainer.iterators.SerialIterator(
        test, BATCH_SIZE, repeat=False, shuffle=False)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(train_chain)

    # Need to find explaination
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = chainer.training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (120000, 'iteration'), OUTPUT)
    # trainer.extend(
    #    extensions.ExponentialShift('lr', 0.1, init=1e-3),
    #    trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=xmldataset.LABEL_NAMES),
        trigger=(10000, 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/loc', 'main/loss/conf',
         'validation/main/map']), )
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.run()


if __name__ == '__main__':
    main()
