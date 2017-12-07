import chainer
import argparse
import yaml

from chainer.optimizer import WeightDecay
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links import SSD300
from chainercv.links.model.ssd import GradientScaling
from multibuildingdetector.transforms.augmentation import ImageAugmentation
from multibuildingdetector.multiboxtrainchain import MultiboxTrainChain
from .readers import xmldataset
from chainer.datasets import TransformDataset
from chainer.training import extensions


def load_config(path):
    with open(path, 'r') as f:
        return yaml.load(f)


def run(input, output, batch_size, iterator='SerialIterator', device=-1):
    model = SSD300(n_fg_class=len(xmldataset.LABEL_NAMES),
                   pretrained_model='imagenet')
    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)

    if device > 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    train = TransformDataset(
        xmldataset.XMLDataset(input),
        ImageAugmentation(model.coder, model.insize, model.mean))
    train_iter = getattr(chainer.iterators, iterator)(train, batch_size)

    test = xmldataset.XMLDataset(input, split='test')
    test_iter = chainer.iterators.SerialIterator(
        test, batch_size, repeat=False, shuffle=False)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(train_chain)

    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    if device is None:
        updater = chainer.training.StandardUpdater(train_iter, optimizer)
    else:
        updater = chainer.training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (120000, 'iteration'), output)

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


def main():
    parser = argparse.ArgumentParser(description='Running configured \
                                                  multibox detector')
    parser.add_argument('-c', '--config', help='Path to config', required=True)

    config_path = vars(parser.parse_args())['config']
    config = load_config(config_path)
    run(**config)


if __name__ == '__main__':
    main()
