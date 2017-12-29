import chainer
import argparse
import yaml
import os
import importlib

from chainer.optimizer import WeightDecay
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links import SSD300
from chainercv.links.model.ssd import GradientScaling
from multibuildingdetector.transforms.augmentation import ImageAugmentation
from multibuildingdetector.multiboxtrainchain import MultiboxTrainChain
from chainer.datasets import TransformDataset
from chainer.training import extensions
from .reader import load_train_test_set


def load_config(path):
    with open(path, 'r') as f:
        return yaml.load(f)


def run(input_dir, output, batch_size, train_split=0.8, iterator='SerialIterator',
        device=-1, pretrained_model='', save_trigger=10000,
        parser_module='XMLParser'):
    if pretrained_model and os.path.isfile(pretrained_model):
        print('Pretrained model {} loaded.'.format(pretrained_model))
    else:
        print('Pretrained model file not found, ' +
              'using imagenet as default.')
        pretrained_model = 'imagenet'
    parser = importlib.import_module('multibuildingdetector.parsers.{}'
                                     .format(parser_module))
    model = SSD300(n_fg_class=len(parser.LABEL_NAMES),
                   pretrained_model=pretrained_model)
    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)

    if device > 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    train, test = load_train_test_set(input_dir, train_split, parser)

    augmented_train = TransformDataset(
        train,
        ImageAugmentation(model.coder, model.insize, model.mean))
    train_iter = getattr(chainer.iterators, iterator)(augmented_train, batch_size)

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
        updater = chainer.training.StandardUpdater(train_iter, optimizer,
                                                   device=device)
    trainer = chainer.training.Trainer(updater, (120000, 'iteration'), output)

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=parser.LABEL_NAMES),
        trigger=(10000, 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/loc', 'main/loss/conf',
         'validation/main/map']), )
    trainer.extend(extensions.snapshot_object(
                   model, 'model_iter_{.updater.iteration}'),
                   trigger=(save_trigger, 'iteration'))

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
