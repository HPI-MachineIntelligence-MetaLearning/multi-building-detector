# set matplotlib backend to run without Xwindows
import matplotlib
matplotlib.use('Agg')

import argparse
import importlib
import os
import chainer
import yaml
import json

from chainer.datasets import TransformDataset
from chainer.optimizer import WeightDecay
from chainer.training import extensions
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from os.path import dirname, join, realpath

from multibuildingdetector.transforms.augmentation import ImageAugmentation
from multibuildingdetector.reader import load_train_test_set
from multibuildingdetector.evaluators.tripletevaluator import TripletEvaluator

PROJECT_DIR = join(dirname(realpath(__file__)), '..')


def load_config(path):
    with open(os.path.join(PROJECT_DIR, path), 'r') as f:
        return yaml.load(f)


def _import_module(package_template, module=''):
    return importlib.import_module(package_template.format(module))


def _import_class(class_path):
    split = class_path.split('.')
    class_name = split[-1]
    package = '.'.join(split[:-1])
    module = _import_module(package)
    return getattr(module, class_name)


def run(input_dir, test_dir, output, batch_size,
        iterator='SerialIterator',
        device=-1, pretrained_model='', save_trigger=10000,
        test_trigger=1000,
        parser_module='XMLParser',
        train_module='MultiboxTrainChain',
        model_module='chainercv.links.SSD300'):
    pretrained_model = join(PROJECT_DIR, pretrained_model)
    if pretrained_model and os.path.isfile(pretrained_model):
        print('Pretrained model {} loaded.'.format(pretrained_model))
    else:
        print('Pretrained model file not found, ' +
              'using imagenet as default.')
        pretrained_model = 'imagenet'
    parser = _import_module('multibuildingdetector.parsers.{}', parser_module)

    model = _import_class(model_module)(n_fg_class=len(parser.LABEL_NAMES),
                                        pretrained_model=pretrained_model)
    model.use_preset('evaluate')
    train_chain = _import_class('multibuildingdetector.trainchains.{}'
                                .format(train_module))(model)
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    train, test = load_train_test_set(input_dir, test_dir, parser)

    augmented_train = TransformDataset(
        train,
        ImageAugmentation(model.coder, model.insize, model.mean))
    train_iter = getattr(chainer.iterators, iterator)(augmented_train,
                                                      batch_size)

    test_iter = chainer.iterators.SerialIterator(
        TransformDataset(
            test,
            ImageAugmentation(model.coder, model.insize, model.mean)),
        batch_size, repeat=False, shuffle=False)

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

    log_fields = [*train_chain.loss_labels]
    if train_module == 'MultiboxTrainChain':
        trainer.extend(
            DetectionVOCEvaluator(
                test_iter, model, use_07_metric=True,
                label_names=parser.LABEL_NAMES),
            trigger=(test_trigger, 'iteration'))
        log_fields.append('validation/main/map')
    else:
        trainer.extend(
            TripletEvaluator(
                test_iter, model,
                label_names=parser.LABEL_NAMES,
                save_plt=True,
                save_path=output),
            trigger=(test_trigger, 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         *log_fields]))
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
    print('Configuration: ', json.dumps(config, indent=4))
    run(**config)


if __name__ == '__main__':
    main()
