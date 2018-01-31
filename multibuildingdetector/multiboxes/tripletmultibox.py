import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class TripletMultibox(chainer.Chain):
    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        self._input_multiplier = 256

        super().__init__()
        with self.init_scope():
            self.loc = chainer.ChainList()
            self.features = chainer.ChainList()

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2
            self.loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.features.add_link(L.Convolution2D(
                n * self._input_multiplier, 3, pad=1, **init))

    def __call__(self, xs):
        mb_locs = list()
        mb_confs = list()
        for i, x in enumerate(xs):
            mb_loc = self.loc[i](x)
            mb_loc = F.transpose(mb_loc, (0, 2, 3, 1))
            mb_loc = F.reshape(mb_loc, (mb_loc.shape[0], -1, 4))
            mb_locs.append(mb_loc)

            mb_conf = self.features[i](x)
            mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
            mb_conf = F.reshape(
                mb_conf, (mb_conf.shape[0], -1, self._input_multiplier))
            mb_confs.append(mb_conf)

        mb_locs = F.concat(mb_locs, axis=1)
        mb_confs = F.concat(mb_confs, axis=1)

        return mb_locs, mb_confs
