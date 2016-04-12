import os
import sys
import mxnet as mx
import numpy as np
import cv2
import logging
import leveldb
import random
import shutil
from datetime import datetime
import data
import sym

def train(batch_size, prefix):
    logging.basicConfig(level=logging.DEBUG)
    flog = logging.FileHandler(filename=prefix+datetime.now().strftime('_%Y_%m_%d-%H_%M.log'), mode='w')
    flog.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(flog)

    scale = (-15, 17)
    data_frames = 1
    flow_frames = 0
    stride = 1
    method = 'multi2'
    if method == 'direct':
        no_left0 = True
        right_whiten = True
    else:
        no_left0 = False
        right_whiten = False

    tan = False
    net = sym.make_l1_sym(scale, 'sum', data_frames, flow_frames, method=method, tan=tan)
    train_data = data.Mov3dStack('data/lmdb', (384,160), batch_size, scale, output_depth=False,
                                 data_frames=data_frames, flow_frames=flow_frames, stride=stride, no_left0=no_left0, right_whiten=right_whiten)
    test_data = data.Mov3dStack('data/lmdb', (384,160), batch_size, scale, output_depth=False,
                                data_frames=data_frames, flow_frames=flow_frames,
                                test_mode=True, stride=stride, no_left0=no_left0, right_whiten=right_whiten)
    test_data = mx.io.ResizeIter(test_data, 100)

    checkpoint = mx.callback.do_checkpoint(prefix)

    def stat(x):
        return  mx.nd.norm(x)/np.sqrt(x.size)
    def std_stat(x):
        return [(mx.nd.norm(x)**2-mx.nd.sum(x)**2)/(x.size)]
    mon = mx.monitor.Monitor(1, stat, pattern='.*output.*', sort=True)#pred.*output.*|softmax_output|feature_output')
    
    def bilinear(name, arr):
        assert name.startswith('deconv_pred')
        weight = np.zeros(np.prod(arr.shape[2:]), dtype='float32')
        shape = arr.shape
        f = np.ceil(shape[3] / 2.)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(np.prod(shape[2:])):
            x = i % shape[3]
            y = (i / shape[3]) % shape[2]
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        weight2 = np.zeros(shape, dtype='float32')
        weight = weight.reshape(shape[2:])
        for i in range(shape[0]):
            weight2[i,i] = weight
        arr[:] = weight2.reshape(shape)

    vgg16 = data.load_vgg(data_frames, flow_frames, two_stream=False)
    init =  mx.init.Load(vgg16, mx.init.Uniform(0.01), True)
    init = mx.init.Mixed(['deconv_pred.*weight', '.*'], [bilinear, init])

    model = mx.model.FeedForward(
        ctx                = [mx.gpu(0)],
        symbol             = net,
        num_epoch          = 100,
        learning_rate      = 0.002,
        lr_scheduler       = mx.lr_scheduler.FactorScheduler(20*1000, 0.1),
        momentum           = 0.9,
        wd                 = 0.0000,
        optimizer          = 'ccsgd',
        initializer        = init,
        epoch_size         = 1000
        )

    model.fit(
        X                  = train_data,
        kvstore            = None,
        batch_end_callback = [mx.callback.Speedometer(batch_size, 10), mx.callback.log_train_metric(10, auto_reset=True)],
        epoch_end_callback = [checkpoint],
        #monitor            = mon,
        eval_metric        = mx.metric.MAE(),
        eval_data          = test_data)

if __name__ == '__main__':
    train(64, 'exp/deep3d')
