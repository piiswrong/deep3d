import mxnet as mx

def make_upsample_sym(data, scale, fuse='sum', method='multi2'):
    # group 1
    conv1_1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = mx.symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = mx.symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    pool3 = mx.symbol.Pooling(
        data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    pool4 = mx.symbol.Pooling(
        data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    pool5 = mx.symbol.Pooling(
        data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=512, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=512, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # output
    fc8 = mx.symbol.FullyConnected(data=drop7, num_hidden=33*12*5, name="pred5")

    pred5 = mx.symbol.Reshape(data=fc8, target_shape=(0, 33, 5, 12))
    bn_pool4 = mx.sym.CuDNNBatchNorm(data=pool4, name='bn_pool4')
    pred4 = mx.symbol.Convolution(data=bn_pool4, kernel=(3,3), pad=(1,1), num_filter=33, name="pred4")
    bn_pool3 = mx.sym.CuDNNBatchNorm(data=pool3, name='bn_pool3')
    pred3 = mx.symbol.Convolution(data=bn_pool3, kernel=(3,3), pad=(1,1), num_filter=33, name="pred3")
    bn_pool2 = mx.sym.CuDNNBatchNorm(data=pool2, name='bn_pool2')
    pred2 = mx.symbol.Convolution(data=bn_pool2, kernel=(3,3), pad=(1,1), num_filter=33, name="pred2")
    bn_pool1 = mx.sym.CuDNNBatchNorm(data=pool1, name='bn_pool1')
    pred1 = mx.symbol.Convolution(data=bn_pool1, kernel=(3,3), pad=(1,1), num_filter=33, name="pred1")

    no_bias = False
    assert fuse == 'sum'
    assert method == 'multi2'
    workspace = 0
    scale = 1
    pred1 = mx.symbol.Activation(data=pred1, act_type='relu')
    pred1 = mx.symbol.Deconvolution(data=pred1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_pred1')
    scale *= 2
    pred2 = mx.symbol.Activation(data=pred2, act_type='relu')
    pred2 = mx.symbol.Deconvolution(data=pred2, kernel=(2*scale, 2*scale), stride=(scale, scale), pad=(scale/2, scale/2), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_pred2')
    scale *= 2
    pred3 = mx.symbol.Activation(data=pred3, act_type='relu')
    pred3 = mx.symbol.Deconvolution(data=pred3, kernel=(2*scale, 2*scale), stride=(scale, scale), pad=(scale/2, scale/2), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_pred3')
    scale *= 2
    pred4 = mx.symbol.Activation(data=pred4, act_type='relu')
    pred4 = mx.symbol.Deconvolution(data=pred4, kernel=(2*scale, 2*scale), stride=(scale, scale), pad=(scale/2, scale/2), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_pred4')
    scale *= 2
    pred5 = mx.symbol.Activation(data=pred5, act_type='relu')
    pred5 = mx.symbol.Deconvolution(data=pred5, kernel=(2*scale, 2*scale), stride=(scale, scale), pad=(scale/2, scale/2), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_pred5')
    feat = mx.symbol.ElementWiseSum(pred1, pred2, pred3, pred4, pred5)
    feat_act = mx.symbol.Activation(data=feat, act_type='relu', name='feat_relu')
    scale = 2
    up = mx.symbol.Deconvolution(data=feat_act, kernel=(2*scale, 2*scale), stride=(scale, scale), pad=(scale/2, scale/2), num_filter=33, no_bias=no_bias, workspace=workspace, name='deconv_predup')
    up = mx.symbol.Activation(data=up, act_type='relu')
    up = mx.symbol.Convolution(data=up, kernel=(3,3), pad=(1,1), num_filter=33)
    
    return up, [pred1, pred2, pred3, pred4, pred5]

def make_l1_sym(scale, fuse='sum', data_frames=1, flow_frames=1, get_softmax=False, method='multi2', upsample=1, tan=False):
    left = mx.symbol.Variable(name="left")
    flow = mx.symbol.Variable(name="flow")

    left0 = mx.symbol.Variable(name='left0')

    if data_frames and flow_frames:
        data = mx.symbol.Concat(left, flow, dim=1, name='data')
    elif data_frames:
        data = left
    elif flow_frames:
        data = flow

    data, _ = make_upsample_sym(data, scale, fuse, method=method)
    if tan:
        data = mx.symbol.Activation(data=data, act_type='tanh')
    if method == 'direct':
        depthdot = data
    else:
        softmax = mx.symbol.SoftmaxActivation(data=data, type='channel', name='softmax')
        depthdot = mx.symbol.DepthDot(data=softmax, label=left0, scale=scale, upsample=upsample, name='depth')
        
    loss = mx.symbol.MAERegressionOutput(data=depthdot, name='l1')
    if get_softmax:
        return mx.symbol.Group([depthdot, softmax])
    else:
        return loss