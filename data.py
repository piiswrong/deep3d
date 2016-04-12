import os
import io
import sys
import mxnet as mx
import numpy as np
import cv2
import logging
import lmdb
import leveldb
import random
import shutil
from datetime import datetime
from PIL import Image
import StringIO
import time
from multiprocessing.pool import ThreadPool
import argparse

def get_prop(txn, key, default, check=False):
    v = txn.get(key)
    if v is None:
        txn.put(key, str(default))
        v = default
    v = type(default)(v)
    if check:
        assert v == default, "%s != %s"%(str(v), str(default))
    return v

def addto_db(db_path, path, suffix_list, chunk_size=24*60*3, chunk_base=10000, movie_base=1000000):
    def show_progress(name):
        if show_progress.i % 1000 == 0:
            print name, show_progress.i
        show_progress.i += 1

    prefix = os.path.basename(path)
    N = 0
    sprefix_list = ''
    sidx = ''
    env = lmdb.open(db_path, map_size=1<<40, max_dbs=5)
    with env.begin(write=True) as txn:
        N = get_prop(txn, 'N', N)
        chunk_size = get_prop(txn, 'chunk_size', chunk_size)
        chunk_base = get_prop(txn, 'chunk_base', chunk_base)
        movie_base = get_prop(txn, 'movie_base', movie_base)

        sprefix_list = txn.get('prefix_list', sprefix_list)
        prefix_list = filter(None, sprefix_list.split(','))
        sidx = txn.get('idx', sidx)
        idx = filter(None, sidx.split('$'))
        if prefix in prefix_list:
            print 'Prefix %s already in database!'%prefix
            return

    prev_ichunk = 0
    prev_count = 0
    for isuffix, suffix in enumerate(suffix_list):
        db = env.open_db(suffix)
        rec = mx.recordio.MXRecordIO('%s_%s.rec'%(path, suffix), 'r')
        base = N*movie_base
        chunks = [[]]
        ichunk = 0
        count = 0
        show_progress.i = 0
        with env.begin(write=True) as txn:
            while True:
                s = rec.read()
                if s is None:
                    break
                if count == chunk_size:
                    count = 0
                    ichunk += 1
                    chunks.append([])
                i = base + ichunk*chunk_base + count
                chunks[-1].append(str(i))
                txn.put('%09d'%i, s, db=db)
                show_progress(prefix+' '+suffix)
                count += 1

        assert isuffix == 0 or (prev_ichunk == ichunk and count == prev_count), '%d vs %d or %d vs %d'%(prev_ichunk, ichunk, count, prev_count)
        prev_ichunk = ichunk
        prev_count = count

    print 'Processed %d chunks'%len(chunks)
    prefix_list.append(prefix)
    idx.append('|'.join([','.join(c) for c in chunks]))

    with env.begin(write=True) as txn:
        txn.put('N', str(N+1))
        txn.put('prefix_list', ','.join(prefix_list))
        txn.put('idx', '$'.join(idx))

def show_db(fdb, suffix_list):
    print 'Showing db ' + fdb
    env = lmdb.open(fdb, max_dbs=5)
    with env.begin() as txn:
        N = int(txn.get('N'))
        chunk_size = int(txn.get('chunk_size'))
        chunk_base = int(txn.get('chunk_base'))
        movie_base = int(txn.get('movie_base'))
        prefix_list = txn.get('prefix_list')
        idx = [[[int(k) for k in j.split(',')] for j in i.split('|')] for i in txn.get('idx').split('$')]
        print len(idx)
        print prefix_list
        print N, chunk_size, chunk_base, movie_base

    assert len(prefix_list.split(',')) == N
    assert len(idx) == N

    for i in range(N):
        print 'mov', i
        base = i * movie_base
        ichunk = 0
        while True:
            pj = None
            for suffix in suffix_list:
                with env.begin(db=env.open_db(suffix)) as txn: 
                    for j in range(chunk_base):
                        v = txn.get('%09d'%(base+ichunk*chunk_base+j))
                        if v is None:
                            break
                        assert base+ichunk*chunk_base+j == idx[i][ichunk][j]
                assert pj is None or pj == j
                pj = j
            if j > 0:
                assert ichunk < len(idx[i])
                assert j == len(idx[i][ichunk]), '%d vs %d'%(j, len(idx[i][ichunk]))
                print 'chunk', ichunk
                ichunk += 1
            if j < chunk_size-1:
                print 'mov', i, 'finish at', ichunk
                break
        assert ichunk == len(idx[i])

def shuffle(path, valid_ratio=0.1, test_ratio=0.3):
    sidx = ''
    env = lmdb.open(path, map_size=1<<40, max_dbs=5)
    with env.begin() as txn:
        sidx = txn.get('idx')
        prefix_list = txn.get('prefix_list').split(',')
    idx = [[[k for k in i.split(',')] for i in j.split('|')][1:-3] for j in sidx.split('$')]
    p = list(range(len(idx)))
    random.shuffle(p)
    idx = [idx[i] for i in p]
    prefix_list = [prefix_list[i] for i in p]

    sep = int((1-test_ratio)*len(idx))

    print 'train ', prefix_list[:sep]
    print 'test ', prefix_list[sep:]

    train_idx = sum(idx[:sep], [])
    random.shuffle(train_idx)
    vsep = int((1-valid_ratio)*len(train_idx))
    valid_idx = sum(train_idx[vsep:], [])
    random.shuffle(valid_idx)
    svalid_idx = ','.join(valid_idx)

    train_idx = sum(train_idx[:vsep], [])
    random.shuffle(train_idx)
    strain_idx = ','.join(train_idx)

    test_idx = sum(idx[sep:], [])
    random.shuffle(test_idx)
    test_idx = sum(test_idx, [])
    stest_idx = ','.join(test_idx)

    random.shuffle(test_idx)
    sshuffled_test_idx = ','.join(test_idx)

    print len(train_idx), len(valid_idx), len(test_idx)
    with env.begin(write=True) as txn:
        txn.put('train_idx', strain_idx)
        txn.put('valid_idx', svalid_idx)
        txn.put('test_idx', stest_idx)
        txn.put('shuffled_test_idx', sshuffled_test_idx)

def make_idx(path):
    env = lmdb.open(path, map_size=1<<40)
    with env.begin() as txn:
        idx = [[[k for k in i.split(',')] for i in j.split('|')] for j in txn.get('idx').split('$')]
        idx = [sum(i, []) for i in idx]
        prefix = [i for i in txn.get('prefix_list').split(',')]
    with env.begin(write=True) as txn:
        for k, v in zip(prefix, idx):
            txn.put(k+'_idx', ','.join(v))

    with env.begin() as txn:
        for k, v in txn.cursor():
            print k

def crop_img(img, p, shape, margin=0, test=False, grid=1):
    if p is None:
        if test:
            p = ((img.shape[1]-shape[0]-margin)/grid/2, (img.shape[0]-shape[1])/grid/2)
        else:
            p = (random.randint(0, (img.shape[1]-shape[0]-margin)/grid),
                 random.randint(0, (img.shape[0]-shape[1])/grid))
    return img[p[1]*grid:p[1]*grid+shape[1], p[0]*grid:p[0]*grid+shape[0]], p

def load_mean(fname, data_iter, label_mean=False, include=None):
    if not fname.endswith('.npz'):
        fname += '.npz'
    if os.path.isfile(fname):
        return np.load(fname)
    else:
        print('Mean file %s not found. Computing mean...'%fname)
        data_iter.reset()
        mean_list = [np.zeros(shape[1:], dtype=np.float64) for name, shape in data_iter.provide_data]
        name_list = [name for name, shape in data_iter.provide_data]
        if label_mean:
            mean_list += [np.zeros(shape[1:], dtype=np.float64) for name, shape in data_iter.provide_label]
            name_list += [name for name, shape in data_iter.provide_label]
        count = 0
        last_mean_list = None
        for batch in data_iter:
            arr_list = batch.data
            if label_mean:
                arr_list += batch.label
            for name, arr, mean in zip(name_list, arr_list, mean_list):
                if include is None or name in include:
                    arr = arr.asnumpy()
                    mean += arr[:arr.shape[0]-batch.pad].sum(axis=0)
            inc = batch.data[0].shape[0]-batch.pad
            count += inc
            if count/1000 > (count-inc)/1000:
                print('processed %d'%count)
                if last_mean_list is None:
                    last_mean_list = [np.zeros_like(mean, dtype=np.float64) for mean in mean_list]
                flag = True
                for mean, last_mean in zip(mean_list, last_mean_list):
                    cur_mean = mean/count
                    if not np.isclose(last_mean, cur_mean).all():
                        print np.max(np.abs(last_mean-cur_mean))
                        flag = False
                        last_mean[:] = cur_mean
                        break
                mean_dict = dict(zip(name_list, [(mean/count).astype(np.float32) for mean in mean_list]))
                np.savez(fname, **mean_dict)
                if flag:
                    break

        data_iter.reset()
        return mean_dict

class Mov3dStack(mx.io.DataIter):
    def __init__(self, path, data_shape, batch_size, scale,
                 mean_file=None, test_mode=False, output_depth=False, data_frames=1, flow_frames=1,
                 source=None, upsample=1, base_shape=None, stride=1, no_left0=False, right_whiten=False):
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.scale = scale
        self.test_mode = test_mode
        self.output_depth = output_depth
        self.data_frames = data_frames
        self.flow_frames = flow_frames
        self.upsample = upsample
        self.base_shape = base_shape
        self.stride = stride
        self.no_left0 = no_left0
        self.right_whiten = right_whiten

        self.fix_p = None

        self.env = lmdb.open(path, map_size=1<<40, max_dbs=5, readonly=True, readahead=False)
        self.ldb = self.env.open_db('l')
        if flow_frames > 0:
            self.fdb = self.env.open_db('flow')
        if output_depth:
            self.ddb = self.env.open_db('depth')
            self.margin = (scale[1] - scale[0])/2
        else:
            self.margin = 0
        self.rdb = self.env.open_db('r')

        self.cur = 0
        with self.env.begin() as txn:
            if source:
                self.idx = [int(i) for i in txn.get(source).split(',')]
            elif self.test_mode:
                self.idx = [int(i) for i in txn.get('shuffled_test_idx').split(',')]
            else:
                self.idx = [int(i) for i in txn.get('shuffled_test_idx').split(',')]
            if self.upsample > 1:
                self.caps = [cv2.VideoCapture('data/raw/%s.mkv'%p) for p in txn.get('prefix_list').split(',')]

        self.provide_data = []
        if data_frames > 0:
            self.provide_data.append(('left', (batch_size, 3*data_frames, data_shape[1], data_shape[0])))
        if flow_frames > 0:
            self.provide_data.append(('flow', (batch_size, 2*flow_frames, data_shape[1], data_shape[0])))
        if not no_left0:
            self.provide_data.append(('left0', (batch_size, 3, data_shape[1]*upsample, data_shape[0]*upsample)))

        self.provide_label = [('l1_label', (batch_size, 3, data_shape[1]*upsample, data_shape[0]*upsample))]
        if self.output_depth:
            self.provide_label = [('softmax_label', (batch_size, data_shape[1]*data_shape[0]))]

        self.left_mean = np.zeros((3, data_shape[1], data_shape[0]))
        self.right_mean = np.zeros((3, data_shape[1], data_shape[0]))

        self.left_mean_nd = mx.nd.array(self.left_mean)
        self.left_mean_nd_1 = self.left_mean_nd.reshape((1,)+self.left_mean_nd.shape)
        self.right_mean_nd = mx.nd.array(self.right_mean)

        if flow_frames > 0:
            self.flow_mean = np.zeros((2, data_shape[1], data_shape[0]))
            self.flow_mean_nd = mx.nd.array(self.flow_mean)

        if mean_file is None:
            mean_file = path+'/mean.npz'
        mean_dict = load_mean(mean_file, self, label_mean=True)
        self.left_mean = mean_dict['left']
        self.right_mean = mean_dict['l1_label']
        if flow_frames > 0:
            self.flow_mean = mean_dict['flow']
            self.flow_mean_nd = mx.nd.array(self.flow_mean)

        self.left_mean_nd = mx.nd.array(self.left_mean)
        self.left_mean_nd_1 = self.left_mean_nd.reshape((1,)+self.left_mean_nd.shape)
        self.right_mean_nd = mx.nd.array(self.right_mean)

    def reset(self):
        logging.info("Mov3dStack.reset at %d"%self.cur)
        self.cur = 0
        if not self.test_mode:
            random.shuffle(self.idx)

    def seek(self, n_iter):
        self.cur = (n_iter*self.batch_size)%len(self.idx)

    def next(self):
        from parse import split

        ndleft = mx.nd.zeros((self.batch_size*self.data_frames, 3, self.data_shape[1], self.data_shape[0]))
        if self.upsample > 1:
            left0 = np.zeros((self.batch_size, self.data_shape[1]*self.upsample, self.data_shape[0]*self.upsample, 3), dtype=np.float32)
        else:
            ndleft0 = mx.nd.zeros((self.batch_size, 3, self.data_shape[1], self.data_shape[0]))
        if self.flow_frames > 0:
            ndflow = mx.nd.zeros((self.batch_size*self.flow_frames, 2, self.data_shape[1], self.data_shape[0]))
        right = np.zeros((self.batch_size, self.data_shape[1]*self.upsample, self.data_shape[0]*self.upsample, 3), dtype=np.float32)
        if self.output_depth:
            depth = np.zeros((self.batch_size, self.data_shape[1]*self.data_shape[0]), dtype=np.float32)


        with self.env.begin() as txn:    
            for i in range(self.batch_size):
                if self.cur >= len(self.idx):
                    i -= 1
                    break
                idx = self.idx[self.cur]
                if self.upsample > 1:
                    nidx = int(idx)
                    mov = nidx/1000000
                    nframe = nidx%1000000
                    nframe = nframe/10000*3*24*60 + nframe%10000
                    if self.caps[mov].get(cv2.CAP_PROP_POS_FRAMES) != nframe:
                        print 'seek', nframe
                        self.caps[mov].set(cv2.CAP_PROP_POS_FRAMES, nframe)
                    ret, frame = self.caps[mov].read()
                    assert ret
                    margin = (frame.shape[0] - 800)/2
                    lframe, rframe = split(frame, reshape=self.base_shape, vert=True, clip=(0, margin, 960, margin+800))

                p = self.fix_p
                if self.output_depth:
                    sd = txn.get('%09d'%idx, db=self.ddb)
                    assert sd is not None
                    _, dimg = mx.recordio.unpack_img(sd, -1)
                    dimg, p = crop_img(dimg, p, self.data_shape, self.margin, test=self.test_mode)
                    depth[i] = dimg.flat

                if self.upsample > 1:
                    rimg, p = crop_img(rframe, p, (self.data_shape[0]*self.upsample, self.data_shape[1]*self.upsample), 0, test=self.test_mode, grid=self.upsample)
                    right[i] = rimg
                else:
                    sr = txn.get('%09d'%idx, db=self.rdb)
                    assert sr is not None
                    _, rimg = mx.recordio.unpack_img(sr, 1)
                    rimg, p = crop_img(rimg, p, self.data_shape, 0, test=self.test_mode)
                    right[i] = rimg

                for j in range(max(1, self.data_frames)):
                    sl = txn.get('%09d'%(idx+(j-self.data_frames/2)*self.stride), db=self.ldb)
                    if sl is None:
                        pass
                    else:
                        _, s = mx.recordio.unpack(sl)
                        mx.nd.imdecode(s, clip_rect=(p[0], p[1], p[0] + self.data_shape[0], p[1] + self.data_shape[1]),
                                       out=ndleft, index=i*self.data_frames+j, channels=3, mean=self.left_mean_nd)

                if self.upsample > 1:
                    limg, p = crop_img(lframe, p, (self.data_shape[0]*self.upsample, self.data_shape[1]*self.upsample), 0, test=self.test_mode, grid=self.upsample)
                    left0[i] = limg
                else:
                    start = i*max(1, self.data_frames)+max(1, self.data_frames)/2
                    ndleft0[i:(i+1)] = ndleft[start:(start+1)] + self.left_mean_nd_1

                for j in range(self.flow_frames):
                    sf = txn.get('%09d'%(idx+(j-self.flow_frames/2)*self.stride), db=self.fdb)
                    if sf is None:
                        pass
                    else:
                        _, s = mx.recordio.unpack(sf)
                        mx.nd.imdecode(s, clip_rect=(p[0], p[1], p[0] + self.data_shape[0], p[1] + self.data_shape[1]),
                                       out=ndflow, index=i*self.flow_frames+j, channels=2, mean=self.flow_mean_nd)
                self.cur += 1

        data = []
        if self.data_frames > 0:
            ndleft = ndleft.reshape((self.batch_size, self.data_frames*3, self.data_shape[1], self.data_shape[0]))
            data.append(ndleft)
        if self.flow_frames > 0:
            ndflow = ndflow.reshape((self.batch_size, self.flow_frames*2, self.data_shape[1], self.data_shape[0]))
            data.append(ndflow)
        if self.upsample > 1:
            data.append(mx.nd.array(left0.transpose((0, 3, 1, 2))))
        elif not self.no_left0:
            data.append(ndleft0)
        right = right.transpose((0, 3, 1, 2))
        if self.right_whiten:
            right -= self.right_mean

        i += 1
        pad = self.batch_size - i
        if pad:
            raise StopIteration
        if self.output_depth:
            return mx.io.DataBatch(data, [mx.nd.array(right), mx.nd.array(depth)], pad, None)
        else:
            return mx.io.DataBatch(data, [mx.nd.array(right)], pad, None)

def load_vgg(data_frames=1, flow_frames=1, two_stream=False):
    vgg16 = {name: arr for name, arr in mx.nd.load('vgg16-0001.params').items() if name.startswith('arg:conv')}
    conv1_weight = vgg16['arg:conv1_1_weight']

    new_shape = list(conv1_weight.shape)
    new_shape[1] = 3*data_frames + 2*flow_frames
    new_conv1_weight = np.zeros(new_shape, dtype=np.float32)
    for i in range(data_frames):
        if i != data_frames/2:
            continue
        new_conv1_weight[:, i*3:(i+1)*3, :, :] = conv1_weight.asnumpy()
    for i in range(2*flow_frames):
        new_conv1_weight[:, data_frames*3+i, :, :] = conv1_weight.asnumpy().mean(axis=1)

    vgg16['arg:conv1_1_weight'] = mx.nd.array(new_conv1_weight)
    return vgg16

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Put recordio files into lmdb')
    parser.add_argument('path', type=str, help='Path to folder containing recordio files')
    parser.add_argument('fdb', type=str, help='Path to output lmdb')
    args = parser.parse_args()

    path = args.path
    fdb = args.fdb
    suffix_list = ['l', 'r']
    # uncomment if working with optical flow or depth data.
    # suffix_list = ['depth', 'flow', 'l', 'r']

    pset = set()
    for fname in os.listdir(path):
        if fname.endswith('.rec'):
            pset.add(fname.split('_')[0])
    print pset
    for fname in pset:
        addto_db(fdb, path+fname, suffix_list)
    
    # show_db(fdb, suffix_list)
    shuffle(fdb)
    make_idx(fdb)