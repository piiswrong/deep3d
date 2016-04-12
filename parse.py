import sys, os
import mxnet as mx
import numpy as np
import cv2
from mxnet.recordio import *
import argparse


def split(frame, reshape=(256,144), vert=True, clip=None):
    if vert == True:
        lframe = frame[:, :frame.shape[1]/2]
        rframe = frame[:, frame.shape[1]/2:]
    elif vert == False:
        lframe = frame[:frame.shape[0]/2, :]
        rframe = frame[frame.shape[0]/2:, :]
    else:
        lframe = frame
        rframe = frame
    if clip is not None:
        lframe = lframe[clip[1]:clip[3], clip[0]:clip[2]]
        rframe = rframe[clip[1]:clip[3], clip[0]:clip[2]]
    if reshape is not None:
        lframe = cv2.resize(lframe, reshape)
        rframe = cv2.resize(rframe, reshape)
    return lframe, rframe

def anaglyph(lframe, rframe):
    frame = np.zeros_like(lframe)
    frame[:,:,:2] = rframe[:,:,:2]
    frame[:,:,2:] = lframe[:,:,2:]
    return frame

def sbs(lframe, rframe):
    frame = np.zeros_like(lframe)
    sep = lframe.shape[1]/2
    frame[:,sep:] = cv2.resize(rframe, (sep, rframe.shape[0]))
    frame[:,:sep] = cv2.resize(lframe, (sep, lframe.shape[0]))
    return frame

class Stereo(object):
    def __init__(self):
        self.min = 0
        self.scale = 16
        self.stereo = None

    def sgbm_create(self, minDisparity, numDisparities, blockSize, mode=cv2.StereoSGBM_MODE_HH):
        self.min = minDisparity - 1
        self.scale = 16
        self.stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                                            numDisparities=numDisparities,
                                            blockSize=blockSize,
                                            P1=8*3*5**2,
                                            P2=32*3*5**2,
                                            disp12MaxDiff=1,
                                            uniquenessRatio=10,
                                            speckleWindowSize=100,
                                            speckleRange=32,
                                            mode=mode)

    def compute(self, lframe, rframe):
        lframe = cv2.cvtColor(lframe, cv2.COLOR_BGR2GRAY)
        rframe = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)
        dframe = self.stereo.compute(lframe, rframe)
        dframe = (dframe/self.scale-self.min).astype(np.uint8)
        return dframe

def flow(prev_frame, cur_frame, tvl1=cv2.createOptFlow_DualTVL1()):
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    return tvl1.calc(prev_frame, cur_frame, None)

def make_frame_data(input_path, output_path, reshape=(256,144), vert=True, clip=None):
    lrecord = MXRecordIO(output_path+'_l.rec', 'w')
    rrecord = MXRecordIO(output_path+'_r.rec', 'w')
    cap = cv2.VideoCapture(input_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if i % 5000 == 0:
            print 'frame ', i
        if not ret:
            break
        lframe, rframe = split(frame, reshape=reshape, vert=vert, clip=clip)
        lrecord.write(pack_img((0,0,i,0), lframe, quality=80))
        rrecord.write(pack_img((0,0,i,0), rframe, quality=80))
        i += 1
    cap.release()
    del lrecord
    del rrecord

def make_flow_data(input_path, output_path):
    irecord = MXRecordIO(input_path, 'r')
    orecord = MXRecordIO(output_path, 'w')
    pframe = None
    i = 0
    while True:
        s = irecord.read()
        if s is None:
            break
        header, frame = unpack_img(s)
        if pframe is None:
            pframe = frame
        f = flow(pframe, frame)
        f = np.clip((f+40)*255.0/80.0, 0.0, 255.0).astype(dtype=np.uint8)
        f = np.concatenate((f, np.zeros((f.shape[0], f.shape[1], 1))), axis=2)
        pframe = frame
        orecord.write(pack_img(header, f, quality=80))
        if i%10 == 0:
            print 'flow ', i
        i += 1
    del orecord

def make_depth_data(prefix):
    lrecord = MXRecordIO(prefix+'_l.rec', 'r')
    rrecord = MXRecordIO(prefix+'_r.rec', 'r')
    orecord = MXRecordIO(prefix+'_depth.rec', 'w')
    stereo = Stereo()
    stereo.sgbm_create(minDisparity=-15, numDisparities=32, blockSize=16)

    i = 0
    while True:
        sl = lrecord.read()
        sr = rrecord.read()
        if sl is None or sr is None:
            break
        header, lframe = unpack_img(sl, 1)
        header, rframe = unpack_img(sr, 1)
        d = stereo.compute(rframe, lframe)
        buf = pack_img(header, d, quality=3, img_fmt='.png')
        orecord.write(buf)
        if i%100 == 0:
            print 'depth ', i
        i += 1

def get_clip_rect(fname, vert=True):
    assert vert
    cap = cv2.VideoCapture(fname)
    for i in range(24*60*2):
        #get rid of logos
        assert cap.isOpened()
        assert cap.read()[0]
    shape = cap.read()[1].shape
    print 'original shape: ', shape
    assert shape[1] == 1920
    assert shape[0] >= 800
    acc = np.zeros(shape, dtype=np.float64)
    for i in range(24*60):
        ret, frame = cap.read()
        assert ret
        acc += frame
    acc /= 24*60
    y0 = 0
    while acc[y0].mean() < 2:
        y0 += 1
    y1 = shape[0]-1
    while acc[y1].mean() < 2:
        y1 -= 1
    y1 += 1
    print 'clip: ', (y1-y0, shape[1])
    diff = (y1-y0)-800
    half = diff/2
    y0 += half
    y0 = max(y0, 0)
    diff = (y1-y0)-800
    y1 -= diff
    y1 = min(y1, shape[0])
    cap.release()
    return (0, y0, 1920/2, y1)


def process_movie(fname, prefix, reshape=(432,180), vert=True):
    print 'processing video '+fname+' with prefix '+prefix

    rect = get_clip_rect(fname, vert)
    print 'clipping to ', rect
    make_frame_data(fname, prefix, reshape=reshape, vert=vert, clip=rect)
    # uncomment if working with depth/flow data
    # make_flow_data(prefix+'_l.rec', prefix+'_flow.rec')
    # make_depth_data(prefix)

def process_movie2d(fname, prefix, reshape=(432,180)):
    print 'processing 2d video '+fname+' with prefix '+prefix

    rect = list(get_clip_rect(fname, True))
    rect[2] = rect[0] + (rect[2]-rect[0])*2
    print 'clipping to ', rect
    make_frame_data(fname, prefix, reshape=reshape, vert=None, clip=rect)
    # uncomment if working with depth/flow data
    # make_flow_data(prefix+'_l.rec', prefix+'_flow.rec')
    # make_depth_data(prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process movies into recordio format')
    parser.add_argument('path', type=str, help='Path to movie file')
    parser.add_argument('prefix', type=str, help='Prefix for recordio file')
    parser.add_argument('--sbs3d', type=bool, default=True, help='Whether this is a side by side 3d video or a plain 2d video')
    args = parser.parse_args()

    if args.sbs3d:
        process_movie(args.path, args.prefix)
    else:
        process_movie2d(args.path, args.prefix)
