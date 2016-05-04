import argparse
import mxnet as mx
import numpy as np
import os
import json
import urllib
import cv2
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
from images2gif import writeGif
import logging
import time
logging.basicConfig(level=logging.DEBUG)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to base image")
ap.add_argument("-j", "--json",default=False,  help="upload images")
ap.add_argument("-o", "--output_folder", default="./", help="output folder")
args = ap.parse_args()
param_path = '/tmp/deep3d-0050.params'
junk = subprocess.check_output(["cp", 'deep3d-symbol.json', '/tmp/'])
if not os.path.exists(param_path):
      urllib.urlretrieve('http://homes.cs.washington.edu/~jxie/download/deep3d-0050.params', param_path)
      model = mx.model.FeedForward.load('/tmp/deep3d', 50, mx.gpu(0))

model = mx.model.FeedForward.load('/tmp/deep3d', 50, mx.gpu(0))

shape = (384, 160)
img = cv2.imread(args.image)
raw_shape = (img.shape[1], img.shape[0])
img = cv2.resize(img, shape)
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.axis('off')
#plt.show()

X = img.astype(np.float32).transpose((2,0,1))
X = X.reshape((1,)+X.shape)
test_iter = mx.io.NDArrayIter({'left': X, 'left0':X})
Y = model.predict(test_iter)

right = np.clip(Y.squeeze().transpose((1,2,0)), 0, 255).astype(np.uint8)
right = Image.fromarray(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
left = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
if args.json != False:
    right_name = str(int(time.time())) + ".jpg"
    right_path = args.output_folder + "/" + right_name
    left_name = str(int(time.time())) + ".jpg"
    left_path = args.output_folder + "/" + left_name
    right.save(right_path,"JPEG") 
    left.save(left_path,"JPEG") 

    right_output = subprocess.check_output(["curl","-s", "--upload-file", right_path,("https://transfer.sh/"+right_name)]).strip()
    left_output = subprocess.check_output(["curl", "-s","--upload-file", left_path,("https://transfer.sh/"+left_name)]).strip()
    data = {'right':right_output,'left':left_output}
    output_file = args.output_folder + "/"+ str(int(time.time())) + ".json"
    print(data)
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile)
else:
    put_file = args.output_folder + "/"+ str(int(time.time())) + ".gif"
    writeGif(output_file, [left, right], duration=0.08)
print(output_file)
