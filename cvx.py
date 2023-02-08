# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/azure_kinect_mkv_reader.py

import argparse
import open3d as o3d
import os
import json
import sys
import cv2
import cv2
import numpy as np
import pickle
from PIL import Image
from skimage.transform import resize

rad = 0
Train_Images = []
Test_Images = []

Train_Depth = []
Test_Depth = []

Train_Dict = {}
Test_Dict = {}

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img


def scale_image(img, factor=1):
    return cv2.resize(img, (int(img.shape[1] * factor), int(img.shape[0] * factor)))


pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(pwd, '..'))
from initialize_config import initialize_config

innie = r"C:\Users\Admin\Downloads\pytorch_ipynb\DataSet"
outie = r"C:\Users\Admin\Downloads\pytorch_ipynb\ecj_1202_v3.mkv"


class ReaderWithCallback:

    def __init__(self, input, output):
        self.flag_exit = False
        self.flag_play = True
        self.input = input
        self.output = output

        self.reader = o3d.io.AzureKinectMKVReader()
        self.reader.open(self.input)
        if not self.reader.is_opened():
            raise RuntimeError("Unable to open file {}".format(outie))

    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def space_callback(self, vis):
        if self.flag_play:
            print('Playback paused, press [SPACE] to continue.')
        else:
            print('Playback resumed, press [SPACE] to pause.')
        self.flag_play = not self.flag_play
        return False

    def run(self):
        global rad
        global Train_Images
        global Test_Images

        global Train_Depth
        global Test_Depth

        global Train_Dict
        global Test_Dict

        glfw_key_escape = 256
        glfw_key_space = 32
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.register_key_callback(glfw_key_space, self.space_callback)

        vis_geometry_added = False
        vis.create_window('reader', 1920, 540)

        print(
            "MKV reader initialized. Press [SPACE] to pause/start, [ESC] to exit."
        )

        if self.output is not None:
            abspath = os.path.abspath(self.output)
            metadata = self.reader.get_metadata()
            o3d.io.write_azure_kinect_mkv_metadata(
                '{}/intrinsic.json'.format(abspath), metadata)

            config = {
                'path_dataset': abspath,
                'path_intrinsic': '{}/intrinsic.json'.format(abspath)
            }
            initialize_config(config)
            with open('{}/config.json'.format(abspath), 'w') as f:
                json.dump(config, f, indent=4)

        idx = 0
        while not self.reader.is_eof() and not self.flag_exit:
            if self.flag_play:
                rgbd = self.reader.next_frame()
                if rgbd is None:
                    continue

                if not vis_geometry_added:
                    vis.add_geometry(rgbd)
                    vis_geometry_added = True

                if self.output is not None:
                    color_filename = '{0}/Train_Images/frame_{1:05d}.jpg'.format(
                        self.output, idx)
                    print('Writing to {}'.format(color_filename))
                    crop = np.array(rgbd.color)

                    crop = cv2.resize(crop[148:720, 354: 926, 0:3], (572, 572))
                    cropped = o3d.geometry.Image(crop)
                    if rad == 15:
                        color_filename = '{0}/Test_Images/frame_{1:05d}.jpg'.format(
                            self.output, idx)
                        pickle_out = open(r"C:\Users\Admin\Downloads\pytorch_ipynb\DataSet\index2.pkl", "wb")
                        frame = 'frame_{1:05d}.jpg'.format(
                            self.output, idx)
                        Test_Images += [frame]
                        pickle.dump(Test_Images, pickle_out)
                        pickle_out.close()


                    else:
                        pickle_out = open(r"C:\Users\Admin\Downloads\pytorch_ipynb\DataSet\index1.pkl", "wb")
                        frame = 'frame_{1:05d}.jpg'.format(
                            self.output, idx)
                        Train_Images += [frame]
                        pickle.dump(Train_Images, pickle_out)
                        pickle_out.close()


                    o3d.io.write_image(color_filename, cropped)


                    # 1280 by 720

                    depth_filename = '{0}/Train_Depth/frame_{1:05d}.png'.format(
                        self.output, idx)
                    print('Writing to {}'.format(depth_filename))

                    cropd = np.array(rgbd.depth)

                    cropd = cv2.resize(cropd[148:720, 354: 926], (572, 572))

                    cropd = o3d.geometry.Image(cropd)
                    if rad == 15:
                        depth_filename = '{0}/Test_Depth/frame_{1:05d}.png'.format(
                            self.output, idx)
                        pickle_out = open(r"C:\Users\Admin\Downloads\pytorch_ipynb\DataSet\index2.pkl", "wb")


                        frame2 = 'frame_{1:05d}.png'.format(
                            self.output, idx)
                        Test_Dict[frame] = frame2

                        #Test_Images += [frame]
                        pickle.dump(Test_Dict, pickle_out)
                        pickle_out.close()
                        rad = 0
                    else :
                        pickle_out = open(r"C:\Users\Admin\Downloads\pytorch_ipynb\DataSet\index1.pkl", "wb")
                        frame2 = 'frame_{1:05d}.png'.format(
                            self.output, idx)
                        Train_Dict[frame] = frame2


                        #frame = 'frame_{1:05d}.png'.format(
                        #    self.output, idx)
                        #Train_Images += [frame]
                        pickle.dump(Train_Dict, pickle_out)
                        pickle_out.close()
                        rad = rad + 1

                    o3d.io.write_image(depth_filename, cropd)
                    idx += 1

            try:
                vis.update_geometry(rgbd)
            except NameError:
                pass
            vis.poll_events()
            vis.update_renderer()

        self.reader.close()


if __name__ == '__main__':

    if innie is None:
        print('No output path, only play mkv')
    elif os.path.isdir(innie):
        print('Output path {} already existing, only play mkv'.format(
            innie))
        innie = None
    else:
        try:
            os.mkdir(innie)
            os.mkdir('{}/Train_Images'.format(innie))
            os.mkdir('{}/Train_Depth'.format(innie))
            os.mkdir('{}/Test_Images'.format(innie))
            os.mkdir('{}/Test_Depth'.format(innie))
        except (PermissionError, FileExistsError):
            print('Unable to mkdir {}, only play mkv'.format(innie))
            innie = None

    reader = ReaderWithCallback(outie, innie)
    reader.run()
