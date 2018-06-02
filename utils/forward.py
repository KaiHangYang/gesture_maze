import numpy as np
import tensorflow as tf
import os
import sys

sys.path.append("../")

from nets import network
from utils import utils

class mCPMHandForward():
    def __init__(self, model_path):
        # Use 6 stage as default
        self.num_of_stage = 6
        self.batch_size = 1
        self.input_img_size = 368
        self.num_of_joints = 21
        self.center_map_radius = 21
        self.model_path = model_path

        if not os.path.isfile(model_path + ".meta"):
            print("The model path is not valid!")
            quit()

        #### Set the parameters of the network
        self.net_model = network.CPM_Model(self.num_of_stage, self.num_of_joints)
        self.input_image = tf.placeholder(dtype=tf.float32, shape=[None, self.input_img_size, self.input_img_size, 3], name="input_image")
        self.input_center_map = tf.placeholder(dtype=tf.float32, shape=[None, self.input_img_size, self.input_img_size, 1], name="input_center_map")
        self.net_model.build_model(self.input_image, self.input_center_map, self.batch_size)
        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, self.model_path)

    def __del__(self):
        self.sess.close()

    # The input image is cropped to 368 x 368
    def preprocess(self, image):
        image = image[np.newaxis]
        image = image / 256.0 - 0.5
        center_map = utils.make_gaussian(self.input_img_size, self.input_img_size / 2, self.input_img_size / 2, self.center_map_radius)
        center_map = center_map[np.newaxis]

        return image, center_map

    def extract(self, heatmaps):
        result_joints = np.zeros([self.num_of_joints, 2], dtype=np.float32)

    def predict(self, image):
        input_image, input_center_map = self.preprocess(image)

        result_heatmaps = self.sess.run([self.net_model.current_heatmap],
                                         feed_dict={
                                             self.input_image: input_image,
                                             self.input_center_map: input_center_map
                                        })

