import numpy as np
import tensorflow as tf
import os
import sys
import cv2

sys.path.append("../")

from nets import network
import utils
import settings
import display_utils

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
        self.net_model = network.CPM_Model(self.num_of_stage, self.num_of_joints + 1)
        self.input_image = tf.placeholder(dtype=tf.float32, shape=[None, self.input_img_size, self.input_img_size, 3], name="input_image")
        self.input_center_map = tf.placeholder(dtype=tf.float32, shape=[None, self.input_img_size, self.input_img_size, 1], name="input_center_map")
        self.net_model.build_model(self.input_image, self.input_center_map, self.batch_size)
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings.gpu_memory_fraction)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, self.model_path)

    def __del__(self):
        self.sess.close()

    # The input image is cropped to 368 x 368
    def preprocess(self, image):
        image = image[np.newaxis]
        image = image / 256.0 - 0.5
        center_map = utils.make_gaussian(self.input_img_size, self.input_img_size / 2, self.input_img_size / 2, self.center_map_radius)
        center_map = center_map[np.newaxis, :, :, np.newaxis]

        return image, center_map

    def extract(self, heatmaps):
        result_joints = np.zeros([self.num_of_joints, 2], dtype=np.float32)
        result_beliefs = np.zeros([self.num_of_joints], dtype=np.float32)

        for hm_num in range(heatmaps.shape[2]):
            point = np.unravel_index(np.argmax(heatmaps[:, :, hm_num]), [self.input_img_size / 8, self.input_img_size / 8])
            result_joints[hm_num] = (point[1] * 8, point[0] * 8)
            result_beliefs[hm_num] = heatmaps[point[0], point[1], hm_num]

        return result_joints, result_beliefs

    def visualizeHeatmaps(self, heatmaps):
        h_width = heatmaps.shape[1]
        h_height = heatmaps.shape[0]
        h_num = heatmaps.shape[2]

        pad_size = 4
        block_size = h_width + 2 * pad_size
        result_size = 5 * block_size
        result_data = -np.ones([result_size, result_size], dtype=np.float32)

        for i in range(self.num_of_joints):
            cur_row = i / 5
            cur_col = i % 5
            result_data[(pad_size + cur_row * block_size):(pad_size + cur_row * block_size + h_height), (pad_size + cur_col * block_size):(pad_size + cur_col * block_size + h_height)] = heatmaps[:, :, i]

        mask = (result_data == -1)

        result_image = display_utils.visualizeHeatmap(result_data, min_val=0, mid_val=0.5, max_val=1.0)

        result_image[mask] = 0

        return result_image



    def predict(self, image):
        input_image, input_center_map = self.preprocess(image)

        result_heatmaps = self.sess.run([self.net_model.current_heatmap],
                                         feed_dict={
                                             self.input_image: input_image,
                                             self.input_center_map: input_center_map
                                        })

        heatmaps = result_heatmaps[0][0, : , :, 0: self.num_of_joints]
        # cv2.imshow("visual_heatmaps", self.visualizeHeatmaps(heatmaps))
        # cv2.waitKey(3)
        result_points, result_beliefs = self.extract(heatmaps)
        return result_points, result_beliefs, heatmaps

