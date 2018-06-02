import numpy as np
import cv2

import utils

class BBTracker():
    def __init__(self, wndWidth = 368, wndHeight = 368, pad_scale = 1.2):
        self.pad_scale = pad_scale
        self.crop_box_size = 368
        self.wndWidth = wndWidth
        self.wndHeight = wndHeight
        self.center = np.asarray([wndWidth / 2.0, wndHeight / 2.0])
        self.box_size = max([wndWidth, wndHeight])
        self.scale = float(self.box_size) / self.crop_box_size
        self.offset = [(self.wndWidth - self.box_size) / 2.0, (self.wndHeight - self.box_size) / 2.0]
        self.initial_frame_num = 5
        self.frame_num = 0
        # self.bbx = [self.center[0] - self.box_size / 2.0, self.center[0] + self.box_size / 2.0, self.center[1] - self.box_size / 2.0, self.center[1] + self.box_size / 2.0]
        self.bbx = [0, 0, wndWidth, wndHeight]
        self.for_test = 0

        self.r_val = [0] * 4
        self.t_val = [0] * 4

    def track(self, points, points_belief, img):
        threshhold_min = max(img.shape[0], img.shape[1]) * 0.7
        threshhold_max = max(img.shape[0], img.shape[1])

        points = np.reshape(points, (-1, 2))

        points = self.get_global_joints(points)

        img_width = img.shape[1]
        img_height = img.shape[0]

        min_x = np.min(points[:, 0])
        min_y = np.min(points[:, 1])
        max_x = np.max(points[:, 0])
        max_y = np.max(points[:, 1])

        box_size = max([max_x - min_x, max_y - min_y]) * self.pad_scale

        if box_size > threshhold_max:
            box_size = threshhold_max
        elif box_size < threshhold_min:
            box_size = threshhold_min

        center = np.asarray([(max_x + min_x) / 2.0, (max_y + min_y) / 2.0])

        if self.frame_num > self.initial_frame_num:
            if pow((self.center[0] - center[0]) ** 2 + (self.center[1] - center[1]) ** 2, 0.5) > 10:
                box_size = int(self.box_size * 0.4 + box_size * 0.6)
                center = self.center * 0.4 + center * 0.6
            else:
                box_size = self.box_size
                center = self.center
        else:
            box_size = int(box_size)

        r_l = int(center[0] - box_size / 2.0)
        r_r = int(r_l + box_size)
        r_t = int(center[1] - box_size / 2.0)
        r_b = int(r_t + box_size)

        raw_l = 0 if r_l < 0 else r_l
        raw_t = 0 if r_t < 0 else r_t
        raw_r = img_width if r_r > img_width else r_r
        raw_b = img_height if r_b > img_height else r_b

        self.offset = [r_l, r_t]


        if not self.valid_beliefs(points_belief):
            # tracking failed
            self.bbx = [0, 0, self.wndWidth, self.wndHeight]
            self.frame_num = 0
            self.center = np.asarray([self.wndWidth / 2.0, self.wndHeight / 2.0])
            self.box_size = max([img.shape[0], img.shape[1]])
            self.scale = float(self.box_size) / self.crop_box_size
            self.offset = [(img.shape[1] - self.box_size) / 2.0, (img.shape[0] - self.box_size) / 2.0]
            img = utils.pad_image(img, self.wndHeight)
            return img
        else:
            if self.frame_num < 2*self.initial_frame_num:
                self.frame_num += 1
            # print(raw_t - r_t, r_b - raw_b, raw_l - r_l , r_r - raw_r)
            result = cv2.copyMakeBorder(img[raw_t:raw_b, raw_l:raw_r], top=raw_t - r_t, bottom=r_b - raw_b, left=raw_l - r_l, right=r_r - raw_r, borderType=cv2.BORDER_CONSTANT, value=[128, 128, 128])

            self.scale = float(result.shape[0]) / self.crop_box_size
            # print(self.scale)
            self.bbx = [raw_l, raw_t, raw_r, raw_b]
            self.box_size = box_size
            self.center = center
            if not result is None:
                result = cv2.resize(result, (self.wndWidth, self.wndHeight))
            return result

    def get_global_joints(self, local_joints):
        local_joints = local_joints.copy()
        local_joints *= self.scale
        local_joints[:, 0] += self.offset[0]
        local_joints[:, 1] += self.offset[1]

        return local_joints


    def valid_beliefs(self, belief):
        if self.frame_num < self.initial_frame_num:
            return True

        total_num = float(len(belief))
        valid_num = np.sum(np.uint8(belief > 0.15))
        # If the belief of half points is below 0.2, then loss tracking
        if valid_num / total_num < 0.65:
            return False
        else:
            return True
