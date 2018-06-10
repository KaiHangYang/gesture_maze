import numpy as np
import cv2

import forward
import utils
import display_utils
import bbtracker
import joints_resolve

class GestureControler():
    def __init__(self, camera_device=0):
        self.cap = cv2.VideoCapture(camera_device)
        if not self.cap.isOpened():
            print("The camera is not settled correctly!")
            quit()

        self.cur_pose = -1
        self.model = forward.mCPMHandForward("./tf_models/cpm_hand_tf/cpm_hand")
        self.tracker = bbtracker.BBTracker(wndWidth = 368, wndHeight = 368, pad_scale = 2)

        self.cap_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cap_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.prev_joints = np.zeros([self.model.num_of_joints, 2], dtype=np.float32)
        self.prev_joints[0] = [self.cap_width - 1, self.cap_height - 1]
        self.prev_beliefs = np.zeros([self.model.num_of_joints], dtype=np.float32)

        self.prev_global_joints = np.zeros([self.model.num_of_joints, 2], dtype=np.float32)
        self.prev_global_joints[0] = [self.cap_width - 1, self.cap_height - 1]

    def track(self):
        _, raw_img = self.cap.read()
        raw_img = cv2.flip(raw_img, 1)

        img = self.tracker.track(self.prev_joints, self.prev_beliefs, raw_img.copy())

        points, beliefs, _ = self.model.predict(img)
        self.prev_joints = points.copy()
        self.prev_beliefs = beliefs.copy()

        global_joints = self.tracker.get_global_joints(points)
        self.prev_global_joints = global_joints
        mean_beliefs = np.mean(beliefs)

        if mean_beliefs >= 0.5:
            self.cur_pose = joints_resolve.resolve(global_joints)
        else:
            self.cur_pose = -1

        raw_img = display_utils.drawLines(raw_img, global_joints)
        cv2.imshow("test2", raw_img)
        cv2.waitKey(1)

    def isMoving(self):
        if self.cur_pose == 1:
            return True
        else:
            return False
    def getQuit(self):
        return self.cur_pose == 2
    def getReset(self):
        return self.cur_pose == 0

    def getPos(self):
        pos_x = np.mean(self.prev_global_joints[:, 0])
        pos_y = np.mean(self.prev_global_joints[:, 1])

        pos_x = 2.0 * pos_x / self.cap_width - 1.0
        pos_y = (2.0 * pos_y / self.cap_height - 1.0) * -1

        return [pos_x, pos_y]
