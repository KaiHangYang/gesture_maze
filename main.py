import numpy as np
import cv2

from utils import forward
from utils import utils
from utils import display_utils
from utils import bbtracker

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    model = forward.mCPMHandForward("./tf_models/cpm_hand_tf/cpm_hand")
    tracker = bbtracker.BBTracker(wndWidth = 368, wndHeight = 368, pad_scale = 2)


    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    prev_joints = np.zeros([model.num_of_joints, 2], dtype=np.float32)
    prev_joints[0] = [cap_width - 1, cap_height - 1]
    prev_beliefs = np.zeros([model.num_of_joints], dtype=np.float32)

    while True:
        _, raw_img = cap.read()

        img = tracker.track(prev_joints, prev_beliefs, raw_img.copy())

        points, beliefs, _ = model.predict(img)

        global_joints = tracker.get_global_joints(points)

        raw_img = display_utils.drawLines(raw_img, global_joints)

        cv2.imshow("test2", raw_img)
        cv2.waitKey(3)

        prev_joints = points.copy()
        prev_beliefs = beliefs.copy()
