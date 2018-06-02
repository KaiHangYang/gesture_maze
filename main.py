import numpy as np
import cv2

from utils import forward
from utils import utils
from utils import display_utils

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    model = forward.mCPMHandForward("./tf_models/cpm_hand_tf/cpm_hand")

    while True:
        _, img = cap.read()

        img = utils.pad_image(img, 368)

        points, heatmaps = model.predict(img)

        # print(points)

        for i in range(5):
            cv2.imshow("hm%d" % i, display_utils.visualizeHeatmap(heatmaps[:, :, i], min_val=0, mid_val=0.5, max_val=1.0))
            cv2.waitKey(3)

        img = display_utils.drawLines(img, points)

        cv2.imshow("test", img)
        cv2.waitKey(3)
