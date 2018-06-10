import numpy as np

def resolve(joints):
    # Simple resolve 

    # calculate the length of the bones
    fingers = np.zeros([5, 4], np.float32)

    # joint - root length
    fingers[0] = np.array([
        np.linalg.norm(joints[0] - joints[1]),
        np.linalg.norm(joints[0] - joints[2]),
        np.linalg.norm(joints[0] - joints[3]),
        np.linalg.norm(joints[0] - joints[4])
        ])

    fingers[1] = np.array([
        np.linalg.norm(joints[0] - joints[5]),
        np.linalg.norm(joints[0] - joints[6]),
        np.linalg.norm(joints[0] - joints[7]),
        np.linalg.norm(joints[0] - joints[8])
        ])

    fingers[2] = np.array([
        np.linalg.norm(joints[0] - joints[9]),
        np.linalg.norm(joints[0] - joints[10]),
        np.linalg.norm(joints[0] - joints[11]),
        np.linalg.norm(joints[0] - joints[12])
        ])

    fingers[3] = np.array([
        np.linalg.norm(joints[0] - joints[13]),
        np.linalg.norm(joints[0] - joints[14]),
        np.linalg.norm(joints[0] - joints[15]),
        np.linalg.norm(joints[0] - joints[16])
        ])

    fingers[4] = np.array([
        np.linalg.norm(joints[0] - joints[17]),
        np.linalg.norm(joints[0] - joints[18]),
        np.linalg.norm(joints[0] - joints[19]),
        np.linalg.norm(joints[0] - joints[20])
        ])

    _flags = np.array([0] * 5)

    for i in range(5):
        if fingers[i][3] < 1.1 * fingers[i][1]:
            _flags[i] = 1

    # Reset 0, Moving 1, Quit 2

    if (_flags[0:5] == np.array([0, 1, 1, 1, 0])).all():
        # print("Reset")
        return 0
    elif (_flags[1:5] == np.array([1, 1, 1, 1])).all():
        # print("Moving")
        return 1
    elif (_flags[1:5] == np.array([0, 0, 1, 1])).all():
        # print("Quit")
        return 2

