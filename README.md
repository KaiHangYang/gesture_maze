手势控制迷宫游戏项目配置
=====

组员：　杨凯航(21721075)，关贞贞(21721073)，陆攀(21721074)

环境配置
-----

+ **声明**：目前我只在 *Linux(Ubuntu 16.04)* 下做过测试，其他系统不保证能够正常运行，不过因为我使用的开发语言为 *Python 2.7.12* 所以理论上来说，在 *Windows* 或者 *MacOS* 上，只要配置正确，应该能够正确运行。**非常重要的一点，请确保主机拥有4GB及以上显存的NVIDIA显卡**，我在GTX 980和GTX 1080Ti上都运行过都能够实时的运行。并且，电脑上还需要安装opencv的VideoCapture可以读取的摄像头。
+ 我所使用的开发环境和语言： *Linux(Ubuntu16.04)*，使用的开发语言为 *Python*版本为**2.7.12**。
+ 首先配置 *Python* 运行环境：
    + 假如电脑上有Python 2.7的环境，就不需要额外的配置，否则，需要去Python官网，下载并且按照官网的安装知道配置安装，这里我不加赘述。
    + Python2.7安装完毕之后，打开终端(Windows为cmd)，接下来要安装一些运行该项目需要的库:
        ```shell
        sudo pip install numpy
        sudo pip install panda3d
        sudo pip install tensorflow-gpu==1.4.0
        sudo pip install opencv-python
        ```
        安装完这些库之后，还需要配置tensorflow-gpu的环境，需要安装对应系统的**CUDA 8**以及**cudnn 6**，详细过程这里不多加介绍，网上资源非常的丰富，并且准确。
    + 这样运行本项目所需要的Python环境配置就完成了。
+ 然后配置项目运行环境
    + 下载训练好的网络模型。链接: https://pan.baidu.com/s/1Xcvi5QE5iEC6Pr1aK0Bj_A 密码: zhuf
    + 然后将下载好的模型解压缩，整个文件夹放在项目根目录下，最终整个项目的目录为:
        ```shell
        .
        ├── main.py
        ├── models
        │   ├── ball.egg.pz
        │   ├── iron05.jpg
        │   ├── limba.jpg
        │   ├── maze01.jpg
        │   ├── maze11.egg
        │   └── maze.egg.pz
        ├── nets
        │   ├── __init__.py
        │   └── network.py
        ├── README.md
        ├── tf_models
        │   └── cpm_hand_tf
        │       ├── cpm_hand.data-00000-of-00001
        │       ├── cpm_hand.index
        │       └── cpm_hand.meta
        └── utils
            ├── bbtracker.py
            ├── display_utils.py
            ├── forward.py
            ├── gesture_control.py
            ├── __init__.py
            ├── joints_resolve.py
            ├── settings.py
            └── utils.py

        ```
        有可能会多出一些*.pyc文件，不用在意。
    + 最后根据你机器所拥有的显卡显存，在settings.py中修改比例，使`你的显存总量 × 比例 >= 3GB`，eg: 我的现存总量为11GB所以我设置的比例为`0.3`。
    + 迷宫配置，我们预备了两个可用的迷宫模型，只需要在settings.py中修改maze_id变量即可，有两个选项一个是"maze"另一个是"maze11"。
+ 项目运行
    + 执行项目只需要在项目根目录下打开终端，然后执行`python main.py`即可。
    + 程序如何使用：(需要注意的是，在运行之前，必须完成以上配置，并且有可用的摄像头安装在电脑上)。运行程序之后，使一只手出现在摄像头拍出的图像区域内，并且区域内有且只能有一只手，如果没有手的话，游戏不会开始。五指张开的时候表示游戏暂停，握拳状态的时候，表示游戏进行。游戏进行的时候，在摄像头可视的范围内移动握拳的手，即可移动迷宫进行游戏，五指张开即可暂停。
+ 以上即为项目配置过程和运行流程。如果有什么问题，请老师联系我。
> 我的联系方式为: QQ: 907987244。加好友问题请填： 你是谁？ 人机交互。我是谁？杨凯航。
