import tensorflow as tf


class CPM_Model(object):
    def __init__(self, stages, joints):
        self.stages = stages
        self.stage_heatmap = []
        self.stage_loss = [0] * stages
        self.total_loss = 0
        self.input_image = None
        self.center_map = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.joints = joints
        self.batch_size = 0

    def build_model(self, input_image, center_map, batch_size):
        self.batch_size = batch_size
        self.input_image = input_image
        self.center_map = center_map
        with tf.variable_scope('pooled_center_map'):
            self.center_map = tf.layers.average_pooling2d(inputs=self.center_map,
                                                          pool_size=[9, 9],
                                                          strides=[8, 8],
                                                          padding='SAME',
                                                          name='center_map')
        with tf.variable_scope('sub_stages'):
            sub_conv1 = tf.layers.conv2d(inputs=input_image,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv1')
            sub_conv2 = tf.layers.conv2d(inputs=sub_conv1,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv2')
            sub_pool1 = tf.layers.max_pooling2d(inputs=sub_conv2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='SAME',
                                                name='sub_pool1')
            sub_conv3 = tf.layers.conv2d(inputs=sub_pool1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv3')
            sub_conv4 = tf.layers.conv2d(inputs=sub_conv3,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv4')
            sub_pool2 = tf.layers.max_pooling2d(inputs=sub_conv4,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='SAME',
                                                name='sub_pool2')
            sub_conv5 = tf.layers.conv2d(inputs=sub_pool2,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv5')
            sub_conv6 = tf.layers.conv2d(inputs=sub_conv5,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv6')
            sub_conv7 = tf.layers.conv2d(inputs=sub_conv6,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv7')
            sub_conv8 = tf.layers.conv2d(inputs=sub_conv7,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv8')
            sub_pool3 = tf.layers.max_pooling2d(inputs=sub_conv8,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='SAME',
                                                name='sub_pool3')
            sub_conv9 = tf.layers.conv2d(inputs=sub_pool3,
                                         filters=512,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv9')
            sub_conv10 = tf.layers.conv2d(inputs=sub_conv9,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='SAME',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv10')
            sub_conv11 = tf.layers.conv2d(inputs=sub_conv10,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='SAME',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv11')
            sub_conv12 = tf.layers.conv2d(inputs=sub_conv11,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='SAME',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv12')
            sub_conv13 = tf.layers.conv2d(inputs=sub_conv12,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='SAME',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv13')
            sub_conv14 = tf.layers.conv2d(inputs=sub_conv13,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='SAME',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv14')
            self.sub_stage_img_feature = tf.layers.conv2d(inputs=sub_conv14,
                                                          filters=128,
                                                          kernel_size=[3, 3],
                                                          strides=[1, 1],
                                                          padding='SAME',
                                                          activation=tf.nn.relu,
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                          name='sub_stage_img_feature')

        with tf.variable_scope('stage_1'):
            conv1 = tf.layers.conv2d(inputs=self.sub_stage_img_feature,
                                     filters=512,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='SAME',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv1')
            self.stage_heatmap.append(tf.layers.conv2d(inputs=conv1,
                                                       filters=self.joints,
                                                       kernel_size=[1, 1],
                                                       strides=[1, 1],
                                                       padding='SAME',
                                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                       name='stage_heatmap'))
        for stage in range(2, self.stages + 1):
            self._middle_conv(stage)

    def _middle_conv(self, stage):
        with tf.variable_scope('stage_' + str(stage)):
            self.current_featuremap = tf.concat([self.stage_heatmap[stage - 2],
                                                 self.sub_stage_img_feature,
                                                 # self.center_map],
                                                 ],
                                                axis=3)
            mid_conv1 = tf.layers.conv2d(inputs=self.current_featuremap,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv1')
            mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv2')
            mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv3')
            mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv4')
            mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv5')
            mid_conv6 = tf.layers.conv2d(inputs=mid_conv5,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv6')
            self.current_heatmap = tf.layers.conv2d(inputs=mid_conv6,
                                                    filters=self.joints,
                                                    kernel_size=[1, 1],
                                                    strides=[1, 1],
                                                    padding='SAME',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='mid_conv7')
            self.stage_heatmap.append(self.current_heatmap)

    def build_loss(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step):
        self.gt_heatmap = gt_heatmap
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_step)
            tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer='Adam')
        self.merged_summary = tf.summary.merge_all()

