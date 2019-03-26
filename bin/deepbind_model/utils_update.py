import os.path
from math import ceil as ceil

import numpy as np
import scipy.stats as stats
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import h5py
k_mer = 3
sum = [0, 5, 22, 87, 344, 1369, 5466, 21849, 87370, 349419]
ENSEMBLE_NUMBER = 6

class ClipInputStruct(object):
    def __init__(self, inf):
        (data_one_hot, labels,
         structures,
         total_cases, seq_length, self.start_pos, self.end_pos) = (inf["data_one_hot"], inf["labels"],
                                                                   inf["structures"], inf["total_cases"],
                                                                   inf["seq_length"], inf['start_pos'], inf['end_pos'])
        self.seq_length = self.end_pos-self.start_pos
        max_len = np.max(self.seq_length)
        data_array = np.zeros([data_one_hot.shape[0], max_len,9])
        data_array[:,:,4:] = 0.0
        structures = np.transpose(structures,[0,2,1])
        for i in range(data_one_hot.shape[0]):
            data_array[i,0:self.seq_length[i],:] = np.append(data_one_hot[i,self.start_pos[i]:self.end_pos[i],:],
                                      structures[i,self.start_pos[i]:self.end_pos[i],:], axis=1)
        self.data = data_array
        self.seq_len = max_len
        self.total_cases = self.data.shape[0]
        self.labels = labels


class NoStructInput(object):
    """The deepbind_CNN model input without structure"""

    def __init__(self, config, inf, validation=False, fold_id=1):
        self.folds = folds = config.folds
        (data_one_hot_training, labels_training,
         data_one_hot_test, labels_test,
         training_cases, test_cases,
         seq_length) = (inf["data_one_hot_training"], inf["labels_training"],
                        inf["data_one_hot_test"], inf["labels_test"],
                        inf["training_cases"], inf["test_cases"],
                        inf["seq_length"])
        labels_training = (labels_training - np.mean(labels_training)) / np.sqrt(np.var(labels_training))
        labels_test = (labels_test - np.mean(labels_test)) / np.sqrt(np.var(labels_test))
        self.training_cases = int(training_cases * config.training_frac)
        self.test_cases = int(test_cases * config.test_frac)
        train_index = range(self.training_cases)
        validation_index = range(self.test_cases)
        if validation:
            kf = KFold(n_splits=folds)
            indices = kf.split(range(self.training_cases))
            check = 1
            for train_idx, val_idx in indices:
                if(check == fold_id):
                    train_index = train_idx
                    validation_index = val_idx
                    break
                check = check + 1
            self.training_data = data_one_hot_training[train_index]
            self.test_data = data_one_hot_training[validation_index]
            self.training_labels = labels_training[train_index]
            self.test_labels = labels_training[validation_index]
            self.training_lens = np.array(inf['seq_len_train'], np.int32)[train_index]
            self.test_lens = np.array(inf['seq_len_test'], np.int32)[validation_index]
        else:
            self.training_data = data_one_hot_training[0:self.training_cases]
            self.test_data = data_one_hot_test[0:self.test_cases]
            self.training_labels = labels_training[0:self.training_cases]
            self.test_labels = labels_test[0:self.test_cases]
            self.training_lens = np.array(inf['seq_len_train'], np.int32)[0:self.training_cases]
            self.test_lens = np.array(inf['seq_len_test'], np.int32)[0:self.test_cases]

        self.seq_length = int(seq_length)
        self.training_cases = self.training_data.shape[0]
        self.test_cases = self.test_data.shape[0]
        train_shuffle = np.random.permutation(np.arange(self.training_data.shape[0]))
        self.training_data = self.training_data[train_shuffle]
        self.training_labels = self.training_labels[train_shuffle]
        self.training_lens = self.training_lens[train_shuffle]

class StructInputRNAcompeteS(object):
    """The deepbind_CNN model input with structure"""

    def __init__(self, config, inf, validation=False, fold_id=1):
        self.folds = folds = config.folds
        (data_one_hot_training, labels_training,
         data_one_hot_test, labels_test,
         structures_training, structures_test,
         training_cases, test_cases,
         seq_length) = (inf["data_one_hot_training"], inf["labels_training"],
                        inf["data_one_hot_test"], inf["labels_test"],
                        inf["structures_train"], inf["structures_test"],
                        inf["training_cases"], inf["test_cases"],
                        inf["seq_length"])
        labels_training = (labels_training - np.mean(labels_training)) / np.sqrt(np.var(labels_training))
        labels_test = (labels_test - np.mean(labels_test)) / np.sqrt(np.var(labels_test))
        self.training_cases = int(training_cases * config.training_frac)
        self.test_cases = int(test_cases * config.test_frac)
        train_index = range(self.training_cases)
        validation_index = range(self.test_cases)
        if validation:
            kf = KFold(n_splits=folds)
            indices = kf.split(np.random.choice(training_cases, replace=False, size=self.training_cases))
            check = 1
            for train_idx, val_idx in indices:
                if(check == fold_id):
                    train_index = train_idx
                    validation_index = val_idx
                    break
                check += 1
            self.training_data = data_one_hot_training[train_index]
            self.test_data = data_one_hot_training[validation_index]
            self.training_labels = labels_training[train_index]
            self.test_labels = labels_training[validation_index]
            self.training_struct = np.transpose(structures_training[train_index],[0,2,1])
            self.test_struct = np.transpose(structures_training[validation_index],[0,2,1])
            self.training_lens = np.array(inf['seq_len_train'],np.int32)[train_index]
            self.test_lens = np.array(inf['seq_len_test'],np.int32)[validation_index]
        else:
            self.training_data = data_one_hot_training[0:self.training_cases]
            self.test_data = data_one_hot_test[0:self.test_cases]
            self.training_labels = labels_training[0:self.training_cases]
            self.test_labels = labels_test[0:self.test_cases]
            self.training_struct = np.transpose(structures_training[0:self.training_cases],[0,2,1])
            self.test_struct = np.transpose(structures_test[0:self.test_cases],[0,2,1])
            self.training_lens = np.array(inf['seq_len_train'], np.int32)[0:self.training_cases]
            self.test_lens = np.array(inf['seq_len_test'], np.int32)[0:self.test_cases]

        self.training_data=np.append(self.training_data,self.training_struct,axis=2)
        self.test_data=np.append(self.test_data,self.test_struct,axis=2)
        self.seq_length = int(seq_length)
        self.training_cases = self.training_data.shape[0]
        self.test_cases = self.test_data.shape[0]


class StructInput(object):
    """The deepbind_CNN model input with structure"""

    def __init__(self, config, inf, validation=False, fold_id=1):
        self.folds = folds = config.folds
        (data_one_hot_training, labels_training,
         data_one_hot_test, labels_test,
         pre_train, pre_test,
         difficult_test, simple_test,
         seq_length) = (inf["data_one_hot_training"][:, :, :], inf["labels_training"][:],
                        inf["data_one_hot_test"][:, :, :], inf["labels_test"][:],
                        inf["pred_train"][:], inf["pred_test"][:],
                        inf["difficult_test"], inf["simple_test"],
                        inf["seq_length"][...])
        labels_training = (labels_training - np.mean(labels_training)) / np.sqrt(np.var(labels_training))
        labels_test = (labels_test - np.mean(labels_test)) / np.sqrt(np.var(labels_test))

        training_cases = data_one_hot_training.shape[0]
        test_cases = data_one_hot_test.shape[0]

        # print(training_cases)
        self.training_cases = int(training_cases * config.training_frac)
        self.test_cases = int(test_cases * config.test_frac)
        train_index = range(self.training_cases)
        validation_index = range(self.test_cases)
        if validation:
            kf = KFold(n_splits=folds)
            indices = kf.split(np.arange(0, self.training_cases))
            check = 1
            for train_idx, val_idx in indices:
                # print(train_idx)
                if(check == fold_id):
                    self.train_index = train_index = train_idx
                    self.test_index = validation_index = val_idx
                    break
                check += 1
            self.training_data = data_one_hot_training[train_index]
            self.test_data = data_one_hot_training[validation_index]
            self.training_labels = labels_training[train_index]
            self.test_labels = labels_training[validation_index]
            self.pred_train = pre_train[train_index]
            self.pred_test = pre_test[validation_index]
            # self.training_struct = np.transpose(structures_training[train_index],[0,2,1])
            # self.test_struct = np.transpose(structures_training[validation_index],[0,2,1])
            self.training_lens = np.array(inf['seq_len_train'],np.int32)[train_index]
            self.test_lens = np.array(inf['seq_len_test'],np.int32)[validation_index]
        else:
            self.training_data = data_one_hot_training[0:self.training_cases]
            self.test_data = data_one_hot_test[0:self.test_cases]
            self.training_labels = labels_training[0:self.training_cases]
            self.test_labels = labels_test[0:self.test_cases]
            self.train_index = np.arange(0, self.training_cases)
            self.test_index = np.arange(0, self.test_cases)
            self.pred_train = pre_train[0:self.training_cases]
            self.pred_test = pre_test[0:self.test_cases]
            # self.training_struct = np.transpose(structures_training[0:self.training_cases],[0,2,1])
            # self.test_struct = np.transpose(structures_test[0:self.test_cases],[0,2,1])
            self.training_lens = np.array(inf['seq_len_train'], np.int32)[0:self.training_cases]
            self.test_lens = np.array(inf['seq_len_test'], np.int32)[0:self.test_cases]


        # print(inf['structures_train'].shape)
        # print(self.test_struct.shape)
        # print(self.test_data.shape)

        # self.training_data=np.append(self.training_data,self.training_struct,axis=2)
        # self.test_data=np.append(self.test_data,self.test_struct,axis=2)
        self.seq_length = int(seq_length)
        self.training_cases = self.training_data.shape[0]
        self.test_cases = self.test_data.shape[0]
        # self.shuffle_id = np.random.permutation(np.arange(self.training_data.shape[0]))
        self.difficult_label = difficult_test
        self.simple_label = simple_test

        self.inf = inf

def model_input(input_config, inf, model, validation=False, fold_id=1):
    if 'struct' in model or 'STRUCT' in model or 'attention' in model:
        return StructInput(input_config, inf, validation, fold_id)
    else:
        return NoStructInput(input_config, inf, validation, fold_id)



class CnnModel(object):
    """The deepbind_CNN model with structure"""

    def __init__(self, config, input_):
        self._config = config
        eta_model = config['eta_model']
        lam_model = config['lam_model']
        self.motif_len = config['filter_lengths'][0]  # Tunable Motif length
        self.num_motifs = config['num_filters'][0]  # Number of tunable motifs
        self.motif_len2 = config['filter_lengths'][1]
        self.num_motifs2 = config['num_filters'][1]
        self._init_op = tf.global_variables_initializer()
        self._x = x = tf.placeholder(tf.float32, shape=[None, None, 4], name='One_hot_data')
        self._y_true = y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')
        self.seq_lens = tf.placeholder(tf.int32, shape=[None], name='seq_lens')
        x_image = tf.expand_dims(x, 2)
        W_conv1 = tf.Variable(tf.random_normal([self.motif_len, 1, 4, self.num_motifs], stddev=0.01), name='W_Conv1')
        b_conv1 = tf.Variable(tf.constant(0.001, shape=[self.num_motifs]), name='b_conv1')

        h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                               strides=[1, 1, 1, 1], padding='SAME')
        h_relu_conv1 = tf.nn.relu(h_conv1 + b_conv1, name='First_layer_output')
        W_conv2 = tf.Variable(tf.random_normal([self.motif_len2, 1, self.num_motifs, 1]), name='W_conv2')
        b_conv2 = tf.Variable(tf.constant(0.001, shape=[1]), name='b_conv2')
        h_conv2 = tf.nn.conv2d(h_relu_conv1, W_conv2,
                               strides=[1, 1, 1, 1], padding='SAME')

        h_relu_conv2 = tf.nn.relu(h_conv2 + b_conv2)

        h_max = tf.reduce_max(h_conv2 + b_conv2, axis=[1, 2, 3], name='h_max')
        h_avg = tf.reduce_mean(h_conv2 + b_conv2, axis=[1, 2, 3], name='h_avg')
        W_final = tf.Variable(tf.random_normal([2, 1], stddev=0.1))
        b_final = tf.Variable(tf.constant(0.001, shape=[]))
        h_final = tf.squeeze(tf.matmul(tf.stack([h_max, h_avg], axis=1), W_final) + b_final)
        cost_batch = tf.square(h_final - y_true)
        self._cost = cost = tf.reduce_mean(cost_batch)
        norm_w = (tf.reduce_sum(tf.abs(W_conv1)) + tf.reduce_sum(tf.abs(W_conv2)))
        optimizer = tf.train.AdamOptimizer(learning_rate=eta_model)
        self._train_op = optimizer.minimize(cost + norm_w * lam_model)
        self._predict_op = h_final

    def initialize(self, session):
        session.run(self._init_op)

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict_op(self):
        return self._predict_op

    @property
    def x(self):
        return self._x

    @property
    def y_true(self):
        return self._y_true

class CnnStructModel(object):
    """The deepbind_CNN model with structure"""

    def __init__(self, config, input_):
        self._config = config
        eta_model = config['eta_model']
        lam_model = config['lam_model']
        self.motif_len = config['filter_lengths'][0]  # Tunable Motif length
        self.num_motifs = config['num_filters'][0]  # Number of tunable motifs
        self.motif_len2 = config['filter_lengths'][1]
        self.num_motifs2 = config['num_filters'][0]
        self.embedding_size = config['embedding_size']
        self.k_mer = int(config['k_mer'])

        self._init_op = tf.global_variables_initializer()
        self._x = x = tf.placeholder(tf.float32, shape=[None, None, 14], name='One_hot_data')
        self._y_true = y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')
        self.seq_lens = tf.placeholder(tf.int32, shape=[None], name='seq_lens')

        seq_input = tf.cast(tf.reshape(tf.slice(self._x, [0, 0, 0], [-1, -1, self.k_mer]), [-1, self.k_mer]), dtype = tf.int32)
        struct_input = tf.slice(self._x, [0, 0, 9], [-1, -1, 5])
        self.structure_embedding = tf.get_variable("str_embedding", shape=[sum[self.k_mer], self.embedding_size],
                                                   initializer=tf.contrib.layers.xavier_initializer())
        structure_result = tf.reshape(tf.gather(self.structure_embedding, seq_input), [-1, 41, self.k_mer * self.embedding_size])
        # x = tf.concat([structure_result, struct_input], axis = 2)
        x = structure_result
        x_image = tf.expand_dims(x, 2)
        W_conv1 = tf.get_variable("W_conv1", shape=[self.motif_len, 1, self.k_mer * self.embedding_size, self.num_motifs],
                                                   initializer=tf.contrib.layers.xavier_initializer())
        b_conv1 = tf.Variable(tf.constant(0.001, shape=[self.num_motifs]), name='b_conv1')

        h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                               strides=[1, 1, 1, 1], padding='SAME')
        h_relu_conv1 = tf.nn.relu(h_conv1 + b_conv1, name='First_layer_output')
        W_conv2 = tf.get_variable("W_conv2", shape=[self.motif_len2, 1, self.num_motifs, 1],
                                                   initializer=tf.contrib.layers.xavier_initializer())
        b_conv2 = tf.Variable(tf.constant(0.001, shape=[1]), name='b_conv2')
        h_conv2 = tf.nn.conv2d(h_relu_conv1, W_conv2,
                               strides=[1, 1, 1, 1], padding='SAME')

        h_relu_conv2 = tf.nn.relu(h_conv2 + b_conv2)

        h_max = tf.reduce_max(h_conv2 + b_conv2, axis=[1, 2, 3], name='h_max')
        h_avg = tf.reduce_mean(h_conv2 + b_conv2, axis=[1, 2, 3], name='h_avg')
        W_final = tf.Variable(tf.random_normal([2, 1], stddev=0.1))
        b_final = tf.Variable(tf.constant(0.001, shape=[]))
        h_final = tf.squeeze(tf.matmul(tf.stack([h_max, h_avg], axis=1), W_final) + b_final)
        cost_batch = tf.square(h_final - y_true)
        self._cost = cost = tf.reduce_mean(cost_batch)
        # norm_w = (tf.reduce_sum(tf.abs(W_conv1)) + tf.reduce_sum(tf.abs(W_conv2)))
        norm_w = tf.reduce_sum(tf.square(self.structure_embedding))
        optimizer = tf.train.AdamOptimizer(learning_rate=eta_model)
        self._train_op = optimizer.minimize(cost + norm_w * lam_model)
        self._predict_op = h_final

    def initialize(self, session):
        session.run(self._init_op)

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict_op(self):
        return self._predict_op

    @property
    def x(self):
        return self._x

    @property
    def y_true(self):
        return self._y_true

import math

class CnnStructModel_attention(object):
    """The deepbind_CNN model with structure"""

    def __init__(self, config, input_):
        self._config = config
        eta_model = config['eta_model']
        lam_model = config['lam_model']
        self.motif_len = config['filter_lengths'][0]  # Tunable Motif length
        self.num_motifs = config['num_filters'][0]  # Number of tunable motifs
        self.motif_len2 = config['filter_lengths'][1]
        self.num_motifs2 = config['num_filters'][0]
        self.embedding_size = config['embedding_size']
        self.k_mer = int(config['k_mer'])

        self._init_op = tf.global_variables_initializer()
        self._x = x = tf.placeholder(tf.float32, shape=[None, None, 9 + 6 * 9], name='One_hot_data')
        self._y_true = y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')
        self.seq_lens = tf.placeholder(tf.int32, shape=[None], name='seq_lens')
        self.prob = tf.placeholder(tf.float32, shape=[None, 6], name='prob')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        seq_input = tf.cast(tf.reshape(tf.slice(self._x, [0, 0, 0], [-1, -1, self.k_mer]), [-1, self.k_mer]),
                            dtype=tf.int32)
        struct_input = tf.slice(self._x, [0, 0, 9], [-1, -1, 6 * 9])
        self.structure_embedding = tf.get_variable("str_embedding", shape=[sum[self.k_mer], self.embedding_size],
                                                   initializer=tf.contrib.layers.xavier_initializer())
        structure_result = tf.reshape(tf.gather(self.structure_embedding, seq_input),
                                      [-1, 1, 41, self.k_mer * self.embedding_size])
        structure_result = tf.tile(structure_result, [1, 6, 1, 1])
        struct_input = tf.transpose(tf.reshape(struct_input, [-1, 41, 6, 9]), perm=[0, 2, 1, 3])
        x = tf.reshape(tf.concat([structure_result, struct_input], axis=3),
                       [-1, 6, 41, 9 + self.k_mer * self.embedding_size])
        # x = tf.reshape(structure_result, [-1, 6, 41, self.k_mer * self.embedding_size])
        h_final_list = []
        self.train_op_list = []
        for i in range(ENSEMBLE_NUMBER):
            one_slice = tf.reshape(tf.slice(x, [0, i, 0, 0], [-1, 1, -1, -1]), [-1, 41, 9 + self.k_mer * self.embedding_size])
            x_image = tf.expand_dims(one_slice, 2)
            self.conv = x_image
            kernel_step = 0
            for layer in range(5):
                if layer == 0:
                    kernel_step = self.motif_len
                else:
                    kernel_step = self.motif_len2
                self.conv = tf.layers.conv2d(self.conv, self.num_motifs, (kernel_step, 1), padding='same',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
                self.conv = tf.layers.batch_normalization(self.conv)
                self.conv = tf.nn.relu(self.conv)
            self.conv2 = tf.reshape(tf.squeeze(self.conv, axis=[2], name='lstm_input'), [-1, 41 * self.num_motifs])
            result = tf.layers.dense(self.conv2, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())

            self._cost = cost = tf.reduce_mean(tf.losses.huber_loss(y_true, tf.squeeze(result, axis=1)))
            norm_w = tf.reduce_sum(tf.square(self.structure_embedding))
            self.train_op_list.append(tf.train.AdamOptimizer(learning_rate=eta_model * self.learning_rate).minimize(cost + norm_w * lam_model))

            result = result * tf.slice(self.prob, [0, i], [-1, 1])
            h_final_list.append(result)


        h_final = tf.reduce_sum(tf.concat(h_final_list, axis = 1), axis=1)
        self._cost = cost = tf.reduce_mean(tf.losses.huber_loss(y_true, h_final))
        norm_w = tf.reduce_sum(tf.square(self.structure_embedding))
        optimizer = tf.train.AdamOptimizer(learning_rate=eta_model * self.learning_rate)
        self._train_op = optimizer.minimize(cost + norm_w * lam_model)
        self._predict_op = h_final

    def initialize(self, session):  
        session.run(self._init_op)

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict_op(self):
        return self._predict_op

    @property
    def x(self):
        return self._x

    @property
    def y_true(self):
        return self._y_true

class RnnStructTrackModel(object):
    """The deepbind_RNN model with structure"""

    def __init__(self, config, input_):
        self._config = config
        eta_model = config['eta_model']
        lam_model = config['lam_model']
        self.motif_len = config['motif_len']  # Tunable Motif length
        self.num_motifs = config['num_motifs']  # Number of tunable motifs
        self.motif_len2 = config['motif_len']
        self._init_op = tf.global_variables_initializer()
        self._x = x = tf.placeholder(tf.float32, shape=[None, None, 9], name='One_hot_data')
        self._y_true = y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')
        x_image = tf.expand_dims(x, 2)

        W_conv1 = tf.Variable(tf.random_normal([self.motif_len, 1, 9, self.num_motifs], stddev=0.01), name='W_Conv1')
        b_conv1 = tf.Variable(tf.constant(0.001, shape=[self.num_motifs]), name='b_conv1')

        h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                               strides=[1, 1, 1, 1], padding='SAME')
        h_relu_conv1 = tf.nn.relu(h_conv1 + b_conv1, name='First_layer_output')
        W_conv2 = tf.Variable(tf.random_normal([self.motif_len2, 1, self.num_motifs, 20]), name='W_conv2')
        b_conv2 = tf.Variable(tf.constant(0.001, shape=[20]), name='b_conv2')
        h_conv2 = tf.nn.conv2d(h_relu_conv1, W_conv2,
                               strides=[1, 1, 1, 1], padding='SAME')
        n_hidden = config.get('lstm_size', 20)
        W_out = tf.Variable(tf.random_normal([n_hidden, 1]), name='W_hidden')
        b_out = tf.Variable(tf.constant(0.001, shape=[1]), name='b_hidden')
        h_input = tf.squeeze(tf.nn.relu(h_conv2 + b_conv2), axis=[2], name='lstm_input')
        lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, num_proj=1)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, h_input, dtype=tf.float32)
        h_final_max = tf.reduce_max(outputs, axis=[1, 2])
        h_final_avg = tf.reduce_mean(outputs, axis=[1, 2])
        W_final = tf.Variable(tf.random_normal([2, 1], stddev=0.1))
        b_final = tf.Variable(tf.constant(0.001, shape=[]))
        h_final = tf.squeeze(tf.matmul(tf.stack([h_final_max, h_final_avg], axis=1), W_final) + b_final)

        cost_batch = tf.square(h_final - y_true)
        self._cost = cost = tf.reduce_mean(cost_batch, name='cost')
        norm_w = (tf.reduce_sum(tf.abs(W_conv1)) + tf.reduce_sum(tf.abs(W_conv2)) + tf.reduce_sum(tf.abs(W_out)))
        optimizer = tf.train.AdamOptimizer(learning_rate=eta_model)

        self._train_op = optimizer.minimize(cost + norm_w * lam_model)
        self._predict_op = h_final
        self.lstm_output_layer = outputs
        self.lstm_state_layer = state
        self.lstm_scalar_weight = W_out
        self.lstm_scalar_bias = b_out

    def initialize(self, session):
        session.run(self._init_op)

    @property
    def input(self):
        return self._input

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict_op(self):
        return self._predict_op

    @property
    def x(self):
        return self._x

    @property
    def y_true(self):
        return self._y_true


class RnnStructModel(object):
    """The deepbind_RNN model with structure"""

    def __init__(self, config, input_):
        self._config = config
        eta_model = config['eta_model']
        lam_model = config['lam_model']
        self.motif_len = config['motif_len']  # Tunable Motif length
        self.num_motifs = config['num_motifs']  # Number of tunable motifs
        self.motif_len2 = config['motif_len']
        self.num_motifs2 = config['num_motifs']
        self.embedding_size = config['embedding_size']
        self.k_mer = int(config['k_mer'])

        self._init_op = tf.global_variables_initializer()
        self._x = x = tf.placeholder(tf.float32, shape=[None, None, 14], name='One_hot_data')
        self._y_true = y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')
        self.seq_lens = tf.placeholder(tf.int32, shape=[None], name='seq_lens')

        seq_input = tf.cast(tf.reshape(tf.slice(self._x, [0, 0, 0], [-1, -1, self.k_mer]), [-1, self.k_mer]), dtype = tf.int32)
        struct_input = tf.slice(self._x, [0, 0, 9], [-1, -1, 5])
        self.structure_embedding = tf.get_variable("str_embedding", shape=[sum[self.k_mer], self.embedding_size],
                                                   initializer=tf.contrib.layers.xavier_initializer())
        structure_result = tf.reshape(tf.gather(self.structure_embedding, seq_input), [-1, 41, self.k_mer * self.embedding_size])
        x = tf.concat([structure_result, struct_input], axis = 2)

        x_image = tf.expand_dims(x, 2)
        W_conv1 = tf.Variable(tf.random_normal([self.motif_len, 1, 5 + self.k_mer * self.embedding_size, self.num_motifs], stddev=0.01), name='W_Conv1')
        b_conv1 = tf.Variable(tf.constant(0.001, shape=[self.num_motifs]), name='b_conv1')

        h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                               strides=[1, 1, 1, 1], padding='SAME')
        h_relu_conv1 = tf.nn.relu(h_conv1 + b_conv1, name='First_layer_output')
        W_conv2 = tf.Variable(tf.random_normal([self.motif_len2, 1, self.num_motifs, 1]), name='W_conv2')
        b_conv2 = tf.Variable(tf.constant(0.001, shape=[1]), name='b_conv2')
        h_conv2 = tf.nn.conv2d(h_relu_conv1, W_conv2,
                               strides=[1, 1, 1, 1], padding='SAME')

        n_hidden = config.get('lstm_size', 20)

        W_out = tf.get_variable("W_hidden", shape=[n_hidden, 1],
                        initializer=tf.contrib.layers.xavier_initializer())
        b_out = tf.Variable(tf.constant(0.001, shape=[1]), name='b_hidden')
        h_input = tf.squeeze(tf.nn.relu(h_conv2 + b_conv2), axis=[2], name='lstm_input')
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, h_input, dtype=tf.float32,sequence_length=self.seq_lens)
        last_step = self.extract_axis_1(outputs, self.seq_lens-1)
        h_final = tf.reduce_sum(tf.matmul(last_step,W_out)+b_out, axis=1)
        cost_batch = tf.square(h_final - y_true)
        self._cost = cost = tf.reduce_mean(cost_batch, name='cost')
        norm_w = (tf.reduce_sum(tf.abs(W_conv1)) + tf.reduce_sum(tf.abs(W_conv2)) + tf.reduce_sum(tf.abs(W_out)))
        optimizer = tf.train.AdamOptimizer(learning_rate=eta_model)

        self._train_op = optimizer.minimize(cost + norm_w * lam_model)
        self._predict_op = h_final
        self.lstm_output_layer = outputs
        self.lstm_state_layer = state
        self.lstm_scalar_weight = W_out
        self.lstm_scalar_bias = b_out

    def initialize(self, session):
        session.run(self._init_op)

    def extract_axis_1(self,data, ind):
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res


    @property
    def input(self):
        return self._input

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict_op(self):
        return self._predict_op

    @property
    def x(self):
        return self._x

    @property
    def y_true(self):
        return self._y_true


class RnnModel(object):
    """The deepbind_RNN model with structure"""

    def __init__(self, config, input_):
        self._config = config
        eta_model = config['eta_model']
        lam_model = config['lam_model']
        self.motif_len = config['motif_len']  # Tunable Motif length
        self.num_motifs = config['num_motifs']  # Number of tunable motifs
        self.motif_len2 = config['motif_len']
        self._init_op = tf.global_variables_initializer()
        self._x = x = tf.placeholder(tf.float32, shape=[None, None, 4], name='One_hot_data')
        self._y_true = y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')
        self.seq_lens = tf.placeholder(tf.int32, shape=[None], name='seq_lens')
        x_image = tf.expand_dims(x, 2)

        W_conv1 = tf.get_variable("W_Conv1", shape=[self.motif_len, 1, 4, self.num_motifs],
                        initializer=tf.contrib.layers.xavier_initializer())
        b_conv1 = tf.Variable(tf.constant(0.001, shape=[self.num_motifs]), name='b_conv1')

        h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                               strides=[1, 1, 1, 1], padding='SAME')
        h_relu_conv1 = tf.nn.relu(h_conv1 + b_conv1, name='First_layer_output')
        W_conv2 = tf.get_variable("W_Conv2", shape=[self.motif_len2, 1, self.num_motifs, 20],
                        initializer=tf.contrib.layers.xavier_initializer())
        b_conv2 = tf.Variable(tf.constant(0.001, shape=[20]), name='b_conv2')
        h_conv2 = tf.nn.conv2d(h_relu_conv1, W_conv2,
                               strides=[1, 1, 1, 1], padding='SAME')
        n_hidden = config.get('lstm_size', 20)
        W_out = tf.Variable(tf.random_normal([n_hidden, 1]), name='W_hidden')
        b_out = tf.Variable(tf.constant(0.001, shape=[1]), name='b_hidden')
        h_input = tf.squeeze(tf.nn.relu(h_conv2 + b_conv2), axis=[2], name='lstm_input')
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, h_input, dtype=tf.float32,sequence_length=self.seq_lens)
        last_step = self.extract_axis_1(outputs, self.seq_lens-1)
        h_final = tf.reduce_sum(tf.matmul(last_step,W_out)+b_out, axis=1)
        cost_batch = tf.square(h_final - y_true)
        self._cost = cost = tf.reduce_mean(cost_batch, name='cost')
        norm_w = (tf.reduce_sum(tf.abs(W_conv1)) + tf.reduce_sum(tf.abs(W_conv2)) + tf.reduce_sum(tf.abs(W_out)))
        optimizer = tf.train.AdamOptimizer(learning_rate=eta_model)

        self._train_op = optimizer.minimize(cost + norm_w * lam_model)
        self._predict_op = h_final
        self.lstm_output_layer = outputs
        self.lstm_state_layer = state
        self.lstm_scalar_weight = W_out
        self.lstm_scalar_bias = b_out

    def initialize(self, session):
        session.run(self._init_op)

    def extract_axis_1(self,data, ind):
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res


    @property
    def input(self):
        return self._input

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict_op(self):
        return self._predict_op

    @property
    def x(self):
        return self._x

    @property
    def y_true(self):
        return self._y_true


def model(config, input, model_type):
    if model_type == 'CNN':
        return CnnModel(config, input)
    elif model_type == 'CNN_struct':
        return CnnStructModel(config, input)
    elif model_type == 'CNN_attention':
        return CnnStructModel_attention(config, input)
    elif model_type == 'RNN_struct':
        return RnnStructModel(config, input)
    elif model_type == 'RNN':
        return RnnModel(config, input)
    elif model_type == 'RNN_struct_track':
        return RnnStructTrackModel(config, input)

def run_clip_epoch_parallel(session, model, input_data, config):
    model = model[0]
    scores = np.zeros(input_data.total_cases)
    auc = 0
    window_size = 40
    Nbatch = input_data.total_cases
    for step in range(Nbatch):
        # print(step)
        fetches = {}
        feed_dict = {}
        seq_batch = input_data.data[step:step + 1, :, :]
        last_element = np.where(seq_batch[0, :, 0] == 0)[0]
        if last_element.shape[0]:
            last_element = last_element[0]
        else:
            last_element = seq_batch.shape[1]
        n_batch_seq = (last_element - window_size)
        batch_scores = np.zeros([n_batch_seq])
        for seq_step in range(n_batch_seq):
            input_sequence = seq_batch[:, seq_step:seq_step + window_size, :]
            feed_dict[model.x] = input_sequence
            fetches['predictions'] = model.predict_op
            vals = session.run(fetches, feed_dict)
            batch_scores[seq_step] = vals['predictions']
        scores[step] = np.mean(batch_scores)
    auc = roc_auc_score(input_data.labels, scores)
    return auc


def run_clip_epoch_shorter(session, models, input_data, config):
    # model = model[0]
    Nbatch = int(ceil(input_data.total_cases * 1.0 / config['minib']))
    minib = config['minib']
    fetches = {}
    feed_dict = {}
    scores = np.zeros([len(models), input_data.total_cases])
    for step in range(Nbatch):
        for i, model in enumerate(models):
            feed_dict[model.x] = input_data.data[(minib * step): (minib * (step + 1)), :, :]
            feed_dict[model.y_true] = input_data.labels[(minib * step): (minib * (step + 1))]
            feed_dict[model.seq_lens] = input_data.seq_length[(minib * step): minib * (step + 1)]
            fetches["predictions" + str(i)] = model.predict_op
        vals = session.run(fetches,feed_dict)
        for j in range(len(models)):
            scores[j,minib*step:minib*(step+1)] = vals['predictions'+str(j)]
    scores_ensemble = np.mean(scores, axis=0)
    auc = roc_auc_score(input_data.labels, scores_ensemble)
    return auc


def run_epoch_parallel(session, models, input_data, config, epoch, train=False, verbose=False, testing=False,
                       scores=False, validation = False):
    input = input_data
    if isinstance(input_data,list):
        Nbatch_train = int(ceil(input_data[0].training_cases * 1.0 / config['minib']))
        Nbatch_test = int(ceil(input_data[0].test_cases * 1.0 / config['minib']))

        training_scores = np.zeros([len(models), input_data[0].training_cases])
        test_scores = np.zeros([len(models), input_data[0].test_cases])
    else:
        Nbatch_train = int(ceil(input_data.training_cases * 1.0 / config['minib']))
        Nbatch_test = int(ceil(input_data.test_cases * 1.0 / config['minib']))

        training_scores = np.zeros([len(models), input_data.training_cases])
        test_scores = np.zeros([len(models), input_data.test_cases])

    minib = config['minib']
    num_models = len(models)
    cost_temp = np.zeros([num_models])
    pearson_train = np.zeros([num_models, 2])

    # print(input.inf['structures_test'].shape)


    learn_rate = 1.
    if epoch >= 5:
        learn_rate *= 0.1
    #
    # if epoch >= 20:
    #     learn_rate *= 0.1

    for step in range(0, Nbatch_train):
        # if step % 500 == 0:
        #     print(str(step) + "/" + str(Nbatch_train))
        fetches = {}
        feed_dict = {}
        # print(isinstance(input_data,list))
        if isinstance(input_data,list):
            for i,(model,input) in enumerate(zip(models,input_data)):
                k_structure = input.train_index[
                    np.arange((minib * step), min(input.training_cases, (minib * (step + 1))))]
                k = np.arange((minib * step), min(input.training_cases, (minib * (step + 1))))
                seq_input = input.training_data[k, :, :]
                struct_input = input.inf['structures_train'][k_structure, ...]
                struct_input = np.transpose(struct_input, [0, 2, 1])
                all_input = np.append(seq_input, struct_input, axis=2)
                for i, model in enumerate(models):
                    feed_dict[model.x] = all_input
                    feed_dict[model.y_true] = input.training_labels[k]
                    feed_dict[model.seq_lens] = input.training_lens[k]
                    feed_dict[model.prob] = input.pred_train[k]
                    feed_dict[model.learning_rate] = learn_rate
                    fetches["cost" + str(i)] = model.cost
                    # if train:
                    #     fetches["eval_op" + str(i) + str(t)] = model.train_op_list[t]
                    fetches["predictions" + str(i)] = model.predict_op
        else:
            # k = input.shuffle_id[(minib * step): (minib * (step + 1))]
            # k.sort()
            k_structure = input_data.train_index[np.arange((minib * step), min(input_data.training_cases, (minib * (step + 1))))]
            k = np.arange((minib * step), min(input_data.training_cases, (minib * (step + 1))))
            seq_input = input.training_data[k, :, :]
            struct_input = input.inf['structures_train'][k_structure,...]
            struct_input = np.transpose(struct_input, [0, 2, 1])
            all_input = np.append(seq_input, struct_input, axis=2)
            for i, model in enumerate(models):
                feed_dict[model.x] = all_input
                feed_dict[model.y_true] = input.training_labels[k]
                feed_dict[model.seq_lens] = input.training_lens[k]
                feed_dict[model.prob] = input.pred_train[k]
                feed_dict[model.learning_rate] = learn_rate
                fetches["cost"+str(i)] = model.cost
                # if train:
                #     fetches["eval_op" + str(i) + str(t)] = model.train_op_list[t]
                fetches["predictions" + str(i)] = model.predict_op
        if train:
            for t in range(ENSEMBLE_NUMBER):
                fetches["eval_op"] = model.train_op_list[t]
                vals = session.run(fetches, feed_dict)
                # print("{}\t{}".format(t,vals['cost0']))
        else:
            vals = session.run(fetches, feed_dict)
        for j in range(num_models):
            cost_temp[j] += vals["cost"+str(j)]
            training_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]
    cost_train = cost_temp / Nbatch_train
    for j in range(num_models):
        if isinstance(input_data, list):
            pearson_train[j, :] = stats.pearsonr(input_data[j].training_labels, training_scores[j, :])
        else:
            pearson_train[j, :] = stats.pearsonr(input_data.training_labels, training_scores[j, :])
    cost_temp = np.zeros([num_models])
    if testing or scores:
        pearson_test = np.zeros([num_models, 2])
        pearson_easy = np.zeros([num_models, 2])
        pearson_difficult = np.zeros([num_models, 2])
        for step in range(Nbatch_test):
            feed_dict = {}
            fetches = {}

            # print(step)
            # print(Nbatch_test)

            if isinstance(input_data, list):
                for i, (model, input) in enumerate(zip(models, input_data)):
                    k_structure = input.test_index[
                        np.arange((minib * step), min(input.test_cases, (minib * (step + 1))))]
                    k = np.arange((minib * step), min(input.test_cases, (minib * (step + 1))))
                    seq_input = input.test_data[k, :, :]
                    if validation == False:
                        struct_input = input.inf['structures_test'][k_structure, :, :]
                    else:
                        struct_input = input.inf['structures_train'][k_structure, :, :]
                    struct_input = np.transpose(struct_input, [0, 2, 1])
                    all_input = np.append(seq_input, struct_input, axis=2)

                    feed_dict[model.x] = all_input
                    feed_dict[model.y_true] = input.test_labels[k]
                    feed_dict[model.seq_lens] = input.test_lens[k]
                    feed_dict[model.prob] = input.pred_test[k]

                    fetches["cost" + str(i)] = model.cost
                    fetches["predictions" + str(i)] = model.predict_op
            else:
                for i, model in enumerate(models):
                    k_structure = input_data.test_index[np.arange((minib * step), min(input_data.test_cases, (minib * (step + 1))))]
                    k = np.arange((minib * step), min(input_data.test_cases, (minib * (step + 1))))
                    seq_input = input.test_data[k, :, :]
                    if validation == False:
                        struct_input = input.inf['structures_test'][k_structure, :, :]
                    else:
                        struct_input = input.inf['structures_train'][k_structure, :, :]
                    struct_input = np.transpose(struct_input, [0, 2, 1])
                    all_input = np.append(seq_input, struct_input, axis=2)

                    feed_dict[model.x] = all_input
                    feed_dict[model.y_true] = input.test_labels[k]
                    feed_dict[model.seq_lens] = input.test_lens[k]
                    feed_dict[model.prob] = input.pred_test[k]

                    fetches["cost"+str(i)] = model.cost
                    fetches["predictions"+str(i)] = model.predict_op
            vals = session.run(fetches, feed_dict)

            for j in range(num_models):
                cost_temp[j] += vals["cost" + str(j)]
                test_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]
                # if isinstance(input_data,list):
                #     mbatchY_test = input_data[i].test_labels[(minib * step): (minib * (step + 1))]
                # else:
                #     mbatchY_test = input_data.test_labels[(minib * step): (minib * (step + 1))]
                # cost_test[j] += vals["cost"+str(j)]
                # pearson_test[j] += stats.pearsonr(mbatchY_test, vals["predictions"+str(j)])[0]
        cost_test = cost_temp / Nbatch_test
        if isinstance(input_data,list):
            pearson_ensemble = stats.pearsonr(input_data[0].test_labels, np.mean(test_scores, axis=0))[0]
            cost_ensemble = np.mean(np.square(input_data[0].test_labels - np.mean(test_scores, axis=0)))
        else:
            pearson_ensemble = stats.pearsonr(input_data.test_labels, np.mean(test_scores, axis=0))[0]
            cost_ensemble = np.mean(np.square(input_data.test_labels-np.mean(test_scores, axis=0)))
        for j in range(num_models):
            if isinstance(input_data, list):
                pearson_test[j, :] = stats.pearsonr(input_data[j].test_labels, test_scores[j, :])
                #pearson_easy[j, :] = stats.pearsonr(input_data[j].test_labels[input_data[j].simple_label], test_scores[j, :][input_data[j].simple_label])
                #pearson_difficult[j, :] = stats.pearsonr(input_data[j].test_labels[input_data[j].difficult_label], test_scores[j, :][input_data[j].difficult_label])
            else:
                pearson_test[j, :] = stats.pearsonr(input_data.test_labels, test_scores[j, :])
                #pearson_easy[j, :] = stats.pearsonr(input_data[j].test_labels[input_data[j].simple_label], test_scores[j, :][input_data[j].simple_label])
                #pearson_difficult[j, :] = stats.pearsonr(input_data[j].test_labels[input_data[j].difficult_label], test_scores[j, :][input_data[j].difficult_label])
        if verbose:
            best_model = np.argmin(cost_train)
            print(
                "Epoch:%04d, minib:%d,Train cost(min)=%0.4f, Train pearson=%0.4f, Test cost(min)=%0.4f, Test Pearson(max)=%0.4f, Difficult_test_Pearson(max)=%0.4f, Simple_test_Pearson(max)=%0.4f Ensemble Pearson=%0.4f Ensemble Cost=%0.4f" %
                (epoch + 1, minib, cost_train[best_model], pearson_train[best_model][0], cost_test[best_model],
                 pearson_test[best_model][0], pearson_difficult[best_model][0], pearson_easy[best_model][0],
                 pearson_ensemble, cost_ensemble))
            print(["%.4f" % p for p in pearson_test[:, 0]])
        if scores:
            return (
            cost_train, cost_test, pearson_train, pearson_test, pearson_easy, pearson_difficult, training_scores,
            test_scores)
        return (
        cost_train, cost_test, pearson_test[:, 0], pearson_easy[:, 0], pearson_difficult[:, 0], pearson_ensemble,
        cost_ensemble)
    return cost_train

def run_epoch_parallel_rnacs(session, models, input_data, config, epoch, train=False, verbose=False, testing=False,
                       scores=False):
    if isinstance(input_data,list):
        Nbatch_train = int(ceil(input_data[0].training_cases * 1.0 / config['minib']))
        Nbatch_test = int(ceil(input_data[0].test_cases * 1.0 / config['minib']))

        training_scores = np.zeros([len(models), input_data[0].training_cases])
        test_scores = np.zeros([len(models), input_data[0].test_cases])
    else:
        Nbatch_train = int(ceil(input_data.training_cases * 1.0 / config['minib']))
        Nbatch_test = int(ceil(input_data.test_cases * 1.0 / config['minib']))

        training_scores = np.zeros([len(models), input_data.training_cases])
        test_scores = np.zeros([len(models), input_data.test_cases])

    minib = config['minib']
    num_models = len(models)
    cost_temp = np.zeros([num_models])
    auc_train = np.zeros([num_models])

    for step in range(Nbatch_train):
        fetches = {}
        feed_dict = {}
        if isinstance(input_data,list):
            for i,(model,input) in enumerate(zip(models,input_data)):
                feed_dict[model.x] = input.training_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input.training_labels[(minib * step): (minib * (step + 1))]
                feed_dict[model.seq_lens] = input.training_lens[(minib*step): minib*(step+1)]
                fetches["cost" + str(i)] = model.cost
                if train:
                    fetches["eval_op" + str(i)] = model.train_op

                fetches["predictions" + str(i)] = model.predict_op
        else:
            for i, model in enumerate(models):
                feed_dict[model.x] = input_data.training_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input_data.training_labels[(minib * step): (minib * (step + 1))]
                feed_dict[model.seq_lens] = input_data.training_lens[(minib * step): minib * (step + 1)]
                fetches["cost"+str(i)] = model.cost
                if train:
                    fetches["eval_op" +str(i)] = model.train_op
                fetches["predictions" + str(i)] = model.predict_op
        vals = session.run(fetches, feed_dict)
        for j in range(num_models):
            cost_temp[j] += vals["cost"+str(j)]
            training_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]
    cost_train = cost_temp / Nbatch_train
    for j in range(num_models):
        if isinstance(input_data, list):
            auc_train[j] = roc_auc_score(input_data[j].training_labels, training_scores[j, :])
        else:
            auc_train[j, :] = roc_auc_score(input_data.training_labels, training_scores[j, :])
    cost_temp = np.zeros([num_models])
    if testing or scores:
        auc_test = np.zeros([num_models])
        for step in range(Nbatch_test):
            feed_dict = {}
            fetches = {}

            if isinstance(input_data, list):
                for i, (model, input) in enumerate(zip(models, input_data)):
                    feed_dict[model.x] = input.test_data[(minib * step): (minib * (step + 1)), :, :]
                    feed_dict[model.y_true] = input.test_labels[(minib * step): (minib * (step + 1))]
                    feed_dict[model.seq_lens] = input.test_lens[(minib * step): minib * (step + 1)]
                    fetches["cost" + str(i)] = model.cost
                    fetches["predictions" + str(i)] = model.predict_op
            else:
                for i, model in enumerate(models):
                    feed_dict[model.x] = input_data.test_data[(minib * step): (minib * (step + 1)), :, :]
                    feed_dict[model.y_true] = input_data.test_labels[(minib * step): (minib * (step + 1))]
                    feed_dict[model.seq_lens] = input_data.test_lens[(minib * step): minib * (step + 1)]
                    fetches["cost"+str(i)] = model.cost
                    fetches["predictions"+str(i)] = model.predict_op
            vals = session.run(fetches, feed_dict)

            for j in range(num_models):
                cost_temp[j] += vals["cost" + str(j)]
                test_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]
                # if isinstance(input_data,list):
                #     mbatchY_test = input_data[i].test_labels[(minib * step): (minib * (step + 1))]
                # else:
                #     mbatchY_test = input_data.test_labels[(minib * step): (minib * (step + 1))]
                # cost_test[j] += vals["cost"+str(j)]
                # pearson_test[j] += stats.pearsonr(mbatchY_test, vals["predictions"+str(j)])[0]
        cost_test = cost_temp / Nbatch_test
        if isinstance(input_data,list):
            auc_ensemble = roc_auc_score(input_data[0].test_labels, np.mean(test_scores, axis=0))
        else:
            auc_ensemble = roc_auc_score(input_data.test_labels, np.mean(test_scores, axis=0))
        for j in range(num_models):
            if isinstance(input_data, list):
                auc_test[j] = roc_auc_score(input_data[j].test_labels, test_scores[j, :])
            else:
                auc_test[j] = roc_auc_score(input_data.test_labels, test_scores[j, :])
        if verbose:
            best_model = np.argmin(cost_train)
            print (
            "Epoch:%04d, Train cost(min)=%0.4f, Train AUC=%0.4f, Test cost(min)=%0.4f, Test AUC(max)=%0.4f Ensemble AUC=%0.4f" %
            (epoch + 1, cost_train[best_model], auc_train[best_model], cost_test[best_model],
             auc_test[best_model],auc_ensemble))
            print(["%.4f" % p for p in auc_test[:]])
        if scores:
            return (cost_train, cost_test, auc_train, auc_test, training_scores, test_scores)
        return (cost_train, cost_test, auc_test)
    return cost_train


def train_model_parallel(session, train_config, models, input_data,epochs, early_stop = False, savedir=None,saver=None, validation = False):
    """Trains a list of models in parallel. Expects a list of inputs of equal length as models. Config file is u """
    num_models = len(models)
    cost_train = np.zeros([epochs, num_models])
    cost_test = np.zeros([epochs, num_models])
    pearson_test = np.zeros([epochs, num_models])
    pearson_easy = np.zeros([epochs, num_models])
    pearson_difficult = np.zeros([epochs, num_models])
    pearson_ensemble = np.zeros([epochs])
    cost_ensemble = np.zeros([epochs])
    pearson_max = -np.inf
    max_minib = train_config['minib']
    num_batch_step = train_config.get('batch_size_steps', 3)
    min_minib = max_minib // (2**(num_batch_step-1))
    if train_config.get('batch_size_boosting', True):
        batch_sizes = [min_minib*(2**((num_batch_step*j)//epochs)) for j in range(epochs)]
    else:
        batch_sizes = [max_minib for j in range(epochs)]
    print(epochs)
    for i in range(epochs):
        train_config['minib'] = batch_sizes[i]
        _ = run_epoch_parallel(session, models, input_data, train_config, i, train=True)
        train_config['minib'] = max_minib
        (cost_train[i], cost_test[i], pearson_test[i], pearson_easy[i], pearson_difficult[i], pearson_ensemble[i], cost_ensemble[i]) = \
        run_epoch_parallel(session, models, input_data, train_config, i, train=False,
                           verbose=True, testing = True, validation = validation)
        if early_stop and not(i%5):
            if pearson_ensemble[i] > pearson_max:
                best_epoch = i
                saver.save(session, savedir)
                print("[*] Saving early stop checkpoint")
    i = epochs - 1
    if saver and(pearson_ensemble[i] > pearson_max):
        best_epoch = i
        saver.save(session, savedir)
        print("[*] Saving checkpoint")

    cost_test = np.transpose(cost_test,[1,0])
    pearson_test = np.transpose(pearson_test,[1,0])
    if early_stop :
        pearson_ensemble = pearson_ensemble[best_epoch]
        cost_ensemble = cost_ensemble[best_epoch]
        pearson_easy = pearson_easy[best_epoch]
        pearson_difficult = pearson_difficult[best_epoch]
    else:
        pearson_ensemble = pearson_ensemble[-1]
        cost_ensemble = cost_ensemble[-1]
        pearson_easy = pearson_easy[-1]
        pearson_difficult = pearson_difficult[-1]
        if saver:
            saver.save(session, savedir)
            print("[*] Saving checkpoint")
    return (cost_test,pearson_test, pearson_easy, pearson_difficult, cost_ensemble, pearson_ensemble)

def train_model_parallel_rnacs(session, config, models, input_data, early_stop = False):
    """Trains a list of models in parallel. Expects a list of inputs of equal length as models. Config file is u """
    if early_stop:
        epochs = config['early_stop_epochs']
    else:
        epochs = 15
    test_epochs = 15
    num_models = len(models)
    cost_train = np.zeros([test_epochs, num_models])
    cost_test = np.zeros([test_epochs, num_models])
    auc_test = np.zeros([test_epochs, num_models])
    # session.run(tf.global_variables_initializer())
    for i in range(15):
        _ = run_epoch_parallel_rnacs(session, models, input_data, config, i, train=True)
        step = i
        (cost_train[step], cost_test[step], auc_test[step]) = \
        run_epoch_parallel_rnacs(session, models, input_data, config, i, train=False,
                           verbose=True, testing = True)

    cost_test = np.transpose(cost_test,[1,0])
    auc_test = np.transpose(auc_test,[1,0])
    return (cost_test,auc_test)


def compute_gradient(session, model, input_data, config):
    Nbatch_test = int(ceil(input_data.test_cases * 1.0 / config['minib']))
    minib = config['minib']
    predictions_test = np.zeros(shape=input_data.test_data.shape[0])
    gradients_test = np.zeros(shape=input_data.test_data.shape)

    for step in range(Nbatch_test):
        fetches = {}
        feed_dict = {}
        feed_dict[model.x] = input_data.test_data[(minib * step): (minib * (step + 1)), :, :]
        feed_dict[model.y_true] = input_data.test_labels[(minib * step): (minib * (step + 1))]
        fetches["predictions"] = model.predict_op
        fetches['gradient'] = tf.gradients(model.cost, model.x)
        vals = session.run(fetches, feed_dict)
        gradients_test[(minib * step): (minib * (step + 1)), :, :] = (vals['gradient'][0])
        predictions_test[(minib * step): (minib * (step + 1))] = (vals['predictions'])
    return predictions_test, gradients_test


def evaluate_model_parallel(session, config, models, input_data):
    """Evaluates a list of models in parallel. Expects a list of inputs of equal length as models"""
    num_models = len(models)
    cost_train = np.zeros([num_models])
    cost_test = np.zeros([num_models])
    pearson_test = np.zeros([num_models])
    (cost_train, cost_test, pearson_test) = \
        run_epoch_parallel(session, models, input_data, config, 0, train=False, verbose=True, testing=True)
    return (cost_test, pearson_test)


def score_model_parallel(session, config, models, input_data):
    if isinstance(input_data, list):
        Nbatch_train = int(ceil(input_data[0].training_cases * 1.0 / config['minib']))
        Nbatch_test = int(ceil(input_data[0].test_cases * 1.0 / config['minib']))
        training_scores = np.zeros([len(models), input_data[0].training_cases])
        test_scores = np.zeros([len(models), input_data[0].test_cases])
    else:
        Nbatch_train = int(ceil(input_data.training_cases * 1.0 / config['minib']))
        Nbatch_test = int(ceil(input_data.test_cases * 1.0 / config['minib']))
        training_scores = np.zeros([len(models), input_data.training_cases])
        test_scores = np.zeros([len(models), input_data.test_cases])
    minib = config['minib']
    num_models = len(models)

    for step in range(Nbatch_train):
        fetches = {}
        feed_dict = {}
        if isinstance(input_data, list):
            for i, (model, input) in enumerate(zip(models, input_data)):
                feed_dict[model.x] = input.training_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input.training_labels[(minib * step): (minib * (step + 1))]
                fetches["predictions" + str(i)] = model.predict_op

        else:
            for i, model in enumerate(models):
                feed_dict[model.x] = input_data.training_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input_data.training_labels[(minib * step): (minib * (step + 1))]
                fetches["predictions" + str(i)] = model.predict_op

        vals = session.run(fetches, feed_dict)
        for j in range(num_models):
            training_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]

    for step in range(Nbatch_test):
        feed_dict = {}
        fetches = {}

        if isinstance(input_data, list):
            for i, (model, input) in enumerate(zip(models, input_data)):
                feed_dict[model.x] = input.test_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input.test_labels[(minib * step): (minib * (step + 1))]
                fetches["predictions" + str(i)] = model.predict_op
        else:
            for i, model in enumerate(models):
                feed_dict[model.x] = input_data.test_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input_data.test_labels[(minib * step): (minib * (step + 1))]
                fetches["predictions" + str(i)] = model.predict_op
        vals = session.run(fetches, feed_dict)

        for j in range(num_models):
            test_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]
    for j in range(num_models):
        pearson_test = stats.pearsonr(input_data.test_labels, test_scores[j, :])
        pearson_training = stats.pearsonr(input_data.training_labels, training_scores[j, :])
    return (training_scores, test_scores, pearson_training, pearson_test)


#
# def create_config_dict(**kwargs):
#     config = {}
#     config.update(kwargs)
#     config['folds'] = 3
#     if config['flag']=='large':
#         config['epochs'] = 15
#         config['early_stop_epochs'] = 15
#         config['test_interval'] = 1
#     elif config['flag'] == 'medium':
#         config['epochs'] = 10
#         config['early_stop_epochs'] = 10
#         config['test_interval'] = 1
#     else:
#         config['epochs'] = 10
#         config['early_stop_epochs'] = 10
#         config['test_interval'] = 1
#     return config


def save_calibration(train_config, best_config):
    protein = train_config['protein']
    model_type = train_config['model_type']
    save_dir = train_config['hp_dir']
    file_name = os.path.join(save_dir,protein+'_'+model_type+'.npz')
    new_cost = best_config['cost']
    new_pearson = best_config['pearson']
    save_new = True
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if (os.path.isfile(file_name)):
        file = np.load(file_name)
        if new_cost >= file['cost']:
            save_new = False

    if (save_new):
        print("[*] Updating best calibration for %s %s"%(protein,model_type))
        np.savez(file_name, **best_config)
    else:
        print("[*] Retaining existing calibration for %s %s" % (protein, model_type))


def save_result(train_config,new_cost, new_pearson, ensemble_size, model_dir, new_easy_pearson, new_difficult_pearson):
    protein = train_config['protein']
    model_type = train_config['model_type']
    save_dir = train_config['result_dir']
    import yaml
    file_name = os.path.join(save_dir, protein + '_' + model_type + '.npz')
    save_new = True
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if (os.path.isfile(file_name)):
        file = np.load(file_name)
        if new_pearson <= file['pearson']:
            save_new = False

    if (save_new):
        print("[*] Updating best result for %s %s" % (protein, model_type))
        np.savez(file_name,
                 cost=new_cost,
                 pearson=new_pearson
                 )
        result_dict = {'cost': float(new_cost), 'pearson': float(new_pearson), 'ensemble_size': int(ensemble_size),
                       'model_dir': str(model_dir), 'pearson_simple': float(new_easy_pearson), 'pearson_difficult': float(new_difficult_pearson)}
        yaml.dump(result_dict, open(os.path.join(save_dir, protein + '_' + model_type + '.yml'),'w'))

def load_calibration(train_config):
    file_name = os.path.join(train_config['hp_dir'], train_config['protein']+'_'+train_config['model_type']+'.npz')
    print(os.path.isfile(file_name))
    if not os.path.isfile(file_name):
        # print("[!] Model is not pre-calibrated!")
        return False
    print("[*] Loading existing best calibration for %s %s" % (train_config['protein'], train_config['model_type']))
    inf = np.load(file_name)
    loaded_config = {}
    loaded_config.update(inf)
    return loaded_config



class input_config(object):
    """Generates configuration for processing input to model"""
    def __init__(self, flag):
        self.folds = 3
        if flag == 'large':
            self.training_frac = 1
            self.test_frac  = 1
        elif flag == 'medium':
            self.training_frac = 0.5
            self.test_frac = 1
        else:
            self.training_frac = 0.2
            self.test_frac = 1


def load_data_rnac2013(protein_name):
    # type: (object, object) -> object


    infile_seq_tmp = open('../data/rnac/sequences.tsv')
    infile_target_tmp = open('../data/rnac/new_targets.tsv')
    seq_train = []
    seq_test = []
    length = []
    target_train = []
    target_test = []
    exp_ids_train = []
    exp_ids_test = []

    average_struct = np.load('../data/rnac/new_average.txt.npy').reshape([-1, 9, 41])
    data_flag = np.zeros([average_struct.shape[0], 1])
    import scipy.sparse
    t = scipy.sparse.load_npz("../data/rnac/top_5_contact_map.txt.npz")
    # feature_map_input = np.array(t.toarray())
    # infile_struct = np.reshape(feature_map_input, [-1, 500, 41]).astype(np.float64)
    # infile_struct = np.mean(infile_struct, axis = 1)
    # t = np.reshape(t, [-1, 500, 41])
    print("pass")

    random_list = np.random.permutation(t.shape[0])
    t = t[random_list, :]
    data_flag = data_flag[random_list, :]
    average_struct = average_struct[random_list, :]
    sequence_input = np.stack(
        [np.load("../data/rnac/embedding_right/embedding" + str(i) + ".txt.npy") for i in range(1, 10)], axis=2)
    # print(sequence_input.shape)
    sequence_input = sequence_input[random_list, :, :]
    predict_input = np.load("../data/rnac/pred_list_5.npy")
    predict_input = predict_input[random_list, :]
    # random_list = np.append(random_list1, random_list2)
    # random_list = np.append(np.array(0), random_list)

    seq_list = []
    target_list = []

    for line_seq in infile_seq_tmp:
        line_target = infile_target_tmp.readline()
        # print(line_target)
        # print(line_seq)
        seq_list.append(line_seq)
        target_list.append(line_target)

    # print(len(seq_list))

    infile_seq_random = open('../data/rnac/sequences_random' + protein_name + '.tsv', 'w')
    infile_target_random = open('../data/rnac/targets_random' + protein_name + '.tsv', 'w')

    # print(seq_list.shape)

    for i in range(len(seq_list)):
        # print(seq_list[random_list[i]])
        # print(random_list[i])
        if i == 0:
            infile_seq_random.write(seq_list[i])
            infile_target_random.write(target_list[i])
        else:
            infile_seq_random.write(seq_list[random_list[i - 1] + 1])
            infile_target_random.write(target_list[random_list[i - 1] + 1])

    infile_seq_random.close()
    infile_target_random.close()

    infile_seq = open('../data/rnac/sequences_random' + protein_name + '.tsv')
    infile_target = open('../data/rnac/targets_random' + protein_name + '.tsv')

    structures_A = []
    structures_B = []
    seq_len_A = []
    seq_len_B = []
    seq_len_train = 41
    num_struct_classes = 9
    belong = {}
    seq_len_train = 0
    seq_len_test = 0

    target_names = infile_target.readline().split()
    target_col = target_names.index(protein_name)

    # if target_id_list is None:
    #     target_id_list = target_names
    # target_cols_idx = np.zeros(len(target_id_list), dtype=int)
    # # target_cols_idx = target_names.index(target_id_list)
    #
    # for i in range(len(target_id_list)):
    #     target_cols_idx[i] = target_names.index(target_id_list[i])
    infile_seq.readline()
    dic_ban = {}
    test_difficult = []
    test_simple = []

    for index, line_seq in enumerate(infile_seq):
        seq = line_seq.split('\t')[2].strip()
        line_target = infile_target.readline()
        target = [line_target.split('\t')[target_col]]
        fold = line_seq.split('\t')[0].strip()
        target_np = np.array(target, dtype=float)
        length.append(len(seq))
        if np.any(np.isnan(target_np)):
            dic_ban[index] = 1
            continue
        if fold == 'A':
            seq_train.append(seq)
            target_train.append(target)
            exp_ids_train.append(line_seq.split('\t')[1].strip())
            seq_len_train = max(seq_len_train, len(seq))
            seq_len_A.append(len(seq))
            belong[index] = 1
        else:
            seq_test.append(seq)
            target_test.append(target)
            exp_ids_test.append(line_seq.split('\t')[1].strip())
            seq_len_test = max(seq_len_test, len(seq))
            seq_len_B.append(len(seq))
            belong[index] = 2
            if data_flag[index, 0] == 1:
                test_difficult.append(len(seq_test) - 1)
            else:
                test_simple.append(len(seq_test) - 1)

    seq_len_A = np.array(seq_len_A)
    seq_len_B = np.array(seq_len_B)
    iter_train = 0
    seq_length = max(seq_len_test, seq_len_train)
    iter_test = 0
    line_id = -1
    times = 0
    save_target = "../data/rnac/npz_archives_update/" + protein_name  + ".h5"
    true_length = 0
    exist_file_A = exist_file_B = False
    block_num_A = block_num_B = 0
    pred_A = []
    pred_B = []
    structure_num = 6
    np.set_printoptions(threshold=np.inf)
    # for line_struct in infile_struct:
    for i in range(len(length)):
        # print(i)
        line_id += 1
        prob = predict_input[line_id][-1]
        rank = np.argsort(-predict_input[line_id][:-1])
        data = np.reshape(t[line_id: (line_id + 1), :].toarray(), [-1, seq_length])
        add_weigh = np.sum(np.reshape(data, [-1, num_struct_classes, seq_length]) * np.expand_dims(np.expand_dims(predict_input[line_id][:-1], axis=1), axis = 2), axis=0)
        average = (average_struct[i, :, :] - add_weigh) / (prob + 0.0000001)

        data = np.reshape(np.reshape(data, [-1, num_struct_classes, seq_length])[rank], [-1, seq_length])
        rank = np.append(rank, np.array([5]))

        line_struct = np.concatenate((data, average), axis=0)
        probs = np.ones([structure_num * num_struct_classes, seq_length]) * (1.0 / num_struct_classes)
        probs[:, 0:length[line_id]] = line_struct[:, 0:length[line_id]]
        # probs = np.reshape(probs, [-1, num_struct_classes, seq_length])
        if (line_id in dic_ban) == False:
            if belong[line_id] == 1:
                structures_A.append(probs)
                pred_A.append(predict_input[line_id][rank])
            elif belong[line_id] == 2:
                structures_B.append(probs)
                pred_B.append(predict_input[line_id][rank])

        length_A = len(structures_A)
        length_B = len(structures_B)

        if length_A == 10000:
            block_num_A += 1
            structures_train = np.array(structures_A, dtype=np.float32) - (1.0 / num_struct_classes)
            if exist_file_A == False:
                exist_file_A = True
                if exist_file_B == False:
                    h5f = h5py.File(save_target, 'w')
                else:
                    h5f = h5py.File(save_target, 'a')
                structures_train_pointer = h5f.create_dataset("structures_train", (10000, num_struct_classes * structure_num, seq_length),
                                             maxshape=(None, num_struct_classes * structure_num, seq_length),
                                             dtype='float32')
            else:
                h5f = h5py.File(save_target, 'a')
                structures_train_pointer = h5f['structures_train']
            structures_train_pointer.resize([block_num_A * 10000, num_struct_classes * structure_num, seq_length])
            structures_train_pointer[(block_num_A - 1) * 10000:block_num_A * 10000] = structures_train
            structures_A = []
            # print(h5f['structures_train'].shape)

        if length_B == 10000:
            block_num_B += 1
            structures_test = np.array(structures_B, dtype=np.float32) - (1.0 / num_struct_classes)
            if exist_file_B == False:
                exist_file_B = True
                if exist_file_A == False:
                    h5f = h5py.File(save_target, 'w')
                else:
                    h5f = h5py.File(save_target, 'a')
                structures_test_pointer = h5f.create_dataset("structures_test", (10000, num_struct_classes * structure_num, seq_length),
                                             maxshape=(None, num_struct_classes * structure_num, seq_length),
                                             dtype='float32')
            else:
                h5f = h5py.File(save_target, 'a')
                structures_test_pointer = h5f['structures_test']
            structures_test_pointer.resize([block_num_B * 10000, num_struct_classes * structure_num, seq_length])
            structures_test_pointer[(block_num_B - 1) * 10000:block_num_B * 10000] = structures_test
            structures_B = []
            # print(h5f['structures_test'].shape)

    length_A = len(structures_A) + block_num_A * 10000
    length_B = len(structures_B) + block_num_B * 10000
    structures_train = np.array(structures_A, dtype=np.float32) - (1.0 / num_struct_classes)
    structures_test = np.array(structures_B, dtype=np.float32) - (1.0 / num_struct_classes)
    h5f = h5py.File(save_target, 'a')
    structures_train_pointer = h5f['structures_train']
    structures_test_pointer = h5f['structures_test']
    structures_train_pointer.resize([length_A, num_struct_classes * structure_num, 41])
    structures_test_pointer.resize([length_B, num_struct_classes * structure_num, 41])
    structures_train_pointer[length_A // 10000 * 10000:length_A] = structures_train
    structures_test_pointer[length_B // 10000 * 10000:length_B] = structures_test

    # for k in range(len(target_id_list)):


    seq_train_enc =[]
    seq_test_enc = []
    for index, i in enumerate(sequence_input):
        if (index in dic_ban) == False:
            if belong[index] == 1:
                seq_train_enc.append(i)
            elif belong[index] == 2:
                seq_test_enc.append(i)

    data_one_hot_training = np.array(seq_train_enc).astype(np.int32)
    data_one_hot_test = np.array(seq_test_enc).astype(np.int32)
    labels_training = np.array([i[0] for i in target_train], dtype=float)
    labels_test = np.array([i[0] for i in target_test], dtype=float)
    training_cases = data_one_hot_training.shape[0]
    test_cases = data_one_hot_test.shape[0]
    seq_length = data_one_hot_training.shape[1]
    train_remove = np.round(0.0005 * training_cases).astype(int)
    test_remove = np.round(0.0005 * test_cases).astype(int)
    train_ind = np.argpartition(labels_training, -train_remove)[-train_remove:]
    test_ind = np.argpartition(labels_test, -test_remove)[-test_remove:]
    train_clamp = np.min(labels_training[train_ind])
    test_clamp = np.min(labels_test[test_ind])
    labels_training[train_ind] = train_clamp
    labels_test[test_ind] = test_clamp

    # Remove this part to train only on set A

    ##############

    # save_target = "../data/rnac/npz_archives_update/" +protein_name
    h5f = h5py.File(save_target, 'a')

    h5f.create_dataset('data_one_hot_training', data=data_one_hot_training)
    h5f.create_dataset('labels_training', data=labels_training)
    h5f.create_dataset('data_one_hot_test', data=data_one_hot_test)
    h5f.create_dataset('labels_test', data=labels_test)
    h5f.create_dataset('training_cases', data=training_cases)
    h5f.create_dataset('test_cases', data=test_cases)
    h5f.create_dataset('seq_len_train', data=seq_len_A)
    h5f.create_dataset('seq_len_test', data=seq_len_B)
    h5f.create_dataset('seq_length', data=seq_length)
    h5f.create_dataset('pred_train', data=np.array(pred_A))
    h5f.create_dataset('pred_test', data=np.array(pred_B))
    h5f.create_dataset('difficult_test', data=np.array(test_difficult, dtype=np.int32))
    h5f.create_dataset('simple_test', data=np.array(test_simple, dtype=np.int32))

def load_data_rnac2009(protein_name):
    data_folder = 'data/rnac_2009/full'
    structure_folder = 'data/rnac_2009/full/structure_annotations'
    training_seqs = []
    training_scores = []
    training_structs = []
    test_structs = []
    num_struct_classes = 5
    test_seqs = []
    test_scores = []
    with open(os.path.join(data_folder, protein_name + '_data_full_A.txt'), 'r') as training_file:
        for line in training_file:
            training_scores.append(line.split('\t')[0])
            training_seqs.append(line.split('\t')[1].strip())

    with open(os.path.join(data_folder, protein_name + '_data_full_B.txt'), 'r') as test_file:
        for line in test_file:
            test_scores.append(line.split('\t')[0])
            test_seqs.append(line.split('\t')[1].strip())

    seq_len_train = max([len(seq) for seq in training_seqs])
    seq_len_test = max([len(seq) for seq in test_seqs])

    with open(os.path.join(structure_folder, protein_name + '_data_full_A_profile'), 'r') as train_struct_file:
        for line in train_struct_file:
            probs = np.ones([num_struct_classes, seq_len_train]) * (1 / num_struct_classes)
            for i in range(5):
                values_line = train_struct_file.next().strip()
                values = np.array(map(np.float32, values_line.split('\t')))
                probs[i, 0:values.shape[0]] = values
            training_structs.append(probs)
    with open(os.path.join(structure_folder, protein_name + '_data_full_B_profile'), 'r') as test_struct_file:
        for line in test_struct_file:
            probs = np.ones([num_struct_classes, seq_len_test]) * (1 / num_struct_classes)
            for i in range(5):
                values_line = test_struct_file.next().strip()
                values = np.array(map(np.float32, values_line.split('\t')))
                probs[i, 0:values.shape[0]] = values
            test_structs.append(probs)



    seq_enc = np.ones((len(training_seqs), seq_len_train, 4)) * 0.25
    for i, case in enumerate(training_seqs):
        for j, nuc in enumerate(case):
            if nuc == 'A':
                seq_enc[i, j] = np.array([1, 0, 0, 0])
            elif nuc == 'G':
                seq_enc[i, j] = np.array([0, 1, 0, 0])
            elif nuc == 'C':
                seq_enc[i, j] = np.array([0, 0, 1, 0])
            elif nuc == 'U':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
            elif nuc == 'T':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
    seq_enc -= 0.25
    data_one_hot_training = np.array(seq_enc)

    seq_enc = np.ones((len(test_seqs), seq_len_test, 4)) * 0.25
    for i, case in enumerate(test_seqs):
        for j, nuc in enumerate(case):
            if nuc == 'A':
                seq_enc[i, j] = np.array([1, 0, 0, 0])
            elif nuc == 'G':
                seq_enc[i, j] = np.array([0, 1, 0, 0])
            elif nuc == 'C':
                seq_enc[i, j] = np.array([0, 0, 1, 0])
            elif nuc == 'U':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
            elif nuc == 'T':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
    seq_enc -= 0.25
    data_one_hot_test = np.array(seq_enc)
    labels_training = np.array(training_scores, dtype=np.float32)
    labels_test = np.array(test_scores, dtype=np.float32)
    training_cases = data_one_hot_training.shape[0]
    test_cases = data_one_hot_test.shape[0]
    save_target = os.path.join('data/rnac_2009/npz_archives_update/', protein_name + '.npz')
    np.savez(save_target, data_one_hot_training=data_one_hot_training,
             labels_training=labels_training,
             data_one_hot_test=data_one_hot_test,
             labels_test=labels_test, training_cases=training_cases,
             test_cases=test_cases,
             structures_train=np.array(training_structs, np.float32),
             structures_test=np.array(test_structs, np.float32),
             seq_length=max(seq_len_train, seq_len_test))
    print("[*] Finished loading data for " + protein_name)


def load_data_clipseq(protein_name):
    data_folder = 'data/GraphProt_CLIP_sequences'
    structure_folder = 'data/GraphProt_CLIP_sequences/structure_annotations/' + protein_name
    structs = []
    num_struct_classes = 5
    seqs = []
    labels = []
    with open(os.path.join(data_folder, protein_name + '.train.positives.fa'), 'r') as pos_file:
        for line in pos_file:
            seqs.append(pos_file.next().strip())
            labels.append(1.0)
    with open(os.path.join(data_folder, protein_name + '.train.negatives.fa'), 'r') as neg_file:
        for line in neg_file:
            seqs.append(neg_file.next().strip())
            labels.append(0.0)
    seq_len = max([len(seq) for seq in seqs])

    with open(os.path.join(structure_folder, protein_name + '.train.positives_combined.txt'), 'r') as pos_struct_file:
        for line in pos_struct_file:
            probs = np.ones([num_struct_classes, seq_len]) * (1.0 / num_struct_classes)
            for i in range(5):
                values_line = pos_struct_file.next().strip()
                values = np.array(map(np.float32, values_line.split('\t')))
                probs[i, 0:values.shape[0]] = values
            structs.append(probs)
    with open(os.path.join(structure_folder, protein_name + '.train.negatives_combined.txt'), 'r') as neg_struct_file:
        for line in neg_struct_file:
            probs = np.ones([num_struct_classes, seq_len]) * (1.0 / num_struct_classes)
            for i in range(5):
                values_line = neg_struct_file.next().strip()
                values = np.array(map(np.float32, values_line.split('\t')))
                probs[i, 0:values.shape[0]] = values
            structs.append(probs)

    seq_enc = np.ones((len(seqs), seq_len, 4)) * 0.25
    for i, case in enumerate(seqs):
        for j, nuc in enumerate(case):
            if nuc == 'A' or nuc == 'a':
                seq_enc[i, j] = np.array([1, 0, 0, 0])
            elif nuc == 'G' or nuc == 'g':
                seq_enc[i, j] = np.array([0, 1, 0, 0])
            elif nuc == 'C' or nuc == 'c':
                seq_enc[i, j] = np.array([0, 0, 1, 0])
            elif nuc == 'U' or nuc == 'u':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
            elif nuc == 'T' or nuc == 't':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
    seq_enc -= 0.25
    data_one_hot = np.array(seq_enc, np.float32)
    labels_array = np.array(labels, np.float32)
    total_cases = data_one_hot.shape[0]
    structures = np.array(structs, np.float32) - (1.0/num_struct_classes)
    save_target = os.path.join('data/GraphProt_CLIP_sequences/npz_archives_update', protein_name + '.npz')
    np.savez(save_target, data_one_hot=data_one_hot,
             labels=labels_array,
             seq_length=seq_len,
             structures=structures,
             total_cases=total_cases)
    print("[*] Finished loading data for " + protein_name)


def load_data_clipseq_shorter(protein_name):
    data_folder = 'data/GraphProt_CLIP_sequences'
    structure_folder = 'data/GraphProt_CLIP_sequences/structure_annotations/' + protein_name
    structs = []
    num_struct_classes = 5
    seqs = []
    labels = []
    pos_start_pos = []
    pos_end_pos = []
    neg_start_pos = []
    neg_end_pos = []
    start_pos = []
    end_pos = []
    with open(os.path.join(data_folder, protein_name + '.train.positives.fa'), 'r') as pos_file:
        for line in pos_file:
            full_seq = pos_file.next().strip()
            bound_region = np.array([int(c.isupper()) for c in full_seq], dtype=np.int32)
            upper_case_region = np.where(bound_region == True)
            start_bound = upper_case_region[0][0]
            end_bound = upper_case_region[0][-1]
            start_pos.append(start_bound)
            end_pos.append(end_bound)
            seqs.append(full_seq)
            labels.append(1.0)
    with open(os.path.join(data_folder, protein_name + '.train.negatives.fa'), 'r') as neg_file:
        for line in neg_file:
            full_seq = neg_file.next().strip()
            bound_region = np.array([int(c.isupper()) for c in full_seq], dtype=np.int32)
            upper_case_region = np.where(bound_region == True)
            start_bound = upper_case_region[0][0]
            end_bound = upper_case_region[0][-1]
            start_pos.append(start_bound)
            end_pos.append(end_bound)
            seqs.append(full_seq)
            labels.append(0.0)
    seq_len = max([len(seq) for seq in seqs])
    with open(os.path.join(structure_folder, protein_name + '.train.positives_combined'), 'r') as pos_struct_file:
        for line in pos_struct_file:
            probs = np.ones([num_struct_classes, seq_len]) * (1.0 / num_struct_classes)
            for i in range(5):
                values_line = pos_struct_file.next().strip()
                values = np.array(map(np.float32, values_line.split('\t')))
                probs[i, 0:values.shape[0]] = values
            structs.append(probs)
    with open(os.path.join(structure_folder, protein_name + '.train.negatives_combined'), 'r') as neg_struct_file:
        for line in neg_struct_file:
            probs = np.ones([num_struct_classes, seq_len]) * (1.0 / num_struct_classes)
            for i in range(5):
                values_line = neg_struct_file.next().strip()
                values = np.array(map(np.float32, values_line.split('\t')))
                probs[i, 0:values.shape[0]] = values
            structs.append(probs)

    seq_enc = np.ones((len(seqs), seq_len, 4)) * 0.25
    for i, case in enumerate(seqs):
        for j, nuc in enumerate(case):
            if nuc == 'A' or nuc == 'a':
                seq_enc[i, j] = np.array([1, 0, 0, 0])
            elif nuc == 'G' or nuc == 'g':
                seq_enc[i, j] = np.array([0, 1, 0, 0])
            elif nuc == 'C' or nuc == 'c':
                seq_enc[i, j] = np.array([0, 0, 1, 0])
            elif nuc == 'U' or nuc == 'u':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
            elif nuc == 'T' or nuc == 't':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
    seq_enc -= 0.25
    data_one_hot = np.array(seq_enc, np.float32)
    labels_array = np.array(labels, np.float32)
    total_cases = data_one_hot.shape[0]
    structures = np.array(structs, np.float32) - (1.0 / num_struct_classes)
    save_target = os.path.join('data/GraphProt_CLIP_sequences/npz_archives_update', protein_name + '.npz')
    np.savez(save_target, data_one_hot=data_one_hot,
             labels=labels_array,
             seq_length=seq_len,
             structures=structures,
             total_cases=total_cases,
             start_pos=np.array(start_pos),
             end_pos=np.array(end_pos))
    print("[*] Finished loading data for " + protein_name)


def load_data_rnac_s(path):
    pos_file = open(os.path.join(path, 'input_seq.fa'))
    neg_file = open(os.path.join(path, 'bg_input_seq.fa'))
    seq_pos = []
    seq_neg = []
    seq_lens_pos = []
    labels_pos = []
    labels_neg = []
    seq_lens_neg = []
    struct_pos = []
    struct_neg = []
    struct_pos_file = open(os.path.join(path, 'input_seq_combined.txt'))
    struct_neg_file = open(os.path.join(path, 'bg_input_seq_combined.txt'))
    num_struct_classes = 5

    for line in pos_file:
        seq = pos_file.next().strip()
        seq_pos.append(seq)
        seq_lens_pos.append(len(seq))
        labels_pos.append(1.0)
    for line in neg_file:
        seq = neg_file.next().strip()
        seq_neg.append(seq)
        seq_lens_neg.append(len(seq))
        labels_neg.append(0.0)
    assert len(seq_pos) == len(seq_neg)

    seq_len_pad_pos = np.max(seq_lens_pos)
    seq_len_pad_neg = np.max(seq_lens_neg)
    for line in struct_pos_file:
        probs = np.ones([num_struct_classes, seq_len_pad_pos]) * (1.0 / num_struct_classes)
        for i in range(5):
            values_line = struct_pos_file.next().strip()
            values = np.array(map(float, values_line.split('\t')))
            probs[i, 0:values.shape[0]] = values
        struct_pos.append(probs)

    for line in struct_neg_file:
        probs = np.ones([num_struct_classes, seq_len_pad_pos]) * (1.0 / num_struct_classes)
        for i in range(5):
            values_line = struct_neg_file.next().strip()
            values = np.array(map(np.float32, values_line.split('\t')))
            probs[i, 0:values.shape[0]] = values
        struct_neg.append(probs)

    assert len(struct_neg) == len(struct_pos)
    assert seq_len_pad_pos == seq_len_pad_neg
    assert len(struct_neg) == len(seq_neg)

    seq_enc_pos = np.ones((len(seq_pos), seq_len_pad_pos, 4)) * 0.25
    for i, case in enumerate(seq_pos):
        for j, nuc in enumerate(case):
            if nuc == 'A':
                seq_enc_pos[i, j] = np.array([1, 0, 0, 0])
            elif nuc == 'G':
                seq_enc_pos[i, j] = np.array([0, 1, 0, 0])
            elif nuc == 'C':
                seq_enc_pos[i, j] = np.array([0, 0, 1, 0])
            elif nuc == 'U':
                seq_enc_pos[i, j] = np.array([0, 0, 0, 1])

    seq_enc_neg = np.ones((len(seq_neg), seq_len_pad_neg, 4)) * 0.25
    for i, case in enumerate(seq_neg):
        for j, nuc in enumerate(case):
            if nuc == 'A':
                seq_enc_neg[i, j] = np.array([1, 0, 0, 0])
            elif nuc == 'G':
                seq_enc_neg[i, j] = np.array([0, 1, 0, 0])
            elif nuc == 'C':
                seq_enc_neg[i, j] = np.array([0, 0, 1, 0])
            elif nuc == 'U':
                seq_enc_neg[i, j] = np.array([0, 0, 0, 1])
    seq_enc_pos -= 0.25
    seq_enc_neg -= 0.25
    struct_pos = np.array(struct_pos, dtype=np.float32) - (1.0 / num_struct_classes)
    struct_neg = np.array(struct_neg, dtype=np.float32) - (1.0 / num_struct_classes)
    train_size = int(0.9 * len(seq_pos))
    test_size = len(seq_pos) - train_size

    data_one_hot_training = []
    data_one_hot_test = []
    labels_training = []
    labels_test = []
    structures_train = []
    structures_test = []
    seq_len_train = []
    seq_len_test = []

    for i in range(train_size):
        data_one_hot_training.append(seq_enc_pos[i])
        data_one_hot_training.append(seq_enc_neg[i])
        structures_train.append(struct_pos[i])
        structures_train.append(struct_neg[i])
        seq_len_train.append(seq_lens_pos[i])
        seq_len_train.append(seq_lens_neg[i])
        labels_training.append(labels_pos[i])
        labels_training.append(labels_neg[i])

    for i in range(train_size, len(seq_pos)):
        data_one_hot_test.append(seq_enc_pos[i])
        data_one_hot_test.append(seq_enc_neg[i])
        structures_test.append(struct_pos[i])
        structures_test.append(struct_neg[i])
        seq_len_test.append(seq_lens_pos[i])
        seq_len_test.append(seq_lens_neg[i])
        labels_test.append(labels_pos[i])
        labels_test.append(labels_neg[i])

    if not (os.path.exists(os.path.join(path, 'npz_archives_update'))):
        os.makedirs(os.path.join(path, 'npz_archives_update'))
    save_path = os.path.join(path, 'npz_archives_update', 'SLBP_rnacs.npz')

    np.savez(save_path,
             data_one_hot_training=data_one_hot_training,
             labels_training=labels_training,
             data_one_hot_test=data_one_hot_test,
             labels_test=labels_test, training_cases=train_size,
             test_cases=test_size,
             structures_train=structures_train,
             structures_test=structures_test,
             seq_len_train=seq_len_train,
             seq_len_test=seq_len_test,
             seq_length=seq_len_pad_pos,
             )

def load_data(protein_name):
    if 'RNCMPT' in protein_name:
        if not (os.path.isfile('../data/rnac/npz_archives_update/' + str(protein_name) + '.h5')):
            print("[!] Processing input for " + protein_name)
            load_data_rnac2013(protein_name)
        return h5py.File('../data/rnac/npz_archives_update/' + str(protein_name) + '.h5', 'r')
    elif protein_name == 'SLBP_rnacs':
        if not (os.path.isfile('../data/rnac_s/npz_archives_update/' + str(protein_name)+'.npz')):
            print("[!] Processing input for " + protein_name)
            load_data_rnac_s('../data/rnac_s')
        return np.load(('../data/rnac_s/npz_archives_update/' + str(protein_name)+'.npz'))
    else:
        if not (os.path.isfile('../data/rnac_2009/npz_archives_update/' + str(protein_name) + '.npz')):
            print("[!] Processing input for " + protein_name)
            load_data_rnac2009(protein_name)
        return np.load('../data/rnac_2009/npz_archives_update/' + str(protein_name) + '.npz')


def generate_configs_CNN(num_calibrations, k_mer = 5):
    configs = []
    # for eta in [0.01, 0.001, 0.0001]:
    #     for lam in [0.001, 0.0001, 0.01, 0.00001]:
    #         for embedding_size in [10, 18, 20, 26, 30]:
    #             for filter_length in [5]:
    #
    #
    # for eta in [0.001]:
    #     for lam in [0.0001]:
    #         for embedding_size in [10]:
    #             for filter_length in [16]:
    for eta in [0.001, 0.0001]:
        for lam in [0.001, 0.005, 0.0001]:
            for embedding_size in [10, 18, 20, 26, 30]:
                for filter_length in [16]:
                    minib = 100
                    test_interval = 10
                    num_conv_layers = 2
                    strides = np.random.choice([1], size=num_conv_layers)
                    pool_windows = np.random.choice([1], size=num_conv_layers)
                    final_pool = np.random.choice(['max', 'avg', 'max_avg'])
                    batchnorm = np.random.choice([True, False])
                    filter_lengths = [16 // (2 ** i) for i in range(num_conv_layers)]
                    filter_lengths[0] = filter_length
                    num_filters = [16 * (i + 1) for i in range(num_conv_layers)]
                    init_scale = 0.00001
                    temp_config = {'eta_model': eta, 'lam_model': lam, 'minib': minib,
                                   'test_interval': test_interval, 'filter_lengths': filter_lengths, 'num_filters': num_filters,
                                   'num_conv_layers': num_conv_layers, 'strides': strides,
                                   'pool_windows': pool_windows,
                                   'batchnorm': batchnorm,
                                   'final_pool': final_pool,
                                   'init_scale': init_scale,
                                   "k_mer": k_mer,
                                   "embedding_size": embedding_size}

                    configs.append(temp_config)
    return configs


def generate_configs_RNN(num_calibrations, k_mer = 5):
    configs = []
    for eta in [0.01, 0.001, 0.0001]:
        for lam in [0.001, 0.0001, 0.01, 0.00001, 0.1]:
            for embedding_size in [10, 14, 18, 20, 24, 26]:
                for num_motifs in [8, 16]:
                    minib = 1000
                    momentum = 0.98
                    test_interval = 10
                    motif_len = 16
                    lstm_size = 5
                    init_scale = 0.00001
                    temp_config = {'eta_model': eta, 'momentum_model': momentum, 'lam_model': lam, 'minib': minib,
                                   'test_interval': test_interval, 'motif_len': motif_len,
                                   'lstm_size': lstm_size,
                                   'num_motifs': num_motifs, 'init_scale': init_scale, 'k_mer':k_mer, 'embedding_size':embedding_size}

                    configs.append(temp_config)
    return configs


def generate_configs(num_calibrations, model_type, k_mer = 5):
    if model_type=='CNN':
        return generate_configs_CNN(num_calibrations)
    if model_type=='CNN_struct':
        return generate_configs_CNN(num_calibrations, k_mer)
    if model_type=='CNN_attention':
        return generate_configs_CNN(num_calibrations, k_mer)
    if model_type=='RNN_struct':
        return generate_configs_RNN(num_calibrations, k_mer)
    if model_type == 'RNN':
        return generate_configs_RNN(num_calibrations)
    if model_type == 'RNN_struct_track':
        return generate_configs_RNN(num_calibrations)


def summarize(train_config):
    save_path = train_config.get('result_dir','results')
    protein_list = ['Fusip', 'HuR', 'PTB', 'RBM4', 'SF2', 'SLM2', 'U1A', 'VTS1', 'YB1',
                    'RNCMPT00100',
                    'RNCMPT00101',
                    'RNCMPT00102',
                    'RNCMPT00103',
                    'RNCMPT00104',
                    'RNCMPT00105',
                    'RNCMPT00106',
                    'RNCMPT00107',
                    'RNCMPT00108',
                    'RNCMPT00109',
                    'RNCMPT00010',
                    'RNCMPT00110',
                    'RNCMPT00111',
                    'RNCMPT00112',
                    'RNCMPT00113',
                    'RNCMPT00114',
                    'RNCMPT00116',
                    'RNCMPT00117',
                    'RNCMPT00118',
                    'RNCMPT00119',
                    'RNCMPT00011',
                    'RNCMPT00120',
                    'RNCMPT00121',
                    'RNCMPT00122',
                    'RNCMPT00123',
                    'RNCMPT00124',
                    'RNCMPT00126',
                    'RNCMPT00127',
                    'RNCMPT00129',
                    'RNCMPT00012',
                    'RNCMPT00131',
                    'RNCMPT00132',
                    'RNCMPT00133',
                    'RNCMPT00134',
                    'RNCMPT00136',
                    'RNCMPT00137',
                    'RNCMPT00138',
                    'RNCMPT00139',
                    'RNCMPT00013',
                    'RNCMPT00140',
                    'RNCMPT00141',
                    'RNCMPT00142',
                    'RNCMPT00143',
                    'RNCMPT00144',
                    'RNCMPT00145',
                    'RNCMPT00146',
                    'RNCMPT00147',
                    'RNCMPT00148',
                    'RNCMPT00149',
                    'RNCMPT00014',
                    'RNCMPT00150',
                    'RNCMPT00151',
                    'RNCMPT00152',
                    'RNCMPT00153',
                    'RNCMPT00154',
                    'RNCMPT00155',
                    'RNCMPT00156',
                    'RNCMPT00157',
                    'RNCMPT00158',
                    'RNCMPT00159',
                    'RNCMPT00015',
                    'RNCMPT00160',
                    'RNCMPT00161',
                    'RNCMPT00162',
                    'RNCMPT00163',
                    'RNCMPT00164',
                    'RNCMPT00165',
                    'RNCMPT00166',
                    'RNCMPT00167',
                    'RNCMPT00168',
                    'RNCMPT00169',
                    'RNCMPT00016',
                    'RNCMPT00170',
                    'RNCMPT00171',
                    'RNCMPT00172',
                    'RNCMPT00173',
                    'RNCMPT00174',
                    'RNCMPT00175',
                    'RNCMPT00176',
                    'RNCMPT00177',
                    'RNCMPT00178',
                    'RNCMPT00179',
                    'RNCMPT00017',
                    'RNCMPT00180',
                    'RNCMPT00181',
                    'RNCMPT00182',
                    'RNCMPT00183',
                    'RNCMPT00184',
                    'RNCMPT00185',
                    'RNCMPT00186',
                    'RNCMPT00187',
                    'RNCMPT00018',
                    'RNCMPT00197',
                    'RNCMPT00199',
                    'RNCMPT00019',
                    'RNCMPT00001',
                    'RNCMPT00200',
                    'RNCMPT00202',
                    'RNCMPT00203',
                    'RNCMPT00205',
                    'RNCMPT00206',
                    'RNCMPT00209',
                    'RNCMPT00020',
                    'RNCMPT00212',
                    'RNCMPT00215',
                    'RNCMPT00216',
                    'RNCMPT00217',
                    'RNCMPT00218',
                    'RNCMPT00219',
                    'RNCMPT00021',
                    'RNCMPT00220',
                    'RNCMPT00223',
                    'RNCMPT00224',
                    'RNCMPT00225',
                    'RNCMPT00226',
                    'RNCMPT00228',
                    'RNCMPT00229',
                    'RNCMPT00022',
                    'RNCMPT00230',
                    'RNCMPT00232',
                    'RNCMPT00234',
                    'RNCMPT00235',
                    'RNCMPT00236',
                    'RNCMPT00237',
                    'RNCMPT00238',
                    'RNCMPT00239',
                    'RNCMPT00023',
                    'RNCMPT00240',
                    'RNCMPT00241',
                    'RNCMPT00245',
                    'RNCMPT00246',
                    'RNCMPT00248',
                    'RNCMPT00249',
                    'RNCMPT00024',
                    'RNCMPT00251',
                    'RNCMPT00252',
                    'RNCMPT00253',
                    'RNCMPT00254',
                    'RNCMPT00255',
                    'RNCMPT00256',
                    'RNCMPT00257',
                    'RNCMPT00258',
                    'RNCMPT00259',
                    'RNCMPT00025',
                    'RNCMPT00261',
                    'RNCMPT00262',
                    'RNCMPT00263',
                    'RNCMPT00265',
                    'RNCMPT00268',
                    'RNCMPT00269',
                    'RNCMPT00026',
                    'RNCMPT00270',
                    'RNCMPT00272',
                    'RNCMPT00273',
                    'RNCMPT00274',
                    'RNCMPT00278',
                    'RNCMPT00279',
                    'RNCMPT00027',
                    'RNCMPT00280',
                    'RNCMPT00281',
                    'RNCMPT00282',
                    'RNCMPT00283',
                    'RNCMPT00284',
                    'RNCMPT00285',
                    'RNCMPT00287',
                    'RNCMPT00288',
                    'RNCMPT00289',
                    'RNCMPT00028',
                    'RNCMPT00291',
                    'RNCMPT00029',
                    'RNCMPT00002',
                    'RNCMPT00031',
                    'RNCMPT00032',
                    'RNCMPT00033',
                    'RNCMPT00034',
                    'RNCMPT00035',
                    'RNCMPT00036',
                    'RNCMPT00037',
                    'RNCMPT00038',
                    'RNCMPT00039',
                    'RNCMPT00003',
                    'RNCMPT00040',
                    'RNCMPT00041',
                    'RNCMPT00042',
                    'RNCMPT00043',
                    'RNCMPT00044',
                    'RNCMPT00045',
                    'RNCMPT00046',
                    'RNCMPT00047',
                    'RNCMPT00049',
                    'RNCMPT00004',
                    'RNCMPT00050',
                    'RNCMPT00051',
                    'RNCMPT00052',
                    'RNCMPT00053',
                    'RNCMPT00054',
                    'RNCMPT00055',
                    'RNCMPT00056',
                    'RNCMPT00057',
                    'RNCMPT00058',
                    'RNCMPT00059',
                    'RNCMPT00005',
                    'RNCMPT00060',
                    'RNCMPT00061',
                    'RNCMPT00062',
                    'RNCMPT00063',
                    'RNCMPT00064',
                    'RNCMPT00065',
                    'RNCMPT00066',
                    'RNCMPT00067',
                    'RNCMPT00068',
                    'RNCMPT00069',
                    'RNCMPT00006',
                    'RNCMPT00070',
                    'RNCMPT00071',
                    'RNCMPT00072',
                    'RNCMPT00073',
                    'RNCMPT00074',
                    'RNCMPT00075',
                    'RNCMPT00076',
                    'RNCMPT00077',
                    'RNCMPT00078',
                    'RNCMPT00079',
                    'RNCMPT00007',
                    'RNCMPT00080',
                    'RNCMPT00081',
                    'RNCMPT00082',
                    'RNCMPT00083',
                    'RNCMPT00084',
                    'RNCMPT00085',
                    'RNCMPT00086',
                    'RNCMPT00087',
                    'RNCMPT00088',
                    'RNCMPT00089',
                    'RNCMPT00008',
                    'RNCMPT00090',
                    'RNCMPT00091',
                    'RNCMPT00093',
                    'RNCMPT00094',
                    'RNCMPT00095',
                    'RNCMPT00096',
                    'RNCMPT00097',
                    'RNCMPT00099',
                    'RNCMPT00009']
    print("[*] Updating result summary")
    model_list = ['CNN_struct', 'CNN', 'RNN_struct', 'RNN', 'RNN_struct_track', 'CNN_attention']
    result_file = open(os.path.join(save_path,'summary.tsv'), 'w')
    heading = 'Protein\t' + '\t'.join(model_list) + '\n'
    result_file.write(heading)
    count = 0
    for protein in protein_list:
        result_file.write(protein)
        for model in model_list:
            if os.path.isfile(os.path.join( save_path,protein + '_' + model + '.npz')):
                read_file = np.load(os.path.join( save_path,protein + '_' + model + '.npz'))
                result_file.write('\t' + str(read_file['pearson']))
                count += 1
            else:
                result_file.write('\t')
        result_file.write('\n')
    print("[*] Update complete, %d record(s) updated" % (count))
