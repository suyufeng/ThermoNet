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

class StructInput(object):

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
            self.training_lens = np.array(inf['seq_len_train'], np.int32)[0:self.training_cases]
            self.test_lens = np.array(inf['seq_len_test'], np.int32)[0:self.test_cases]


        self.seq_length = int(seq_length)
        self.training_cases = self.training_data.shape[0]
        self.test_cases = self.test_data.shape[0]
        self.difficult_label = difficult_test
        self.simple_label = simple_test

        self.inf = inf

def model_input(input_config, inf, model, validation=False, fold_id=1):
    if model == 'ThermoNet':
        return StructInput(input_config, inf, validation, fold_id)

import math

class ThermoNet(object):
    def __init__(self, config, input_):
        self._config = config
        eta_model = config['eta_model']
        lam_model = config['lam_model']
        self.motif_len = config['filter_lengths'][0]
        self.num_motifs = config['num_filters'][0]
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


def model(config, input, model_type):
    if model_type == 'ThermoNet':
        return ThermoNet(config, input)

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

    learn_rate = 1.
    if epoch >= 5:
        learn_rate *= 0.1

    for step in range(0, Nbatch_train):
        fetches = {}
        feed_dict = {}
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
                    fetches["predictions" + str(i)] = model.predict_op
        else:
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
                fetches["predictions" + str(i)] = model.predict_op
        if train:
            for t in range(ENSEMBLE_NUMBER):
                fetches["eval_op"] = model.train_op_list[t]
                vals = session.run(fetches, feed_dict)
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
            else:
                pearson_test[j, :] = stats.pearsonr(input_data.test_labels, test_scores[j, :])
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


def train_model_parallel(session, train_config, models, input_data,epochs, early_stop = False, savedir=None,saver=None, validation = False):
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
    print("pass")

    random_list = np.random.permutation(t.shape[0])
    t = t[random_list, :]
    data_flag = data_flag[random_list, :]
    average_struct = average_struct[random_list, :]
    sequence_input = np.stack(
        [np.load("../data/rnac/embedding_right/embedding" + str(i) + ".txt.npy") for i in range(1, 10)], axis=2)
    sequence_input = sequence_input[random_list, :, :]
    predict_input = np.load("../data/rnac/pred_list_5.npy")
    predict_input = predict_input[random_list, :]

    seq_list = []
    target_list = []

    for line_seq in infile_seq_tmp:
        line_target = infile_target_tmp.readline()
        seq_list.append(line_seq)
        target_list.append(line_target)


    infile_seq_random = open('../data/rnac/sequences_random' + protein_name + '.tsv', 'w')
    infile_target_random = open('../data/rnac/targets_random' + protein_name + '.tsv', 'w')


    for i in range(len(seq_list)):
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
    for i in range(len(length)):
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


def load_data(protein_name):
    if 'RNCMPT' in protein_name:
        if not (os.path.isfile('../data/rnac/npz_archives_update/' + str(protein_name) + '.h5')):
            print("[!] Processing input for " + protein_name)
            load_data_rnac2013(protein_name)
        return h5py.File('../data/rnac/npz_archives_update/' + str(protein_name) + '.h5', 'r')

def generate_configs_CNN(num_calibrations, k_mer = 5):
    configs = []
    for eta in [0.001, 0.0001, 0.00001]:
        for lam in [0.001, 0.0001, 0.00001]:
            for embedding_size in [10, 20, 30]:
                for filter_length in [16, 64]:
                    for k_mer in [2, 3, 4, 5]:
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

def generate_configs(num_calibrations, model_type, k_mer = 5):
    if model_type=='ThermoNet':
        return generate_configs_CNN(num_calibrations, k_mer)
