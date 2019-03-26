import numpy as np
import tensorflow as tf
import h5py
import deepbind_model.utils_update as utils


def calibrate_model(train_config):
    best_config = utils.load_calibration(train_config)
    if not(best_config):
        print("[!] Model for %s %s  is not pre-calibrated!"%(train_config['protein'],train_config['model_type']))
    elif not(train_config['recalibrate']):
        return best_config
    else:
        print("[!] Recalibrating %s %s model"%(train_config['protein'],train_config['model_type']))

    target_protein = train_config['protein']
    num_calibrations = train_config.get('num_calibrations',5)
    model_type = train_config['model_type']
    inf = utils.load_data(target_protein)

    input_configuration = utils.input_config('large')
    configs = utils.generate_configs(num_calibrations, model_type, train_config['k_mer'])
    # print(configs)
    inputs = []
    num_calibrations = len(configs)
    folds = train_config['cv_folds']
    max_minib = train_config['minib']
    for i in range(1,folds+1):
        inputs.append(utils.model_input(input_configuration, inf, model_type, validation=True, fold_id=i))
    # print(inputs[fold].shape)
    epochs = train_config['hp_epochs']
    val_cost = np.zeros([folds, num_calibrations, epochs])
    val_pearson = np.zeros([folds, num_calibrations, epochs])

    max_per_round = train_config['Max_per_round']

    num_round = ((len(configs) - 1) // max_per_round) + 1

    if max_per_round == 1:
        num_round -= 1

    for fold in range(folds):
        print("[*] Evaluating fold %d" % (fold+1))
        with tf.Graph().as_default():
            for t in range(num_round):
                models = []
                for i in range(max_per_round):
                    if t * max_per_round + i >= len(configs):
                        break
                # for i in range(2):
                    with tf.variable_scope('model' + str(t * max_per_round + i)):
                        models.append(utils.model(configs[t * max_per_round + i], inputs[fold], model_type))
                config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
                config.gpu_options.allow_growth = True
                with tf.Session(config=config) as session:
                    session.run(tf.global_variables_initializer())
                    train_config['minib'] = 100
                    upper_bound = min(len(configs), (t + 1) * max_per_round)
                    (val_cost[fold, t * max_per_round: upper_bound, :],
                     val_pearson[fold, t * max_per_round: upper_bound, :],_,_) = \
                        utils.train_model_parallel(session, train_config, models, inputs[fold],epochs=epochs, early_stop=False, validation=True)

    val_cost = np.mean(np.transpose(val_cost, [1, 2, 0]), axis=2)
    val_pearson = np.mean(np.transpose(val_pearson, [1, 2, 0]), axis=2)
    best_calibration_idx = int(np.argmin(np.min(val_cost, axis=1)))
    best_calibration = configs[best_calibration_idx]
    best_cost = np.min(val_cost[best_calibration_idx])
    best_pearson = np.max(val_pearson[best_calibration_idx])
    best_calibration['pearson'] = best_pearson
    best_calibration['cost'] = best_cost
    best_calibration['round'] = int(np.argmin(np.min(val_cost, axis=0))) + 1
    utils.save_calibration(train_config, best_calibration)
    train_config['minib'] = max_minib
    return best_calibration
