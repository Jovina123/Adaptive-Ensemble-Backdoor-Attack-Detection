#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang


import h5py
import numpy as np
import tensorflow as tf
from keras.preprocessing import image


def dump_image(x, filename, format):
    img = image.array_to_img(x, scale=False)
    img.save(filename, format)
    return


def fix_gpu_memory(mem_fraction=1):
    if not tf.__version__.startswith('1'):
        # TF2 / Keras 3 (e.g. Colab): no tf.Session / K.set_session API.
        # GPU memory growth is set differently in TF2; do that instead and
        # skip session setup entirely, since eager execution needs no session.
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
        return None

    import tensorflow.compat.v1 as tf1
    import keras.backend as K

    gpu_options = tf1.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
    tf_config = tf1.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    init_op = tf1.global_variables_initializer()
    sess = tf1.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)

    return sess


def load_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset
