#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline glue code used by the dashboard (app.py).

Wraps the existing project code (utils_backdoor, injection/injection_utils,
visualizer.Visualizer, mad_outlier_detection) into three operations that the
dashboard can trigger:

    1. train_model(...)   -> optionally poisons a clean dataset on the fly
                              (or trains clean), saves a .h5 model
    2. run_detection(...) -> reverse-engineers a trigger per class with
                              Visualizer, saves pattern/mask/fusion images,
                              then runs MAD outlier detection over the
                              per-class mask L1 norms
    3. dataset_info(...)  -> quick introspection of an uploaded .h5 file

Every long-running function takes a `log` callback (str -> None) and a
`progress` callback (done:int, total:int, stage:str -> None) so the caller
(a background thread in app.py) can stream status to the browser.
"""

import os
import sys
import random

import numpy as np
import h5py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
INJECTION_DIR = os.path.join(PROJECT_ROOT, 'injection')
if INJECTION_DIR not in sys.path:
    sys.path.insert(0, INJECTION_DIR)

import utils_backdoor  # noqa: E402
from injection_utils import construct_mask_box  # noqa: E402


def injection_func(mask, pattern, adv_img):
    """Blend a trigger pattern into an image according to its mask.
    (Same definition as injection/gtsrb_injection_example.py.)"""
    return mask * pattern + (1 - mask) * adv_img

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'webapp', 'results')

for d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    if not os.path.exists(d):
        os.makedirs(d)


def _noop_log(msg):
    pass


def _noop_progress(done, total, stage=''):
    pass


# --------------------------------------------------------------------------- #
# TensorFlow / Keras session handling
#
# This project pins TF 1.10 / Keras 2.2.2 (graph mode). Each job gets its own
# fresh graph + session so that training/detection jobs never leak Keras
# global state into one another when run back-to-back from the Flask app.
# --------------------------------------------------------------------------- #

# Max CPU threads TF is allowed to use. Override with the AEBAD_CPU_THREADS
# env var (e.g. `set AEBAD_CPU_THREADS=1` before starting the server) if it's
# still too heavy. Defaults to 2, which keeps training/detection usable
# without pinning every core.
CPU_THREADS = int(os.environ.get('AEBAD_CPU_THREADS', '2'))


def new_session():
    import tensorflow as tf
    import keras.backend as K

    config = tf.ConfigProto(
        intra_op_parallelism_threads=CPU_THREADS,
        inter_op_parallelism_threads=CPU_THREADS,
        device_count={'CPU': 1},
    )
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph, config=config)
        K.set_session(sess)
    return graph, sess


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #

def dataset_info(path):
    """Return shapes/keys of an uploaded .h5 dataset without loading it fully."""
    info = {'path': path, 'keys': [], 'shapes': {}, 'num_classes': None}
    with h5py.File(path, 'r') as hf:
        for name in hf:
            info['keys'].append(name)
            info['shapes'][name] = list(hf[name].shape)
    for y_key in ('Y_train', 'Y_test'):
        if y_key in info['shapes'] and len(info['shapes'][y_key]) == 2:
            info['num_classes'] = info['shapes'][y_key][1]
            break
    return info


def _load_keys(path, keys):
    return utils_backdoor.load_dataset(path, keys=keys)


# --------------------------------------------------------------------------- #
# Model definition (same architecture as injection/gtsrb_injection_example.py,
# generalised to an arbitrary input shape / class count so it also works with
# datasets other than GTSRB).
# --------------------------------------------------------------------------- #

def build_model(input_shape, num_classes, base=32, dense=512):
    import keras
    from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    from keras.models import Sequential

    model = Sequential()
    model.add(Conv2D(base, (3, 3), padding='same',
                      input_shape=input_shape, activation='relu'))
    model.add(Conv2D(base, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(base * 2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 4, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(base * 4, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.Adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    return model


class _PoisonGenerator(object):
    """Yields batches where a fraction of samples carry an injected trigger."""

    def __init__(self, X, Y, target_ls, pattern_dict, inject_ratio,
                 batch_size, num_classes):
        self.X = X
        self.Y = Y
        self.target_ls = target_ls
        self.pattern_dict = pattern_dict
        self.inject_ratio = inject_ratio
        self.batch_size = batch_size
        self.num_classes = num_classes

    def _infect(self, img, tgt):
        import keras
        mask, pattern = random.choice(self.pattern_dict[tgt])
        adv_img = injection_func(mask, pattern, np.copy(img))
        return adv_img, keras.utils.to_categorical(tgt, num_classes=self.num_classes)

    def __call__(self):
        batch_X, batch_Y = [], []
        n = len(self.Y)
        while True:
            idx = random.randrange(0, n)
            cur_x, cur_y = self.X[idx], self.Y[idx]
            if random.uniform(0, 1) < self.inject_ratio:
                tgt = random.choice(self.target_ls)
                cur_x, cur_y = self._infect(cur_x, tgt)
            batch_X.append(cur_x)
            batch_Y.append(cur_y)
            if len(batch_Y) == self.batch_size:
                yield np.array(batch_X), np.array(batch_Y)
                batch_X, batch_Y = [], []


def poison_dataset_to_file(src_path, dst_path, target_ls, pattern_size,
                            margin, poison_ratio, log=_noop_log):
    """
    Non-generator variant: produce a *static* poisoned copy of a clean .h5
    dataset (used when the user wants a poisoned dataset file they can
    re-use/download, rather than on-the-fly poisoning during training).
    """
    dataset = _load_keys(src_path, ['X_train', 'Y_train', 'X_test', 'Y_test'])
    X_train, Y_train = dataset['X_train'], dataset['Y_train']
    X_test, Y_test = dataset['X_test'], dataset['Y_test']

    input_shape = X_train.shape[1:]
    num_classes = Y_train.shape[1]
    pattern_dict = construct_mask_box(target_ls=target_ls, image_shape=input_shape,
                                       pattern_size=pattern_size, margin=margin)

    import keras

    def poison_split(X, Y, ratio):
        X = np.copy(X)
        Y = np.copy(Y)
        n_poison = int(len(Y) * ratio)
        idxs = np.random.choice(len(Y), n_poison, replace=False)
        for idx in idxs:
            tgt = random.choice(target_ls)
            mask, pattern = random.choice(pattern_dict[tgt])
            X[idx] = injection_func(mask, pattern, X[idx])
            Y[idx] = keras.utils.to_categorical(tgt, num_classes=num_classes)
        return X, Y

    log('Poisoning training split (%d samples, ratio=%.2f)...' % (len(Y_train), poison_ratio))
    X_train_p, Y_train_p = poison_split(X_train, Y_train, poison_ratio)
    log('Poisoning test split (%d samples, ratio=%.2f)...' % (len(Y_test), poison_ratio))
    X_test_p, Y_test_p = poison_split(X_test, Y_test, poison_ratio)

    with h5py.File(dst_path, 'w') as hf:
        hf.create_dataset('X_train', data=X_train_p)
        hf.create_dataset('Y_train', data=Y_train_p)
        hf.create_dataset('X_test', data=X_test_p)
        hf.create_dataset('Y_test', data=Y_test_p)

    log('Poisoned dataset written to %s' % dst_path)
    return dst_path


def train_model(dataset_path, model_name, mode='clean', target_labels=None,
                 pattern_size=4, margin=1, inject_ratio=0.2, epochs=10,
                 base=32, dense=512, log=_noop_log, progress=_noop_progress):
    """
    mode:
      'clean'  - train on the dataset as-is, no poisoning
      'poison' - poison on the fly during training (backdoor injection)
    Returns dict with model_path + final metrics.
    """
    import keras

    graph, sess = new_session()
    with graph.as_default():
        with sess.as_default():
            log('Loading dataset %s' % dataset_path)
            dataset = _load_keys(dataset_path, ['X_train', 'Y_train', 'X_test', 'Y_test'])
            X_train = np.array(dataset['X_train'], dtype='float32')
            Y_train = np.array(dataset['Y_train'], dtype='float32')
            X_test = np.array(dataset['X_test'], dtype='float32')
            Y_test = np.array(dataset['Y_test'], dtype='float32')

            input_shape = X_train.shape[1:]
            num_classes = Y_train.shape[1]
            log('X_train %s, Y_train %s, %d classes' %
                (str(X_train.shape), str(Y_train.shape), num_classes))

            model = build_model(input_shape, num_classes, base=base, dense=dense)

            batch_size = 32
            model_path = os.path.join(MODEL_DIR, model_name)

            if mode == 'poison':
                if not target_labels:
                    target_labels = [0]
                target_labels = [int(t) for t in target_labels]
                log('Poison mode. target labels=%s pattern_size=%d margin=%d inject_ratio=%.2f' %
                    (target_labels, pattern_size, margin, inject_ratio))
                pattern_dict = construct_mask_box(
                    target_ls=target_labels, image_shape=input_shape,
                    pattern_size=pattern_size, margin=margin)

                gen = _PoisonGenerator(X_train, Y_train, target_labels, pattern_dict,
                                        inject_ratio, batch_size, num_classes)()
                test_gen = _PoisonGenerator(X_test, Y_test, target_labels, pattern_dict,
                                             1.0, batch_size, num_classes)()

                steps_per_epoch = max(1, len(Y_train) // batch_size)

                class _Cb(keras.callbacks.Callback):
                    def on_epoch_end(self_, epoch, logs=None):
                        _, clean_acc = model.evaluate(X_test, Y_test, verbose=0)
                        _, atk_acc = model.evaluate_generator(test_gen, steps=50, verbose=0)
                        log('epoch %d/%d - clean_acc=%.4f - backdoor_success_rate=%.4f' %
                            (epoch + 1, epochs, clean_acc, atk_acc))
                        progress(epoch + 1, epochs, 'training')

                model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                     verbose=0, callbacks=[_Cb()])

                loss, acc = model.evaluate(X_test, Y_test, verbose=0)
                loss, backdoor_acc = model.evaluate_generator(test_gen, steps=100, verbose=0)
                log('Final Test Accuracy: %.4f | Final Backdoor Success Rate: %.4f' %
                    (acc, backdoor_acc))
                metrics = {'test_accuracy': float(acc), 'backdoor_success_rate': float(backdoor_acc)}
            else:
                log('Clean training mode.')

                class _Cb(keras.callbacks.Callback):
                    def on_epoch_end(self_, epoch, logs=None):
                        log('epoch %d/%d - loss=%.4f - acc=%.4f - val_loss=%.4f - val_acc=%.4f' %
                            (epoch + 1, epochs, logs.get('loss', 0), logs.get('acc', 0),
                             logs.get('val_loss', 0), logs.get('val_acc', 0)))
                        progress(epoch + 1, epochs, 'training')

                model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                          validation_data=(X_test, Y_test), verbose=0, callbacks=[_Cb()])
                loss, acc = model.evaluate(X_test, Y_test, verbose=0)
                log('Final Test Accuracy: %.4f' % acc)
                metrics = {'test_accuracy': float(acc)}

            if os.path.exists(model_path):
                os.remove(model_path)
            model.save(model_path)
            log('Model saved to %s' % model_path)

    return {'model_path': model_path, 'input_shape': list(input_shape),
            'num_classes': int(num_classes), 'metrics': metrics}


# --------------------------------------------------------------------------- #
# Detection: reverse-engineer a trigger per class (Visualizer), then flag
# outliers via MAD, exactly as gtsrb_visualize_example.py + mad_outlier_detection.py
# --------------------------------------------------------------------------- #

def outlier_detection(l1_norm_list, labels):
    consistency_constant = 1.4826
    l1_norm_list = np.array(l1_norm_list, dtype='float64')
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    mad = mad if mad > 0 else 1e-9
    anomaly_index = np.abs(np.min(l1_norm_list) - median) / mad

    flagged = []
    for i, y_label in enumerate(labels):
        if l1_norm_list[i] > median:
            continue
        score = np.abs(l1_norm_list[i] - median) / mad
        if score > 2:
            flagged.append({'label': int(y_label), 'l1_norm': float(l1_norm_list[i]),
                             'anomaly_score': float(score)})
    flagged = sorted(flagged, key=lambda x: x['l1_norm'])
    return {'median': float(median), 'mad': float(mad),
            'anomaly_index': float(anomaly_index), 'flagged': flagged}


def run_detection(model_path, dataset_path, run_id, steps=200, batch_size=32,
                   lr=0.1, init_cost=1e-3, patience=5, cost_multiplier=2,
                   attack_succ_threshold=0.99, log=_noop_log, progress=_noop_progress):
    from keras.models import load_model
    from keras.preprocessing.image import ImageDataGenerator
    from visualizer import Visualizer

    graph, sess = new_session()
    result_dir = os.path.join(RESULTS_DIR, run_id)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with graph.as_default():
        with sess.as_default():
            log('Loading dataset %s' % dataset_path)
            dataset = _load_keys(dataset_path, ['X_test', 'Y_test'])
            X_test = np.array(dataset['X_test'], dtype='float32')
            Y_test = np.array(dataset['Y_test'], dtype='float32')
            input_shape = X_test.shape[1:]
            num_classes = Y_test.shape[1]
            img_color = input_shape[2] if len(input_shape) == 3 else 1

            log('Loading model %s' % model_path)
            model = load_model(model_path)

            datagen = ImageDataGenerator()
            gen = datagen.flow(X_test, Y_test, batch_size=batch_size)

            mini_batch = max(1, (1000 // batch_size))

            visualizer = Visualizer(
                model, intensity_range='raw', regularization='l1',
                input_shape=input_shape, init_cost=init_cost, steps=steps,
                lr=lr, num_classes=num_classes, mini_batch=mini_batch,
                upsample_size=1, attack_succ_threshold=attack_succ_threshold,
                patience=patience, cost_multiplier=cost_multiplier,
                img_color=img_color, batch_size=batch_size, verbose=0,
                save_last=False, early_stop=True, early_stop_threshold=1.0,
                early_stop_patience=5 * patience)

            mask_shape = np.ceil(np.array(input_shape[0:2], dtype=float)).astype(int)

            l1_norms = []
            per_label = []

            for i, y_target in enumerate(range(num_classes)):
                log('Reverse-engineering trigger for label %d/%d ...' %
                    (y_target, num_classes - 1))
                pattern_init = np.random.random(input_shape) * 255.0
                mask_init = np.random.random(mask_shape)

                pattern, mask, mask_upsample, logs = visualizer.visualize(
                    gen=gen, y_target=y_target, pattern_init=pattern_init,
                    mask_init=mask_init)

                l1_norm = float(np.sum(np.abs(mask_upsample)))
                l1_norms.append(l1_norm)

                pattern_file = 'label_%d_pattern.png' % y_target
                mask_file = 'label_%d_mask.png' % y_target
                fusion_file = 'label_%d_fusion.png' % y_target

                utils_backdoor.dump_image(pattern, os.path.join(result_dir, pattern_file), 'png')
                utils_backdoor.dump_image(
                    np.expand_dims(mask_upsample, axis=2) * 255,
                    os.path.join(result_dir, mask_file), 'png')
                fusion = np.multiply(pattern, np.expand_dims(mask_upsample, axis=2))
                utils_backdoor.dump_image(fusion, os.path.join(result_dir, fusion_file), 'png')

                per_label.append({
                    'label': int(y_target),
                    'l1_norm': l1_norm,
                    'pattern_img': '%s/%s' % (run_id, pattern_file),
                    'mask_img': '%s/%s' % (run_id, mask_file),
                    'fusion_img': '%s/%s' % (run_id, fusion_file),
                })

                progress(i + 1, num_classes, 'detecting')

            log('Running MAD outlier detection over %d labels...' % num_classes)
            labels = [p['label'] for p in per_label]
            outlier_result = outlier_detection(l1_norms, labels)
            flagged_labels = set(f['label'] for f in outlier_result['flagged'])
            for p in per_label:
                p['flagged'] = p['label'] in flagged_labels

            log('Detection complete. %d label(s) flagged as likely backdoored: %s' %
                (len(outlier_result['flagged']),
                 [f['label'] for f in outlier_result['flagged']]))

    return {
        'run_id': run_id,
        'num_classes': int(num_classes),
        'per_label': per_label,
        'outlier': outlier_result,
    }
