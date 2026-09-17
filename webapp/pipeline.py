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
import json

# --------------------------------------------------------------------------- #
# CPU thread limits. These MUST be set before numpy/scipy/tensorflow are
# imported, since that's when the BLAS libraries (OpenBLAS/MKL) read them and
# spin up their thread pools. Override any of them with a real environment
# variable set before starting the server (e.g. `$env:AEBAD_CPU_THREADS=1`)
# if you want a different value without editing this file.
# --------------------------------------------------------------------------- #
CPU_THREADS = os.environ.get('AEBAD_CPU_THREADS', '4')
os.environ.setdefault('AEBAD_CPU_THREADS', CPU_THREADS)
os.environ.setdefault('OMP_NUM_THREADS', CPU_THREADS)
os.environ.setdefault('OPENBLAS_NUM_THREADS', CPU_THREADS)
os.environ.setdefault('MKL_NUM_THREADS', CPU_THREADS)
os.environ.setdefault('NUMEXPR_NUM_THREADS', CPU_THREADS)

# --------------------------------------------------------------------------- #
# This codebase was written for TF1/Keras2-style session & graph APIs
# (tf.Session, tf.Graph, keras.backend.set_session, ImageDataGenerator, ...).
# Modern TensorFlow bundles Keras 3 by default, which removed all of these.
# Installing the `tf_keras` package and setting this flag makes `import keras`
# (and tensorflow's own internal keras usage) resolve to the legacy
# Keras-2-compatible implementation instead, which keeps this old code
# working with minimal changes. This MUST be set before tensorflow/keras are
# imported anywhere in the process.
# --------------------------------------------------------------------------- #
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
CPU_THREADS = int(CPU_THREADS)

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


def _meta_path(model_path):
    """Sidecar JSON file storing training ground-truth next to a saved model."""
    return model_path + '.meta.json'


def load_model_meta(model_path):
    """Return the training metadata dict for a model, or None if not found/unreadable."""
    path = _meta_path(model_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


# --------------------------------------------------------------------------- #
# TensorFlow / Keras session handling
#
# This project pins TF 1.10 / Keras 2.2.2 (graph mode). Each job gets its own
# fresh graph + session so that training/detection jobs never leak Keras
# global state into one another when run back-to-back from the Flask app.
# --------------------------------------------------------------------------- #

# Max CPU threads TF is allowed to use. Override with the AEBAD_CPU_THREADS
# env var (e.g. `set AEBAD_CPU_THREADS=1` before starting the server) if it's
# still too heavy. (Value is already resolved and BLAS-related env vars set
# near the top of this file, before numpy/tensorflow were imported.)


def new_session():
    """Return (graph, sess) for TF1-style graph isolation, or (None, None)
    on TF2/Keras 3 runtimes (e.g. Colab) where there is no session/graph API
    to set up - eager execution handles this per-call, so we just no-op.
    Use `session_scope(graph, sess)` below instead of calling
    `graph.as_default()`/`sess.as_default()` directly, so both cases work.
    """
    import tensorflow as tf

    if not tf.__version__.startswith('1'):
        # TF2 (and Keras 3, which ships with modern TF) has no
        # tf.Session / K.set_session - nothing to do here.
        return None, None

    import tensorflow.compat.v1 as tf1
    from tensorflow.keras import backend as K

    config = tf1.ConfigProto(
        intra_op_parallelism_threads=CPU_THREADS,
        inter_op_parallelism_threads=CPU_THREADS,
        device_count={'CPU': 1},
    )
    graph = tf1.Graph()
    with graph.as_default():
        sess = tf1.Session(graph=graph, config=config)
        K.set_session(sess)
    return graph, sess


def session_scope(graph, sess):
    """Context manager that mirrors `graph.as_default(): sess.as_default():`
    when both are real TF1 objects, and is a no-op under TF2 (graph/sess
    will be None from new_session())."""
    import contextlib

    if graph is None or sess is None:
        return contextlib.nullcontext()

    stack = contextlib.ExitStack()
    stack.enter_context(graph.as_default())
    stack.enter_context(sess.as_default())
    return stack


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
    for y_key in ('Y_train', 'Y_test', 'Y'):
        if y_key in info['shapes'] and len(info['shapes'][y_key]) == 2:
            info['num_classes'] = info['shapes'][y_key][1]
            break
    keys_set = set(info['keys'])
    info['has_train_split'] = {'X_train', 'Y_train', 'X_test', 'Y_test'}.issubset(keys_set)
    info['needs_auto_split'] = (not info['has_train_split']) and (
        {'X_test', 'Y_test'}.issubset(keys_set) or {'X', 'Y'}.issubset(keys_set))
    return info


def _load_keys(path, keys):
    return utils_backdoor.load_dataset(path, keys=keys)


# --------------------------------------------------------------------------- #
# Model definition (same architecture as injection/gtsrb_injection_example.py,
# generalised to an arbitrary input shape / class count so it also works with
# datasets other than GTSRB).
# --------------------------------------------------------------------------- #

def build_model(input_shape, num_classes, base=32, dense=512):
    from tensorflow import keras
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    from tensorflow.keras.models import Sequential


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
        from tensorflow import keras
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


def _load_train_test(dataset_path, split_ratio=0.8, log=_noop_log):
    """
    Load a dataset for training, tolerant of different layouts:
      - Proper X_train/Y_train/X_test/Y_test  -> used as-is (split_ratio ignored).
      - Only a single pool (X_test/Y_test, or generic X/Y) -> shuffled and split
        into train/test using split_ratio (fraction going to train).
    Returns (X_train, Y_train, X_test, Y_test, used_split_ratio_or_None).
    """
    with h5py.File(dataset_path, 'r') as hf:
        keys = set(hf.keys())

    if {'X_train', 'Y_train', 'X_test', 'Y_test'}.issubset(keys):
        dataset = _load_keys(dataset_path, ['X_train', 'Y_train', 'X_test', 'Y_test'])
        X_train = np.array(dataset['X_train'], dtype='float32')
        Y_train = np.array(dataset['Y_train'], dtype='float32')
        X_test = np.array(dataset['X_test'], dtype='float32')
        Y_test = np.array(dataset['Y_test'], dtype='float32')
        log('Dataset already has train/test split (%d train, %d test).' %
            (len(Y_train), len(Y_test)))
        return X_train, Y_train, X_test, Y_test, None

    if {'X_test', 'Y_test'}.issubset(keys):
        x_key, y_key = 'X_test', 'Y_test'
    elif {'X', 'Y'}.issubset(keys):
        x_key, y_key = 'X', 'Y'
    else:
        raise ValueError(
            'Dataset has keys %s - expected either X_train/Y_train/X_test/Y_test, '
            'or a single pool as X_test/Y_test or X/Y to auto-split.' % sorted(keys))

    dataset = _load_keys(dataset_path, [x_key, y_key])
    X = np.array(dataset[x_key], dtype='float32')
    Y = np.array(dataset[y_key], dtype='float32')

    split_ratio = float(split_ratio)
    split_ratio = min(max(split_ratio, 0.05), 0.95)

    n = len(Y)
    idx = np.random.permutation(n)
    n_train = int(round(n * split_ratio))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    log('Dataset has only "%s"/"%s" (%d samples). Auto-splitting %.0f%% train / %.0f%% test '
        '-> %d train, %d test.' % (x_key, y_key, n, split_ratio * 100, (1 - split_ratio) * 100,
                                    len(Y_train), len(Y_test)))
    return X_train, Y_train, X_test, Y_test, split_ratio


def train_model(dataset_path, model_name, mode='clean', target_labels=None,
                 pattern_size=4, margin=1, inject_ratio=0.2, epochs=10,
                 base=32, dense=512, train_split_ratio=0.8,
                 log=_noop_log, progress=_noop_progress):
    """
    mode:
      'clean'  - train on the dataset as-is, no poisoning
      'poison' - poison on the fly during training (backdoor injection)
    train_split_ratio:
      Used only when the dataset does not already have a train/test split
      (i.e. it only has X_test/Y_test or X/Y) - fraction of samples put into
      the training set, the rest becomes the test set.
    Returns dict with model_path + final metrics.
    """
    import keras

    graph, sess = new_session()
    with session_scope(graph, sess):
            log('Loading dataset %s' % dataset_path)
            X_train, Y_train, X_test, Y_test, used_split_ratio = _load_train_test(
                dataset_path, split_ratio=train_split_ratio, log=log)

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

            meta = {
                'mode': mode,
                'true_backdoor_labels': target_labels if mode == 'poison' else [],
                'pattern_size': pattern_size if mode == 'poison' else None,
                'margin': margin if mode == 'poison' else None,
                'inject_ratio': inject_ratio if mode == 'poison' else None,
                'dataset_path': dataset_path,
                'train_split_ratio': used_split_ratio,
                'metrics': metrics,
            }
            with open(_meta_path(model_path), 'w') as f:
                json.dump(meta, f, indent=2)

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


def classification_metrics(per_label, true_backdoor_labels):
    """
    Standard binary-classification metrics, treating each scanned label as one
    sample: positive = "this label is backdoored".
      true_backdoor_labels : iterable of int, the labels actually poisoned
                              (ground truth, provided by the user)
      per_label[i]['flagged']: the detector's prediction for that label
    """
    true_set = set(int(t) for t in true_backdoor_labels)
    tp = fp = tn = fn = 0
    for p in per_label:
        actual_positive = p['label'] in true_set
        predicted_positive = p['flagged']
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'fpr': fpr, 'fnr': fnr,
    }


def run_detection(model_path, dataset_path, run_id, steps=200, batch_size=32,
                   lr=0.1, init_cost=1e-3, patience=5, cost_multiplier=2,
                   attack_succ_threshold=0.99, labels_to_scan=None,
                   true_backdoor_labels=None,
                   log=_noop_log, progress=_noop_progress):
    """
    labels_to_scan: optional list/iterable of class indices to scan instead of
    every class. Useful to cut runtime drastically while testing (e.g. scan
    just 5 classes instead of all 43). MAD outlier detection still needs a
    reasonable spread of "normal" classes to compare against, so don't drop
    below ~5-10 labels if you want the flagging to mean anything.

    true_backdoor_labels: optional list/iterable of class indices that are
    ACTUALLY poisoned (ground truth, known because you injected them, or
    because you're testing on a known model). If given, Accuracy/Precision/
    Recall/F1/FPR/FNR are computed against the detector's flags. If omitted,
    only the raw outlier-detection results are returned (no metrics).
    """
    import time as _time
    from keras.models import load_model
    # ImageDataGenerator removed in Keras 3; replaced with a plain
    # batch generator below (no augmentation was actually used here).
    from visualizer import Visualizer

    _start_time = _time.time()

    auto_meta_used = False
    if not true_backdoor_labels:
        meta = load_model_meta(model_path)
        if meta and meta.get('true_backdoor_labels'):
            true_backdoor_labels = meta['true_backdoor_labels']
            auto_meta_used = True
            log('Auto-loaded ground-truth backdoored label(s) from training metadata: %s' %
                true_backdoor_labels)

    graph, sess = new_session()
    result_dir = os.path.join(RESULTS_DIR, run_id)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with session_scope(graph, sess):
            log('Loading dataset %s' % dataset_path)
            dataset = _load_keys(dataset_path, ['X_test', 'Y_test'])
            X_test = np.array(dataset['X_test'], dtype='float32')
            Y_test = np.array(dataset['Y_test'], dtype='float32')
            input_shape = X_test.shape[1:]
            num_classes = Y_test.shape[1]
            img_color = input_shape[2] if len(input_shape) == 3 else 1

            log('Loading model %s' % model_path)
            model = load_model(model_path)

            def _simple_batch_generator(X, Y, batch_size):
                n = len(X)
                while True:
                    idx = np.random.permutation(n)
                    for start in range(0, n, batch_size):
                        batch_idx = idx[start:start + batch_size]
                        yield X[batch_idx], Y[batch_idx]

            gen = _simple_batch_generator(X_test, Y_test, batch_size)

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

            target_list = (sorted(set(int(t) for t in labels_to_scan))
                            if labels_to_scan else list(range(num_classes)))
            log('Scanning %d of %d classes: %s' %
                (len(target_list), num_classes, target_list))

            l1_norms = []
            per_label = []

            for i, y_target in enumerate(target_list):
                log('Reverse-engineering trigger for label %d (%d/%d) ...' %
                    (y_target, i + 1, len(target_list)))
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

                progress(i + 1, len(target_list), 'detecting')

            log('Running MAD outlier detection over %d scanned labels...' % len(target_list))
            labels = [p['label'] for p in per_label]
            outlier_result = outlier_detection(l1_norms, labels)
            flagged_labels = set(f['label'] for f in outlier_result['flagged'])
            for p in per_label:
                p['flagged'] = p['label'] in flagged_labels

            log('Detection complete. %d label(s) flagged as likely backdoored: %s' %
                (len(outlier_result['flagged']),
                 [f['label'] for f in outlier_result['flagged']]))

    detection_time = _time.time() - _start_time
    log('Detection time: %.2f sec' % detection_time)

    metrics = None
    if true_backdoor_labels:
        metrics = classification_metrics(per_label, true_backdoor_labels)
        metrics['detection_time_sec'] = float(detection_time)
        log('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f FPR=%.4f FNR=%.4f' %
            (metrics['accuracy'], metrics['precision'], metrics['recall'],
             metrics['f1_score'], metrics['fpr'], metrics['fnr']))

    return {
        'run_id': run_id,
        'num_classes': int(num_classes),
        'num_scanned': len(target_list),
        'per_label': per_label,
        'outlier': outlier_result,
        'detection_time_sec': float(detection_time),
        'metrics': metrics,
        'true_backdoor_labels': list(true_backdoor_labels) if true_backdoor_labels else None,
        'ground_truth_source': 'training_metadata' if auto_meta_used else ('user_provided' if metrics else None),
    }
