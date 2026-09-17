#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang
#
# NOTE: Rewritten for TF2 / Keras 3 eager execution (e.g. Google Colab).
# The original implementation built a static TF1 computation graph
# (K.placeholder, opt.get_updates, K.function(..., updates=...)) which no
# longer exists in modern Keras. This version keeps the exact same
# algorithm (mask/pattern reverse-engineering with a tanh reparameterisation
# and an adaptive L1-regularisation cost, "Neural Cleanse" style) but runs
# each optimisation step eagerly inside a tf.GradientTape, matching how
# TF2/Keras 3 actually executes code.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import UpSampling2D, Cropping2D

import utils_backdoor

from decimal import Decimal


class Visualizer:

    # upsample size, default is 1
    UPSAMPLE_SIZE = 1
    # pixel intensity range of image and preprocessing method
    # raw: [0, 255]
    # mnist: [0, 1]
    # imagenet: imagenet mean centering
    # inception: [-1, 1]
    INTENSITY_RANGE = 'raw'
    # type of regularization of the mask
    REGULARIZATION = 'l1'
    # threshold of attack success rate for dynamically changing cost
    ATTACK_SUCC_THRESHOLD = 0.99
    # patience
    PATIENCE = 10
    # multiple of changing cost, down multiple is the square root of this
    COST_MULTIPLIER = 1.5,
    # if resetting cost to 0 at the beginning
    # default is true for full optimization, set to false for early detection
    RESET_COST_TO_ZERO = True
    # min/max of mask
    MASK_MIN = 0
    MASK_MAX = 1
    # min/max of raw pixel intensity
    COLOR_MIN = 0
    COLOR_MAX = 255
    # number of color channel
    IMG_COLOR = 3
    # whether to shuffle during each epoch
    SHUFFLE = True
    # batch size of optimization
    BATCH_SIZE = 32
    # verbose level, 0, 1 or 2
    VERBOSE = 1
    # whether to return log or not
    RETURN_LOGS = True
    # whether to save last pattern or best pattern
    SAVE_LAST = False
    # epsilon used in tanh
    EPSILON = K.epsilon()
    # early stop flag
    EARLY_STOP = True
    # early stop threshold
    EARLY_STOP_THRESHOLD = 0.99
    # early stop patience
    EARLY_STOP_PATIENCE = 2 * PATIENCE
    # save tmp masks, for debugging purpose
    SAVE_TMP = False
    # dir to save intermediate masks
    TMP_DIR = 'tmp'
    # whether input image has been preprocessed or not
    RAW_INPUT_FLAG = False

    def __init__(self, model, intensity_range, regularization, input_shape,
                 init_cost, steps, mini_batch, lr, num_classes,
                 upsample_size=UPSAMPLE_SIZE,
                 attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
                 patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
                 reset_cost_to_zero=RESET_COST_TO_ZERO,
                 mask_min=MASK_MIN, mask_max=MASK_MAX,
                 color_min=COLOR_MIN, color_max=COLOR_MAX, img_color=IMG_COLOR,
                 shuffle=SHUFFLE, batch_size=BATCH_SIZE, verbose=VERBOSE,
                 return_logs=RETURN_LOGS, save_last=SAVE_LAST,
                 epsilon=EPSILON,
                 early_stop=EARLY_STOP,
                 early_stop_threshold=EARLY_STOP_THRESHOLD,
                 early_stop_patience=EARLY_STOP_PATIENCE,
                 save_tmp=SAVE_TMP, tmp_dir=TMP_DIR,
                 raw_input_flag=RAW_INPUT_FLAG):

        assert intensity_range in {'imagenet', 'inception', 'mnist', 'raw'}
        assert regularization in {None, 'l1', 'l2'}

        self.model = model
        self.intensity_range = intensity_range
        self.regularization = regularization
        self.input_shape = input_shape
        self.init_cost = init_cost
        self.steps = steps
        self.mini_batch = mini_batch
        self.lr = lr
        self.num_classes = num_classes
        self.upsample_size = upsample_size
        self.attack_succ_threshold = attack_succ_threshold
        self.patience = patience
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.reset_cost_to_zero = reset_cost_to_zero
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.color_min = color_min
        self.color_max = color_max
        self.img_color = img_color
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.verbose = verbose
        self.return_logs = return_logs
        self.save_last = save_last
        self.epsilon = epsilon
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.save_tmp = save_tmp
        self.tmp_dir = tmp_dir
        self.raw_input_flag = raw_input_flag

        mask_size = np.ceil(np.array(input_shape[0:2], dtype=float) /
                            upsample_size)
        mask_size = mask_size.astype(int)
        self.mask_size = mask_size
        mask = np.zeros(self.mask_size)
        pattern = np.zeros(input_shape)
        mask = np.expand_dims(mask, axis=2)

        mask_tanh = np.zeros_like(mask)
        pattern_tanh = np.zeros_like(pattern)

        # prepare mask/pattern as eager tf.Variables (these are what we
        # actually optimise via gradient descent - everything derived from
        # them below is recomputed fresh on every forward pass).
        self.mask_tanh_tensor = tf.Variable(mask_tanh, dtype=tf.float32,
                                            trainable=True, name='mask_tanh')
        self.pattern_tanh_tensor = tf.Variable(pattern_tanh, dtype=tf.float32,
                                               trainable=True, name='pattern_tanh')

        # static layers reused across every forward pass (their weights
        # don't change - only the mask/pattern variables above do)
        self.upsample_layer = UpSampling2D(
            size=(self.upsample_size, self.upsample_size))

        # figure out the crop needed to get back to input_shape, same as
        # the original code (uses a dummy forward pass through the mask
        # tensor's static shape, which doesn't depend on variable values)
        dummy_mask_tensor = tf.expand_dims(
            tf.repeat(tf.zeros_like(self.mask_tanh_tensor), repeats=self.img_color, axis=2),
            axis=0)
        uncrop_shape = self.upsample_layer(dummy_mask_tensor).shape[1:]
        self.cropping_layer = Cropping2D(
            cropping=((0, uncrop_shape[0] - self.input_shape[0]),
                      (0, uncrop_shape[1] - self.input_shape[1])))

        self.cost = self.init_cost
        # Tensor mirror of self.cost so _train_step (compiled with
        # @tf.function below) picks up cost changes via .assign() without
        # forcing a retrace on every value change.
        self.cost_tensor = tf.Variable(float(self.init_cost), dtype=tf.float32,
                                       trainable=False, name='cost')
        self.opt = keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.9)

        pass

    def _keras_preprocess(self, x_input, intensity_range):
        if intensity_range == 'raw':
            x_preprocess = x_input
        elif intensity_range == 'imagenet':
            # 'RGB'->'BGR'
            x_tmp = x_input[..., ::-1]
            mean = tf.constant([[[103.939, 116.779, 123.68]]], dtype=tf.float32)
            x_preprocess = x_tmp - mean
        elif intensity_range == 'inception':
            x_preprocess = (x_input / 255.0 - 0.5) * 2.0
        elif intensity_range == 'mnist':
            x_preprocess = x_input / 255.0
        else:
            raise Exception('unknown intensity_range %s' % intensity_range)
        return x_preprocess

    def _keras_reverse_preprocess(self, x_input, intensity_range):
        if intensity_range == 'raw':
            x_reverse = x_input
        elif intensity_range == 'imagenet':
            mean = tf.constant([[[103.939, 116.779, 123.68]]], dtype=tf.float32)
            x_reverse = x_input + mean
            x_reverse = x_reverse[..., ::-1]
        elif intensity_range == 'inception':
            x_reverse = (x_input / 2 + 0.5) * 255.0
        elif intensity_range == 'mnist':
            x_reverse = x_input * 255.0
        else:
            raise Exception('unknown intensity_range %s' % intensity_range)
        return x_reverse

    def _mask_and_pattern(self):
        """Recompute the current mask/mask_upsample/pattern tensors from the
        (trainable) tanh variables. Called fresh every forward pass so the
        GradientTape sees the current variable values."""
        mask_tensor_unrepeat = (tf.tanh(self.mask_tanh_tensor) /
                                (2 - self.epsilon) + 0.5)
        mask_tensor_unexpand = tf.repeat(
            mask_tensor_unrepeat, repeats=self.img_color, axis=2)
        mask_tensor = tf.expand_dims(mask_tensor_unexpand, axis=0)
        mask_upsample_tensor_uncrop = self.upsample_layer(mask_tensor)
        mask_upsample_tensor = self.cropping_layer(mask_upsample_tensor_uncrop)

        pattern_raw_tensor = (
            (tf.tanh(self.pattern_tanh_tensor) / (2 - self.epsilon) + 0.5) *
            255.0)

        return mask_tensor, mask_upsample_tensor, pattern_raw_tensor

    def _forward(self, X_batch):
        """Given a raw input batch, build the adversarial batch and run it
        through the model. Returns (output, mask_tensor, mask_upsample_tensor,
        pattern_raw_tensor)."""
        mask_tensor, mask_upsample_tensor, pattern_raw_tensor = self._mask_and_pattern()
        reverse_mask_tensor = tf.ones_like(mask_upsample_tensor) - mask_upsample_tensor

        input_tensor = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        if self.raw_input_flag:
            input_raw_tensor = input_tensor
        else:
            input_raw_tensor = self._keras_reverse_preprocess(
                input_tensor, self.intensity_range)

        # IMPORTANT: MASK OPERATION IN RAW DOMAIN
        X_adv_raw_tensor = (
            reverse_mask_tensor * input_raw_tensor +
            mask_upsample_tensor * pattern_raw_tensor)

        X_adv_tensor = self._keras_preprocess(X_adv_raw_tensor, self.intensity_range)
        output_tensor = self.model(X_adv_tensor, training=False)

        return output_tensor, mask_tensor, mask_upsample_tensor, pattern_raw_tensor

    @tf.function(reduce_retracing=True)
    def _train_step(self, X_batch, Y_target):
        """One eager-compiled optimisation step (wrapped in @tf.function so
        TF traces it once into a graph instead of re-running the whole
        Python forward pass through eager ops every single call - this is
        the main speed win over plain eager execution). Returns
        (loss_ce, loss_reg, loss, loss_acc) tensors matching the original
        K.function(...) output shapes."""
        y_true_tensor = tf.convert_to_tensor(Y_target, dtype=tf.float32)

        with tf.GradientTape() as tape:
            output_tensor, _, mask_upsample_tensor, _ = self._forward(X_batch)

            loss_acc = categorical_accuracy(y_true_tensor, output_tensor)
            loss_ce = categorical_crossentropy(y_true_tensor, output_tensor)

            if self.regularization is None:
                loss_reg = tf.constant(0.0)
            elif self.regularization == 'l1':
                loss_reg = tf.reduce_sum(tf.abs(mask_upsample_tensor)) / self.img_color
            elif self.regularization == 'l2':
                loss_reg = tf.sqrt(tf.reduce_sum(tf.square(mask_upsample_tensor)) / self.img_color)

            loss = loss_ce + loss_reg * self.cost_tensor

        grads = tape.gradient(loss, [self.pattern_tanh_tensor, self.mask_tanh_tensor])
        self.opt.apply_gradients(zip(grads, [self.pattern_tanh_tensor, self.mask_tanh_tensor]))

        return (loss_ce, loss_reg * tf.ones_like(loss_ce), loss, loss_acc)

    def _set_cost(self, value):
        self.cost = value
        self.cost_tensor.assign(float(value))

    def reset_opt(self):
        # Recreate the optimizer from scratch rather than poking at its
        # internal weights - simpler and equivalent under TF2's eager
        # optimizers (they carry their own moment estimates internally).
        self.opt = keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.9)
        pass

    def reset_state(self, pattern_init, mask_init):

        print('resetting state')

        # setting cost
        if self.reset_cost_to_zero:
            self._set_cost(0)
        else:
            self._set_cost(self.init_cost)

        # setting mask and pattern
        mask = np.array(mask_init)
        pattern = np.array(pattern_init)
        mask = np.clip(mask, self.mask_min, self.mask_max)
        pattern = np.clip(pattern, self.color_min, self.color_max)
        mask = np.expand_dims(mask, axis=2)

        # convert to tanh space
        mask_tanh = np.arctanh((mask - 0.5) * (2 - self.epsilon))
        pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - self.epsilon))
        print('mask_tanh', np.min(mask_tanh), np.max(mask_tanh))
        print('pattern_tanh', np.min(pattern_tanh), np.max(pattern_tanh))

        self.mask_tanh_tensor.assign(mask_tanh)
        self.pattern_tanh_tensor.assign(pattern_tanh)

        # resetting optimizer states
        self.reset_opt()

        pass

    def save_tmp_func(self, step):

        _, mask_upsample_tensor, pattern_raw_tensor = self._mask_and_pattern()

        cur_mask = mask_upsample_tensor.numpy()
        cur_mask = cur_mask[0, ..., 0]
        img_filename = (
            '%s/%s' % (self.tmp_dir, 'tmp_mask_step_%d.png' % step))
        utils_backdoor.dump_image(np.expand_dims(cur_mask, axis=2) * 255,
                                  img_filename,
                                  'png')

        cur_fusion = (mask_upsample_tensor * pattern_raw_tensor).numpy()
        cur_fusion = cur_fusion[0, ...]
        img_filename = (
            '%s/%s' % (self.tmp_dir, 'tmp_fusion_step_%d.png' % step))
        utils_backdoor.dump_image(cur_fusion, img_filename, 'png')

        pass

    def visualize(self, gen, y_target, pattern_init, mask_init):

        # since we use a single optimizer repeatedly, we need to reset
        # optimzier's internal states before running the optimization
        self.reset_state(pattern_init, mask_init)

        # best optimization results
        mask_best = None
        mask_upsample_best = None
        pattern_best = None
        reg_best = float('inf')

        # logs and counters for adjusting balance cost
        logs = []
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # counter for early stop
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        # vectorized target
        Y_target = to_categorical([y_target] * self.batch_size,
                                  self.num_classes)

        # loop start
        for step in range(self.steps):

            # record loss for all mini-batches
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for idx in range(self.mini_batch):
                X_batch, _ = next(gen)
                if X_batch.shape[0] != Y_target.shape[0]:
                    Y_target = to_categorical([y_target] * X_batch.shape[0],
                                              self.num_classes)
                (loss_ce_value,
                    loss_reg_value,
                    loss_value,
                    loss_acc_value) = self._train_step(X_batch, Y_target)
                loss_ce_list.extend(list(np.array(loss_ce_value).flatten()))
                loss_reg_list.extend(list(np.array(loss_reg_value).flatten()))
                loss_list.extend(list(np.array(loss_value).flatten()))
                loss_acc_list.extend(list(np.array(loss_acc_value).flatten()))

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_loss_acc = np.mean(loss_acc_list)

            # check to save best mask or not
            if avg_loss_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                mask_tensor, mask_upsample_tensor, pattern_raw_tensor = self._mask_and_pattern()
                mask_best = mask_tensor.numpy()
                mask_best = mask_best[0, ..., 0]
                mask_upsample_best = mask_upsample_tensor.numpy()
                mask_upsample_best = mask_upsample_best[0, ..., 0]
                pattern_best = pattern_raw_tensor.numpy()
                reg_best = avg_loss_reg

            # verbose
            if self.verbose != 0:
                if self.verbose == 2 or step % (self.steps // 10) == 0:
                    print('step: %3d, cost: %.2E, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
                          (step, Decimal(float(self.cost)), avg_loss_acc, avg_loss,
                           avg_loss_ce, avg_loss_reg, reg_best))

            # save log
            logs.append((step,
                         avg_loss_ce, avg_loss_reg, avg_loss, avg_loss_acc,
                         reg_best, self.cost))

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and
                        cost_up_flag and
                        early_stop_counter >= self.early_stop_patience):
                    print('early stop')
                    break

            # check cost modification
            if self.cost == 0 and avg_loss_acc >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self._set_cost(self.init_cost)
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2E' % Decimal(float(self.cost)))
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if self.verbose == 2:
                    print('up cost from %.2E to %.2E' %
                          (Decimal(float(self.cost)),
                           Decimal(float(self.cost * self.cost_multiplier_up))))
                self._set_cost(self.cost * self.cost_multiplier_up)
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                if self.verbose == 2:
                    print('down cost from %.2E to %.2E' %
                          (Decimal(float(self.cost)),
                           Decimal(float(self.cost / self.cost_multiplier_down))))
                self._set_cost(self.cost / self.cost_multiplier_down)
                cost_down_flag = True

            if self.save_tmp:
                self.save_tmp_func(step)

        # save the final version
        if mask_best is None or self.save_last:
            mask_tensor, mask_upsample_tensor, pattern_raw_tensor = self._mask_and_pattern()
            mask_best = mask_tensor.numpy()
            mask_best = mask_best[0, ..., 0]
            mask_upsample_best = mask_upsample_tensor.numpy()
            mask_upsample_best = mask_upsample_best[0, ..., 0]
            pattern_best = pattern_raw_tensor.numpy()

        if self.return_logs:
            return pattern_best, mask_best, mask_upsample_best, logs
        else:
            return pattern_best, mask_best, mask_upsample_best
