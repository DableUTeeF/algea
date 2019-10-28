import tensorflow as tf
import warnings
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import os
import re
import subprocess
from keras import backend as K


class GPUUtilizeMon(Callback):
    def __init__(self):
        super().__init__()
        self.history = {}
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.history[self.epoch] = {'temperature': [], 'fan_percent': [], 'gpu_percent': [], 'ram_percent': []}

    def on_batch_end(self, batch, logs=None):
        command = 'nvidia-smi'
        p = subprocess.check_output(command)
        c = re.findall(r'\b\d+C', str(p))
        temperature = int(c[0][:-1])
        n = re.findall(r'\b\d+%', str(p))
        fan_percent = int(n[0][:-1])
        gpu_percent = int(n[1][:-1])
        ram_using = re.findall(r'\b\d+MiB+ /', str(p))[0][:-5]
        ram_total = re.findall(r'/  \b\d+MiB', str(p))[0][3:-3]
        ram_percent = int(ram_using) / int(ram_total)
        self.history[self.epoch]['temperature'].append(temperature)
        self.history[self.epoch]['fan_percent'].append(fan_percent)
        self.history[self.epoch]['gpu_percent'].append(gpu_percent)
        self.history[self.epoch]['ram_percent'].append(ram_percent)


class CustomTensorBoard(TensorBoard):
    """ to log the loss after each batch
    """

    def __init__(self, settings_str, log_every=1, **kwargs):
        super(CustomTensorBoard, self).__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0
        self.settings_str = settings_str
        self.history = {}
        self.epoch = 0
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    @staticmethod
    def call_process():
        command = 'nvidia-smi'
        p = subprocess.check_output(command)
        c = re.findall(r'\b\d+C', str(p))
        temperature = int(c[0][:-1])
        n = re.findall(r'\b\d+%', str(p))
        fan_percent = int(n[0][:-1])
        gpu_percent = int(n[1][:-1])
        ram_using = re.findall(r'\b\d+MiB+ /', str(p))[0][:-5]
        ram_total = re.findall(r'/  \b\d+MiB', str(p))[0][3:-3]
        ram_percent = int(ram_using) / int(ram_total)
        history = {'temperature': temperature, 'fan_percent': fan_percent, 'gpu_percent': gpu_percent,
                   'ram_percent': ram_percent}
        return history

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            history = self.call_process()
            for name, value in history.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super(CustomTensorBoard, self).on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        TensorBoard.on_train_begin(self, logs=logs)

        tensor = tf.convert_to_tensor(self.settings_str)
        summary = tf.summary.text("Run Settings", tensor)

        with tf.Session() as sess:
            s = sess.run(summary)
            self.writer.add_summary(s)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.history[self.epoch] = {'temperature': [], 'fan_percent': [], 'gpu_percent': [], 'ram_percent': []}
    #     print(f'\033[{np.random.randint(31, 37)}m')


class CustomModelCheckpoint(ModelCheckpoint):
    """ to save the template model, not the multi-GPU model
    """

    def __init__(self, model_to_save, **kwargs):
        super(CustomModelCheckpoint, self).__init__(**kwargs)
        self.model_to_save = model_to_save

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_to_save.save_weights(filepath, overwrite=True)
                        else:
                            self.model_to_save.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_to_save.save_weights(filepath, overwrite=True)
                else:
                    self.model_to_save.save(filepath, overwrite=True)
        print(f'\033[{np.random.randint(31, 37)}m')
        super(CustomModelCheckpoint, self).on_batch_end(epoch, logs)
