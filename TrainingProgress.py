import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class TrainingProgress(Callback):
    def __init__(self, total_epochs, total_batches, log_dir, update_freq=100):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.update_freq = update_freq
        self.batch_count = 0
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_train_begin(self, logs=None):
        self.batch_count = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.update_freq == 0 or self.batch_count == self.total_batches:
            with self.writer.as_default():
                tf.summary.scalar("Loss/train", logs["loss"], step=self.batch_count)
                tf.summary.scalar("Mean Absolute Error/train", logs["mean_absolute_error"], step=self.batch_count)
                tf.summary.scalar("Learning Rate", tf.keras.backend.get_value(self.model.optimizer.learning_rate), step=self.batch_count)
                for layer in self.model.layers:
                    for weight in layer.weights:
                        tf.summary.histogram(weight.name, weight, step=self.batch_count)
                tf.summary.flush()

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            tf.summary.scalar("Loss/validation", logs.get("val_loss"), step=epoch)
            tf.summary.scalar("Mean Absolute Error/validation", logs.get("val_mean_absolute_error"), step=epoch)
            tf.summary.flush()