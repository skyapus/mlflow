import tensorflow as tf
import tensorflow_datasets as tfds
import os
from datetime import datetime

(ds_train, ds_test), ds_info = tfds.load(
 'mnist',
 split=['train', 'test'],
 shuffle_files=True,
 as_supervised=True,
 with_info=True,
)

# Normalizes images: `uint8` -> `float32`
def normalize_img(image, label):
 return tf.cast(image, tf.float32) / 255., label # Train Dataset
ds_train = ds_train.map(
 normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE) # Test Dataset
ds_test = ds_test.map(
 normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


# Define the layers
inputs = tf.keras.Input(shape=(28, 28, 1))
hidden = tf.keras.layers.Flatten()(inputs)
hidden2 = tf.keras.layers.Dense(128, activation='relu')(hidden)
outputs = tf.keras.layers.Dense(10, activation='softmax')(hidden2)# Optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.002)# Create a Model object
model = tf.keras.Model(inputs, outputs)# Compile the model
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


checkpoint_path = "models/train"

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)




model.fit(ds_train, epochs=50, validation_data=ds_test,
    batch_size=128, callbacks=[cp_callback, tensorboard_callback])


