import tensorflow as tf
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
"G:\\AI_dataset\\dom\\segmentation\\val\\img",validation_split=0.2,subset="training",seed=0,image_size=(3000,3000),batch_size=32)
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
tf.print(train_ds)