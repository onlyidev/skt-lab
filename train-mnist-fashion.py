# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import math
import tensorflow.keras.layers as tfl
import tensorflow.keras.callbacks as tcl

# %%
classes = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
]
DIM = 28
CH = 3
BATCH = 32
TRAIN = True
EPOCHS = 2

# %%
ds = tfds.load("huggingface:fashion_mnist", split="all", as_supervised=True, data_dir="/drive")

# %%
def dimensionLoweringConvolutionBlock(filters=32, kernel_size=(3, 3), dropout=0.2, activation='relu', padding="same", normalization=True, dim_divisor=2):
    s = [
        tfl.Conv2D(filters, kernel_size, activation=activation, padding=padding),
        tfl.BatchNormalization(),
        tfl.Conv2D(filters, kernel_size, activation=activation, padding=padding)
        ]
    if (normalization):
            s.append(tfl.BatchNormalization())
    s.extend(
        [
            tfl.MaxPooling2D((dim_divisor, dim_divisor)),
            tfl.Dropout(dropout),
        ]
    )
    return tf.keras.Sequential(s)

def mlpHidden(neurons=1, dropout=0.2, activation='relu', normalization=True):
    s = [ tfl.Dense(neurons, activation=activation) ]
    if (normalization):
        s.append(tfl.BatchNormalization())
    s.append(tfl.Dropout(dropout))
    return tf.keras.Sequential(s)

# %%
model = tf.keras.Sequential([
    tfl.Input(shape=(DIM, DIM, CH), batch_size=BATCH),
    dimensionLoweringConvolutionBlock(filters=32, dropout=0.2),
    dimensionLoweringConvolutionBlock(filters=64, dropout=0.3),
    dimensionLoweringConvolutionBlock(filters=128, dropout=0.4),
    tfl.Flatten(),
    mlpHidden(neurons=128, dropout=0.5),
    tfl.Dense(len(classes), activation='softmax') # 10 classes, probability
])

# %%
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['sparse_categorical_accuracy'])

# %%
augmentation = tf.keras.Sequential([
    # tfl.RandomFlip("horizontal_and_vertical"),
    tfl.RandomRotation(0.2),
])

# %%
dsCount = len(ds)
trainCount = math.floor(0.8*dsCount)
testCount = math.floor(0.1*dsCount)
validateCount = dsCount - trainCount - testCount
ds = ds.shuffle(32)
train = ds.take(trainCount)
test = ds.skip(trainCount).take(testCount)
validate = ds.skip(trainCount+testCount).take(validateCount)
print(f"Training: {len(train)}\nTesting: {len(test)}\nValidation: {len(validate)}")

# %%
def normalize(img, label):
  image = tf.cast(img, tf.float32) / 255.
  return (image, label)

train = train\
.map(normalize)\
.cache()\
.shuffle(buffer_size=BATCH)\
.batch(BATCH, drop_remainder=True)\
.map(lambda img, label: (augmentation(img, training=True), label), num_parallel_calls=tf.data.AUTOTUNE)\
.prefetch(tf.data.AUTOTUNE)

test = test\
.map(normalize)\
.cache()\
.batch(BATCH, drop_remainder=True)\
.prefetch(tf.data.AUTOTUNE)

# %%

try:
    if TRAIN:
        raise Exception("Training forced.")
    model.load_weights("/drive/fashion_mnist.weights.h5")
except:
    print("No weights found. Training from scratch.")
    early_stopping = tcl.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tcl.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6)
    history = model.fit(train, epochs = EPOCHS, validation_data = test, callbacks=[early_stopping, reduce_lr], batch_size=BATCH)
    model.save_weights("/drive/fashion_mnist.weights.h5")

# %%
if "history" in locals():
    h=history.history
    plt.plot(h["loss"], label="Training Loss")
    plt.plot(h["val_loss"], label="Testing Loss")
    plt.legend()
    plt.show()

# %%
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

ds = validate.map(normalize).batch(BATCH,drop_remainder=True).prefetch(tf.data.AUTOTUNE)

y_true = []
y_pred = []

for images, labels in ds:
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(model.predict(images), axis=1))

conf_matrix = confusion_matrix(y_true, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred))


