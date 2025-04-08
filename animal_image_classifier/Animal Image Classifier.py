#Project Name: Animal Image Classifier (Computer Vision with TensorFlow & Keras)
from pathlib import Path
from sklearn import metrics

# Define path to your dataset
dataset_path = Path("/Users/hasanismayilov/Downloads/archive-5/raw-img")

# Translation dictionary (Italian → English)
translate = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", 
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", 
    "scoiattolo": "squirrel", "spider": "spider", "ragno": "spider"
}
import os

for folder in dataset_path.iterdir():
    if folder.is_dir():
        italian = folder.name.lower()
        english = translate.get(italian)
        if english and english != italian:
            new_folder = dataset_path / english
            if not new_folder.exists():
                folder.rename(new_folder)
                print(f"Renamed '{italian}' → '{english}'")
            else:
                print(f"Folder '{english}' already exists. Skipping.")

import tensorflow as tf
from tensorflow import keras

img_height = 180
img_width = 180
batch_size = 32
seed = 123

# Load the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Class Labels:", class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)




from tensorflow.keras import layers 

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


model = keras.Sequential([
    data_augmentation,  # 👈 Apply random transformations
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dropout(0.5),  # 👈 Reduces overfitting
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])


model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

epochs = 10
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs
)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np

for images, labels in val_ds.take(1):
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    plt.figure(figsize=(15, 8))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        actual = class_names[labels[i]]
        predicted = class_names[predicted_classes[i]]

        color = "green" if actual == predicted else "red"
        plt.title(f"Actual: {actual}\nPredicted: {predicted}", color=color)
        plt.axis("off")
    break

model.save("animal_classifier_model.h5")

