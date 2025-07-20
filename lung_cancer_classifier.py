import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Image size and batch
img_height, img_width = 128, 128
batch_size = 32

# Set dataset directory
dataset_dir = os.path.join(os.getcwd(), 'Dataset')
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'test')

# Print GPU info
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model building
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(3, activation='softmax')  # Update to 2 if you have 2 classes
    ])
    return model

# Plotting training
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Predict folder of images
def predict_folder(folder_path, model_path='lung_cancer_classifier.h5'):
    model = load_model(model_path)

    pred_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    pred_generator = pred_datagen.flow_from_directory(
        folder_path,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None,
        shuffle=False
    )

    predictions = model.predict(pred_generator)
    class_indices = train_generator.class_indices
    label_map = {v: k for k, v in class_indices.items()}
    predicted_classes = np.argmax(predictions, axis=1)

    print("\n--- Predictions ---")
    for i, pred_class in enumerate(predicted_classes):
        file_name = pred_generator.filenames[i]
        confidence = predictions[i][pred_class]
        print(f"{file_name}: Predicted as {label_map[pred_class]} with confidence {confidence:.2f}")

# Main execution
def main(image_folder=None):
    model_file = 'lung_cancer_classifier.h5'

    if not os.path.exists(model_file):
        print("[INFO] Training model as .h5 file not found...")

        model = build_model()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(
            train_generator,
            epochs=5,
            validation_data=validation_generator
        )

        test_loss, test_acc = model.evaluate(test_generator)
        print(f'Test accuracy: {test_acc:.4f}')
        model.save(model_file)

        plot_training_history(history)
    else:
        print("[INFO] Model already exists. Skipping training.")

    if image_folder:
        predict_folder(image_folder, model_file)

# CLI argument parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to folder with subfolders of images for prediction (e.g., malignant, benign)', default=None)
    args = parser.parse_args()

    main(args.image)
