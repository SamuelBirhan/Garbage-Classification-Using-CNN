from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print('GPU is set')

tf.config.list_physical_devices('GPU')

# Suppress warnings
import warnings

warnings.filterwarnings("ignore", message="You are saving your model as an HDF5 file.*")


# Data processing function
def processing_data(data_path):
    """Data processing."""
    # Data augmentation and preprocessing
    train_data = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.1
    )

    validation_data = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.1
    )

    train_generator = train_data.flow_from_directory(
        data_path,
        target_size=(150, 150),
        batch_size=50,
        class_mode='categorical',
        subset='training',
        seed=0
    )

    validation_generator = validation_data.flow_from_directory(
        data_path,
        target_size=(150, 150),
        batch_size=50,
        class_mode='categorical',
        subset='validation',
        seed=0
    )

    return train_generator, validation_generator


# Create a complex CNN model
def create_model(train_generator, validation_generator, save_model_path):
    """Create a more complex CNN model."""
    model = Sequential([
        # Layer 1
        Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Layer 2
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Layer 3
        Conv2D(256, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_generator,
        epochs=100,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stopping]
    )

    # Save the model
    model.save(save_model_path)

    return model, history


# Plot training history
def plot_history(history):
    """Plot training history."""
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.show()

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.show()


# Main function
def main():
    """Main function for training and evaluation."""
    data_path = "./data/Training_Data"  # Dataset path
    save_model_path = 'garbage_classifier.h5'  # Save model path and name

    # Load data
    train_generator, validation_generator = processing_data(data_path)

    # Create and train model
    trained_model, history = create_model(train_generator, validation_generator, save_model_path)

    # Plot training history
    plot_history(history)


if __name__ == '__main__':
    main()
