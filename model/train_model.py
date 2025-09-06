import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import os

# Configuration
data_dir = r"C:\Users\tishya\OneDrive\Desktop\days_left_for_banana_die\dataset"
img_size = (224, 224)
batch_size = 16
class_names = ['overripe', 'ripe', 'rotten', 'unripe']
num_classes = len(class_names)


def preprocess_data(images, labels):
    """Preprocessing function for EfficientNet"""
    images = tf.cast(images, tf.float32)
    images = keras.applications.efficientnet.preprocess_input(images)
    return images, labels


def create_datasets():
    """Create and preprocess datasets"""
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        color_mode="rgb"
    ).map(preprocess_data, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    val_ds = keras.utils.image_dataset_from_directory(
        f"{data_dir}/valid",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        color_mode="rgb"
    ).map(preprocess_data, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    test_ds = keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        color_mode="rgb"
    ).map(preprocess_data, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_model():
    """Build the EfficientNet-based model"""
    tf.keras.backend.clear_session()
    input_shape = (224, 224, 3)

    # Load base model with error handling
    try:
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )
        print("Successfully loaded EfficientNetB0 with ImageNet weights")
    except Exception as e:
        print(f"Error loading EfficientNetB0: {e}")
        print("Trying alternative approach...")

        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=input_shape
        )

        try:
            base_model.load_weights(keras.utils.get_file(
                'efficientnetb0_notop.h5',
                'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/efficientnetb0_notop.h5',
                cache_subdir='models'
            ), by_name=True)
            print("Successfully loaded weights manually")
        except:
            print("Using model without pretrained weights")

    base_model.trainable = False

    # Build model architecture
    inputs = keras.Input(shape=input_shape, name='input_layer')

    # Data augmentation layers
    x = inputs
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    x = layers.RandomHeight(0.1)(x)
    x = layers.RandomWidth(0.1)(x)
    x = layers.RandomBrightness(0.1)(x)

    # Feature extraction and classification
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation="softmax", name='predictions')(x)

    model = keras.Model(inputs, outputs, name='banana_classifier')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def train_model(model, train_ds, val_ds):
    """Train the model with proper callbacks"""
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Robust callbacks to avoid pickle issues
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Add ModelCheckpoint with error handling
    try:
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "banana_model_best.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))
    except:
        print("ModelCheckpoint may cause issues, using simplified callbacks")

    print("Starting initial training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    return history


def fine_tune_model(model, base_model, train_ds, val_ds, history):
    """Fine-tune the model"""
    print("Starting fine-tuning...")

    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    simple_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,  # Reduced for efficiency
        initial_epoch=len(history.history['loss']),
        callbacks=simple_callbacks,
        verbose=1
    )

    return fine_tune_history


def evaluate_model(model, test_ds):
    """Evaluate model and create visualizations"""
    print("Evaluating model...")

    # Get overall performance
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")

    # Generate predictions for detailed analysis
    y_true = []
    y_pred = []

    for batch_num, (images, labels) in enumerate(test_ds):
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

        if batch_num % 5 == 0:
            print(f"Processed batch {batch_num + 1}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Create confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Final Accuracy: {test_acc:.3f}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    # Print classification report
    print("\nClassification Report:")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Per-class accuracy visualization
    accuracy_by_class = []
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_accuracy = np.sum((y_pred == i) & class_mask) / np.sum(class_mask)
            accuracy_by_class.append(class_accuracy)
            print(f"{class_name.capitalize()}: {class_accuracy:.3f} accuracy")

    # Create per-class accuracy chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, accuracy_by_class,
                   color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    plt.ylabel('Accuracy')
    plt.title(f'Per-Class Accuracy (Overall: {test_acc:.3f})')
    plt.ylim(0, 1.0)

    for bar, acc in zip(bars, accuracy_by_class):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("per_class_accuracy.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    return test_acc


def save_model(model):
    """Save the trained model"""
    print("Saving model...")

    try:
        model.save("banana_model_final.keras")
        print("Model saved as 'banana_model_final.keras'")
    except Exception as e:
        print(f"Error saving model: {e}")


def main():
    """Main training pipeline"""
    print("Banana Classification Model Training")
    print("=" * 40)

    # Create datasets
    train_ds, val_ds, test_ds = create_datasets()
    print("Datasets created successfully!")

    # Build model
    model, base_model = build_model()
    print("Model built successfully!")
    model.summary()

    # Train model
    history = train_model(model, train_ds, val_ds)
    print("Initial training completed!")

    # Fine-tune model
    fine_tune_history = fine_tune_model(model, base_model, train_ds, val_ds, history)
    print("Fine-tuning completed!")

    # Evaluate model
    test_acc = evaluate_model(model, test_ds)

    # Save model
    save_model(model)

    # Final summary
    print("\nTraining Complete!")
    print("=" * 40)
    print("Generated files:")
    print("- banana_model_final.keras (Your trained model)")
    print("- confusion_matrix.jpg (Performance matrix)")
    print("- per_class_accuracy.jpg (Per-class performance)")
    print(f"\nFinal Test Accuracy: {test_acc:.1%}")
    print(f"Model Status: {'Excellent' if test_acc > 0.9 else 'Good' if test_acc > 0.8 else 'Needs Improvement'}")

    print("\nTo use your model:")
    print("model = keras.models.load_model('banana_model_final.keras')")


if __name__ == "__main__":
    main()