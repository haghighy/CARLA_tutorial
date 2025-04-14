import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIRECTORY = 'dataset'
CATEGORIES = ['rain', 'fog', 'night', 'day']
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

def load_images_and_labels():
    images_list = []
    labels_list = []
    label_mapping = {category: idx for idx, category in enumerate(CATEGORIES)}

    for category in CATEGORIES:
        category_dir = os.path.join(DATA_DIRECTORY, category)
        if not os.path.exists(category_dir):
            print(f"Directory {category_dir} is missing!")
            continue

        for img_filename in os.listdir(category_dir):
            img_filepath = os.path.join(category_dir, img_filename)
            try:
                image = load_img(img_filepath, target_size=IMAGE_SIZE)
                image_array = img_to_array(image)
                image_array = preprocess_input(image_array)
                images_list.append(image_array)
                labels_list.append(label_mapping[category])
            except Exception as e:
                print(f"Error loading image {img_filepath}: {e}")
                continue

    images_array = np.array(images_list)
    labels_array = np.array(labels_list)
    return images_array, labels_array

def create_weather_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_metrics(training_history):
    plt.figure(figsize=(14, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(training_history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Accuracy During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(training_history.history['loss'], label='Train Loss', color='blue')
    plt.plot(training_history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    print("Training results saved as 'training_results.png'")

def evaluate_model_and_confusion_matrix(model, X_test, y_test):
    predictions = model.predict(X_test, batch_size=BATCH_SIZE)
    predicted_classes = np.argmax(predictions, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, predicted_classes, target_names=CATEGORIES))

    cm = confusion_matrix(y_test, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.savefig('confusion_matrix_results.png')
    print("Confusion matrix saved as 'confusion_matrix_results.png'")

def run_training_and_evaluation():
    print("Loading dataset...")
    X, y = load_images_and_labels()
    print(f"Loaded {len(X)} images with shape {X.shape}")

    # Splitting the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    print(f"Training data size: {len(X_train)} images")
    print(f"Validation data size: {len(X_val)} images")
    print(f"Test data size: {len(X_test)} images")

    print("Creating model...")
    model = create_weather_model(num_classes=len(CATEGORIES))

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint_callback = ModelCheckpoint('best_weather_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop_callback, model_checkpoint_callback],
        verbose=1
    )

    print("\nEvaluating the model on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    plot_metrics(history)
    evaluate_model_and_confusion_matrix(model, X_test, y_test)

    model.save('final_weather_model.h5')
    print("Final model saved as 'final_weather_model.h5'")

if __name__ == '__main__':
    run_training_and_evaluation()
