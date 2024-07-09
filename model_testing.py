import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_test_data(data_dir, target_size=(150, 150), batch_size=32):
    """Load test data using flow_from_directory."""
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,  # No labels will be returned
        shuffle=False  # Ensure the order is maintained
    )

    return test_generator, test_generator.samples


def predict(model_path, test_generator):
    """
    Perform prediction on test data generator
    """
    # Load the saved model
    model = load_model(model_path)

    # Perform prediction on test data
    predictions = model.predict(test_generator, steps=len(test_generator))

    return predictions


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, fontsize=14):
    """
    Plot the confusion matrix with the specified class order
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap=cmap, fmt='g')

    # Shift the x and y ticks to the middle of the cells
    plt.xticks(ticks=np.arange(len(classes)) + 0.5, labels=classes, rotation=45, fontsize=fontsize)
    plt.yticks(ticks=np.arange(len(classes)) + 0.5, labels=classes, fontsize=fontsize)

    plt.xlabel('Predicted labels', fontsize=fontsize)
    plt.ylabel('True labels', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.show()


def plot_precision_recall_curve(y_true, y_scores, class_names):
    """
    Plot the precision-recall curve for each class
    """
    plt.figure(figsize=(12, 8))

    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        average_precision = average_precision_score(y_true[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP={average_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()


def plot_precision_confidence_curve(y_true, y_scores, class_names):
    """
    Plot the precision-confidence curve for each class
    """
    plt.figure(figsize=(12, 8))

    for i, class_name in enumerate(class_names):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])
        plt.plot(thresholds, precision[:-1], lw=2, label=f'{class_name}')

    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.title('Precision-Confidence Curve')
    plt.legend(loc='best')
    plt.show()


def plot_f1_confidence_curve(y_true, y_scores, class_names):
    """
    Plot the F1-confidence curve for each class
    """
    plt.figure(figsize=(12, 8))

    for i, class_name in enumerate(class_names):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall)
        plt.plot(thresholds, f1_scores[:-1], lw=2, label=f'{class_name}')

    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1-Confidence Curve')
    plt.legend(loc='best')
    plt.show()


def main():
    model_path = 'garbage_classifier.h5'
    test_data_dir = './data/Testing_Data'  # Directory containing test images

    # Get unique class names and assign indices
    class_names = sorted(os.listdir(test_data_dir))

    # Load test data
    test_generator, num_samples = load_test_data(test_data_dir)

    # Perform prediction on test data
    y_scores = predict(model_path, test_generator)

    # Convert predicted probabilities to class labels
    y_pred_labels = np.argmax(y_scores, axis=1)

    # Create true_labels based on the directory structure
    true_labels = test_generator.classes
    y_true = np.eye(len(class_names))[true_labels]

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, y_pred_labels)

    # Plot the confusion matrix
    plot_confusion_matrix(cm, classes=class_names)

    # Print classification report
    print(classification_report(true_labels, y_pred_labels, target_names=class_names))

    # Plot precision-recall curve
    plot_precision_recall_curve(y_true, y_scores, class_names)

    # Plot precision-confidence curve
    plot_precision_confidence_curve(y_true, y_scores, class_names)

    # Plot F1-confidence curve
    plot_f1_confidence_curve(y_true, y_scores, class_names)


if __name__ == '__main__':
    main()
