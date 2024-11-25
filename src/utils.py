import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix, precision_score, recall_score, RocCurveDisplay
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import random

def generate_data_paths_with_label(data_directory):
    filepaths = []
    labels = []

    folders = os.listdir(data_directory)
    for folder in folders:
        folder_path = os.path.join(data_directory, folder)
        filelist = os.listdir(folder_path)
        for file in filelist:
            if 'mask' not in file:
                fpath = os.path.join(folder_path, file)
                filepaths.append(fpath)
                labels.append(folder)

    # Concatenate data paths with labels into one dataframe
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df

# Reference : https://www.kaggle.com/code/aditimondal23/vgg19-breast
# Function to evaluate the model
def evaluation(model, x_train, y_train, x_val, y_val, x_test, y_test, history):
    train_loss, train_acc = model.evaluate(x_train, y_train.toarray())
    val_loss, val_acc = model.evaluate(x_val, y_val.toarray())
    test_loss_value, test_accuracy = model.evaluate(x_test, y_test.toarray())

    y_pred = model.predict(x_test)
    y_pred_label = np.argmax(y_pred, axis=1)
    y_true_label = np.argmax(y_test.toarray(), axis=1)

    f1_measure = f1_score(y_true_label, y_pred_label, average='weighted')
    roc_score = roc_auc_score(y_test.toarray(), y_pred)
    kappa_score = cohen_kappa_score(y_true_label, y_pred_label)
    precision = precision_score(y_true_label, y_pred_label, average='weighted')
    recall = recall_score(y_true_label, y_pred_label, average='weighted')
    cm = confusion_matrix(y_true_label, y_pred_label)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {f1_measure:.4f}")
    print(f"Kappa Score: {kappa_score:.4f}")
    print(f"ROC AUC Score: {roc_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return y_true_label, y_pred, cm

# Plotting function for model performance metrics with improved alignment and spacing
def Plotting(encoder, acc, val_acc, loss, val_loss, y_true, y_pred, cm):
    # Plot accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Model's Metrics Visualization", fontsize=16)

    # Accuracy Plot
    ax1.plot(range(1, len(acc) + 1), acc, label='Training Accuracy', color='blue', linestyle='-')
    ax1.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy', color='orange', linestyle='--')
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    # Loss Plot
    ax2.plot(range(1, len(loss) + 1), loss, label='Training Loss', color='red', linestyle='-')
    ax2.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', color='green', linestyle='--')
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Adjust space between accuracy and loss plots
    plt.show()
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.categories_[0])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)  # Remove grid for clarity
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # Adjust spacing for clarity
    plt.show()

    # Plot ROC Curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(encoder.categories_[0])):
        fpr, tpr, _ = roc_curve(y_true, y_pred[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve for {encoder.categories_[0][i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Calculate precision, recall, and F1 score for each class
    precision = precision_score(y_true, y_pred.argmax(axis=1), average=None)
    recall = recall_score(y_true, y_pred.argmax(axis=1), average=None)
    f1 = f1_score(y_true, y_pred.argmax(axis=1), average=None)

    # Bar plot for precision, recall, and F1 score with improved spacing
    metrics = [precision, recall, f1]
    metrics_names = ['Precision', 'Recall', 'F1 Score']
    class_labels = encoder.categories_[0]

    x = np.arange(len(class_labels))  # Label locations
    width = 0.2  # Width of the bars
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, metric, width, label=metrics_names[i])

    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Precision, Recall, and F1 Score for Each Class')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_labels)
    ax.legend()
    plt.grid(axis='y')  # Add gridlines for readability
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)  # Adjust space around bar plots
    plt.show()

# Fit and evaluate the model, and visualize the performance
def fit_evaluate(encoder, model, x_train, y_train, x_test, y_test, bs, Epochs, patience):
    # Early stopping to prevent overfitting
    es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True, verbose=1)
    # Model checkpoint to save the best model based on validation accuracy
    mc = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

    # Split training data further into train and validation sets
    x1_train, x_val, y1_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42,
                                                        stratify=y_train.toarray())
    # Fit the model
    history = model.fit(x1_train, y1_train.toarray(),
                        validation_data=(x_val, y_val.toarray()),
                        epochs=Epochs,
                        batch_size=bs,
                        callbacks=[es, mc])

    # Retrieve history for training and validation metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # Evaluate the model and collect true/predicted labels
    y_true, y_pred, cm = evaluation(model, x1_train, y1_train, x_val, y_val, x_test, y_test, history)

    # Plot training history, ROC curves, and class-specific precision, recall, and F1 score
    Plotting(encoder, acc, val_acc, loss, val_loss, y_true, y_pred, cm)


# Function to visualize multiple predictions from the dataset with added comments
def visualize_model_performance(model, x_test, y_test, class_labels=['benign', 'normal', 'malignant'], num_samples=12):
    # Select random samples from the test set
    indices = random.sample(range(len(x_test)), num_samples)
    test_images = x_test[indices]
    true_labels = y_test.toarray()[indices]  # Convert sparse matrix to array if needed
    true_labels = np.argmax(true_labels, axis=1)  # Get true class indices

    # Make predictions on the selected test samples
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)  # Get predicted class indices
    # Set up the plotting layout for 4 images per row
    plt.figure(figsize=(18, 12))
    rows = num_samples // 4 + int(num_samples % 4 != 0)  # Calculate the required number of rows

    for i, (img, true_label, pred_label, pred_prob) in enumerate(
            zip(test_images, true_labels, predicted_labels, predictions)):
        plt.subplot(rows, 4, i + 1)  # Arrange images in 4 columns
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"True: {class_labels[true_label]}\nPred: {class_labels[pred_label]} ({max(pred_prob) * 100:.2f}%)",
                  fontsize=12, color="blue" if true_label == pred_label else "red")  # Color title based on correctness

    plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Adjust spacing between plots
    plt.suptitle("Model Predictions on Test Images", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


