import os

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer


def load_data_from_folders(base_path):
    images = []
    labels = []

    for label in os.listdir(base_path):
        folder_path = os.path.join(base_path, label)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = Image.open(image_path)
                images.append(np.array(image).astype('float32'))
                labels.append(int(label))

    images = np.array(images).reshape(-1, 28 * 28)
    labels = np.array(labels)
    return images, labels


train_path = 'dáta/train'
test_path = 'dáta/test'

# Load the data
train_images, train_labels = load_data_from_folders(train_path)
test_images, test_labels = load_data_from_folders(test_path)

# Encoding data
encoder = LabelBinarizer(sparse_output=False)
train_labels_one_hot = encoder.fit_transform(train_labels.reshape(-1, 1))

test_labels_one_hot = encoder.transform(test_labels.reshape(-1, 1))

# Compute the weight matrix W
X = train_images
Y = train_labels_one_hot
W = np.linalg.pinv(X.T @ X) @ X.T @ Y


def predict(images, W):
    return np.argmax(images @ W, axis=1)


# evaluation
test_predictions = predict(test_images, W)

confusion_matrix = confusion_matrix(test_labels, test_predictions)


def print_confusion_matrix(p_confusion_matrix):
    print('Confusion matrix:')
    print('     0. 1. 2. 3. 4. 5. 6. 7. 8. 9.')
    for i in range(p_confusion_matrix.shape[0]):
        print(f'{i}. {p_confusion_matrix[i]}')


print_confusion_matrix(confusion_matrix)

# accuracy
accuracy = accuracy_score(test_labels, test_predictions)
print(f"Accuracy: {accuracy}")

# precision
precision = precision_score(test_labels, test_predictions, average='weighted')
print(f"Precision: {precision}")

# recall
recall = recall_score(test_labels, test_predictions, average='weighted')
print(f"Recall: {recall}")

# F1 score
f1 = f1_score(test_labels, test_predictions, average='weighted')
print(f"F1 Score: {f1}")
