import os

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer


def load_data(path):
    images = []
    labels = []

    for label in os.listdir(path):
        folder_path = os.path.join(path, label)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = Image.open(image_path)
                images.append(np.array(image).astype('float64'))
                labels.append(int(label))

    images = np.array(images).reshape(-1, 28 * 28)
    labels = np.array(labels)
    return images, labels


train_path = 'dáta/train'
test_path = 'dáta/test'

# Load the data
images_train, labels_train = load_data(train_path)
images_test, labels_test = load_data(test_path)

# Encoding data
encoder = LabelBinarizer(sparse_output=False)
train_labels_one_hot = encoder.fit_transform(labels_train.reshape(-1, 1))

test_labels_one_hot = encoder.transform(labels_test.reshape(-1, 1))

# weights -> 𝛽=(𝐴𝑇𝐴)^-1 * 𝐴𝑇𝑌
X = images_train
Y = train_labels_one_hot
inverse_matrix = np.linalg.pinv(X.T @ X)
W = inverse_matrix @ X.T @ Y


def predict_data(images, W):
    return np.argmax(images @ W, axis=1)


# evaluation
test_predictions = predict_data(images_test, W)

confusion_matrix = confusion_matrix(labels_test, test_predictions)


def print_confusion_matrix(p_confusion_matrix):
    print('Confusion matrix:')
    print('     0. 1. 2. 3. 4. 5. 6. 7. 8. 9.')
    for i in range(p_confusion_matrix.shape[0]):
        print(f'{i}. {p_confusion_matrix[i]}')


print_confusion_matrix(confusion_matrix)

# accuracy
accuracy = accuracy_score(labels_test, test_predictions)
print(f"Accuracy: {accuracy * 100}%")

# precision
precision = precision_score(labels_test, test_predictions, average='weighted')
print(f"Precision: {precision * 100}%")

# recall
recall = recall_score(labels_test, test_predictions, average='weighted')
print(f"Recall: {recall * 100}%")

# error rate
error_rate = 100 - (accuracy * 100)
print(f'Error rate: {error_rate}%')
