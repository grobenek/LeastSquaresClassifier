import os

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix


def load_data_from_folders(base_path):
    images = []
    labels = []

    # Loop over the folders
    for label in sorted(os.listdir(base_path)):
        # Ensure that the label is a directory
        folder_path = os.path.join(base_path, label)
        if os.path.isdir(folder_path):
            # Loop over the images in the folder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = Image.open(image_path)
                images.append(np.array(image).astype('float32'))
                labels.append(int(label))  # The folder name represents the label

    # Reshape images and convert labels to a numpy array
    images = np.array(images).reshape(-1, 28 * 28)  # Assuming all images are 28x28
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


# Function to predict labels
def predict(images, W):
    return np.argmax(images @ W, axis=1)


# Evaluate the model
test_predictions = predict(test_images, W)

confusion_matrix = confusion_matrix(test_labels, test_predictions)


def print_confusion_matrix(p_confusion_matrix):
    print('Confusion matrix:')
    print('     0. 1. 2. 3. 4. 5. 6. 7. 8. 9.')
    for i in range(p_confusion_matrix.shape[0]):
        print(f'{i}. {p_confusion_matrix[i]}')


print_confusion_matrix(confusion_matrix)
