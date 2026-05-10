# 🐶 End-to-End Multi-class Dog Breed Classification

This project implements an end-to-end multi-class image classifier using TensorFlow 2.x and TensorFlow Hub to identify the breed of a dog from an image.

## 1. Problem Statement
Given an image of a dog, the goal is to accurately identify its breed. This is a common real-world problem, for example, identifying a dog's breed from a photo taken at a cafe.

## 2. Data
The dataset used for this project is from Kaggle's Dog Breed Identification competition.

**Dataset Source:** [Kaggle Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification)

**Key Data Characteristics:**
*   **Type:** Unstructured image data.
*   **Classes:** 120 unique dog breeds.
*   **Training Set:** Approximately 10,000+ images with labels.
*   **Test Set:** Approximately 10,000+ images without labels (for prediction).

## 3. Evaluation
Submissions are evaluated on **Multi-class Log Loss** between the predicted probability and the observed target.

## 4. Features
*   **Deep Learning / Transfer Learning:** Due to the nature of image data, deep learning, specifically transfer learning, is employed.
*   **120 Dog Breeds:** The model is designed to classify images into one of 120 distinct dog breeds.
*   **TensorFlow 2.x & TensorFlow Hub:** Utilizes modern TensorFlow features and pre-trained models for efficiency and performance.

## 5. Project Setup and Dependencies

To run this notebook, you'll need the following libraries:

*   `tensorflow`
*   `tensorflow_hub`
*   `tf_keras` (imported as `keras`)
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`

You can install them using pip:
```bash
pip install tensorflow tensorflow_hub tf_keras pandas numpy matplotlib seaborn scikit-learn
```

**Data Unzipping:** The raw dataset needs to be unzipped into your working directory. The notebook assumes the data is unzipped into `/content/drive/MyDrive/Colab Notebooks/Dog Vision/`.

```python
# Example of unzipping the dataset
# !unzip "/content/drive/MyDrive/Colab Notebooks/Dog Vision/dog-breed-identification.zip"
```

**GPU Check:** Ensure a GPU is available for faster training.
```python
import tensorflow as tf
print("GPU", "available (YES....!!)" if tf.config.list_physical_devices("GPU") else "Not available :(")
```

## 6. Methodology

### 6.1 Data Preparation and Preprocessing

1.  **Loading Labels:** `labels.csv` is loaded using Pandas to get image IDs and their corresponding dog breeds.
2.  **Filename Generation:** Image file paths are constructed from the image IDs.
3.  **Label Encoding:** Breed names are converted into numerical boolean arrays (one-hot encoding).
4.  **Validation Set Creation:** A custom validation set is created from the training data (e.g., 80/20 split for initial experiments).
5.  **Image Preprocessing Function (`process_image`):**
    *   Reads image files from paths.
    *   Decodes JPEG images to 3 color channels.
    *   Normalizes pixel values from `0-255` to `0-1`.
    *   Resizes images to `224x224` (IMG_SIZE).
6.  **Batching (`create_data_batches`):**
    *   Converts image file paths and labels into `(image, label)` TensorFlow `tf.data.Dataset` tuples.
    *   Shuffles training data for randomness.
    *   Batches data for efficient model processing (e.g., `BATCH_SIZE = 32`).

### 6.2 Model Architecture

A transfer learning model is built using a pre-trained MobileNetV2 from TensorFlow Hub.

*   **Base Model:** `https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5`
*   **Architecture:**
    *   `hub.KerasLayer` (input layer from TensorFlow Hub)
    *   `keras.layers.Dense` output layer with `softmax` activation for multi-class classification.
*   **Compilation:**
    *   `loss`: `CategoricalCrossentropy`
    *   `optimizer`: `Adam`
    *   `metrics`: `accuracy`

### 6.3 Model Training

1.  **Callbacks:**
    *   **TensorBoard:** For logging and visualizing training progress (`create_tensorboard_callback`).
    *   **EarlyStopping:** To prevent overfitting, monitoring `val_accuracy` for initial training or `accuracy` for full training with a `patience` of 3 epochs.
2.  **Training Procedure (`train_model`):**
    *   Initial training is performed on a subset of data (e.g., 1000 images) to ensure functionality.
    *   Full training is then performed on the entire dataset.

### 6.4 Making and Evaluating Predictions

1.  **Prediction Generation:** The trained model makes predictions on validation and test data, outputting probability arrays.
2.  **Label Conversion:** A helper function (`get_pred_label`) converts probability arrays to the predicted dog breed label.
3.  **Visualization (`plot_pred`, `plot_pred_conf`):** Functions are created to visualize individual predictions, showing the image, predicted label, prediction probability, and true label. Another function plots the top 10 prediction confidences.
4.  **Confusion Matrix:** A confusion matrix is generated using `sklearn.metrics.confusion_matrix` and visualized with `seaborn.heatmap` to assess model performance across all classes.

### 6.5 Saving and Reloading Models

Functions (`save_model`, `load_model`) are implemented to save the trained model (in `.keras` format) and reload it for future use or deployment.

### 6.6 Making Predictions on Custom Images

The workflow includes processing custom images (from a user-defined directory) through the `create_data_batches` function (with `test_data=True`), making predictions with the loaded model, and visualizing the predicted labels on the custom images.

## 7. Results and Submission

The final predictions on the test dataset are formatted into a Pandas DataFrame with an 'id' column and columns for each dog breed's prediction probability. This DataFrame is then saved as a CSV file for submission to the Kaggle competition.

**Example Submission Format (first few rows):**
```
id,affenpinscher,afghan_hound,african_hunting_dog,airedale,...
image_id_1,0.0001,0.0005,0.00001,0.0002,...
image_id_2,0.00002,0.00008,0.00015,0.00003,...
...
```

## 8. Usage

1.  **Clone the Repository:** Download the project files.
2.  **Download Data:** Obtain the `dog-breed-identification.zip` from Kaggle and unzip it into the specified path.
3.  **Run the Notebook:** Execute the Jupyter/Colab notebook cells sequentially.
4.  **Custom Images:** Place your own dog images in a directory (e.g., `my_dogs/`) and update `custom_path` to get predictions.
