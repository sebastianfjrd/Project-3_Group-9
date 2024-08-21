# Project-3_Group-9
Project 3
FINAL PROJECT REQUIREMENTS:
Using CNN model  to attack Skin Cancer! 

This project focuses on detecting melanoma through images. Melanoma is a dangerous form of skin cancer. Early detection and accurate diagnosis are crucial for patient outcomes, making this a problem worth solving.
Using Kaggle, we found a large dataset that was sufficiently large enough to effectively train our ML model or neural network with a high degree of accuracy, to ensure that your results are reliable.

The dataset used in this project consists of 10,605 images of skin lesions, categorized as benign or malignant. This dataset is sufficiently large enough for training a convolutional neural network (CNN) model with high accuracy.
Evaluate the trained model(s) using testing data. Include any calculations, metrics, or visualizations needed to evaluate the performance.

The model’s performance is evaluated using accuracy, precision, recall, F1 score, AUC-ROC, and a confusion matrix. Visualizations like the ROC curve and Precision-Recall curve are also included to assess model performance.
We used 
TensorFlow/Keras: Used to build, train, and evaluate the CNN model for melanoma detection.
We also used an additional library or technology which was not covered in class, such as:

Potential to integrate PyTorch or Whisper (OpenAI’s automatic speech recognition system), or other advanced technologies for future iterations of this project, like
Gradio speech to text. 

PROJECT SUMMARY:
Dataset: The project utilizes a melanoma skin cancer dataset containing 10,605 images labeled as benign and malignant. These images were collected from various sources and are split into training, validation, and test sets.

Data Preprocessing:

Images are resized to 300x300 pixels.
Data normalization is applied by scaling pixel values to the range [0, 1].
Data augmentation techniques, including random rotation, translation, zoom, and flipping, are used to increase the diversity of the training set and improve the model’s accuracy.

Model Building:

A custom Convolutional Neural Network (CNN) architecture is built from scratch using TensorFlow/Keras.
The architecture includes multiple convolutional layers followed by max pooling, flattening, and fully connected layers.
Regularization techniques like Dropout and L2 regularization are applied to prevent overfitting.

Model Training:

The model is compiled using the Adam optimizer and categorical cross-entropy loss.
Training is performed over multiple epochs, with early stopping and learning rate reduction callbacks to optimize model performance.

RESULTS AND CONCLUSIONS:
Output:

The model achieved a test accuracy of 92.01%, with corresponding precision, recall, F1 score, and AUC-ROC metrics that indicate strong performance in detecting melanoma from images.

Model Details:

Custom CNN Architecture: Built from scratch, featuring convolutional layers for feature extraction, max pooling for down-sampling, and dense layers for classification.
Regularization: Applied Dropout and L2 regularization to prevent overfitting.
Callbacks: Utilized ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau to enhance the model’s performance and prevent overfitting.

Test Set Evaluation:

The model’s performance is evaluated on a separate test set, with the accuracy, loss, confusion matrix, and classification report providing insights into its effectiveness.
Future Improvements:

Incorporate advanced architectures like ResNet or EfficientNet, or explore transfer learning with pretrained models to further enhance accuracy.
Experiment with additional data augmentation techniques or synthetic data generation to improve model generalization.
Prediction and Visualization:

The final model’s predictions are evaluated using confusion matrices and classification reports.
ROC and Precision-Recall curves are plotted to visualize the model’s performance across different thresholds.
