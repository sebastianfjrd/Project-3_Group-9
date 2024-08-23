# Melanoma /Skin Cancer Detection Using CNN Detection Project

## **SkinLink for Melanoma Detection**

##**Executive Summary**
This project focuses on detecting melanoma, a dangerous form of skin cancer, through image analysis using a Convolutional Neural Network (CNN). The dataset sourced from Kaggle used for training consisted of 10,605 images of skin lesions, categorized as benign or malignant. The model's performance is evaluated using accuracy, precision, recall, F1 score, AUC-ROC, and a confusion matrix. The goal is to provide an automated system that assists both patients and medical professionals in the early detection of melanoma, potentially improving survival rates.

**Early Detection is Key**
Skin cancer is the most common type of cancer. Early detection is crucial for effective treatment. Visual examination by a dermatologist, followed by a biopsy, is the standard procedure for diagnosis. This project aims to provide a tool that helps in the early detection of melanoma by analyzing images of skin lesions.

**Benefits to Patients & Medical Professionals**
For Patients: Faster and more accurate diagnosis can lead to earlier intervention, reducing the risk of advanced disease and improving survival rates. Patients gain quicker access to diagnostic results, which can be life-saving.

**For Medical Professionals##: Improved diagnostic tools can reduce dermatologists' workload, allow for more efficient use of healthcare resources, and help in standardizing diagnosis across different settings. This can lead to cost savings and better patient outcomes.

**Project Summary**
Dataset
The project utilizes a melanoma skin cancer dataset containing 10,605 images labeled as benign and malignant. These images were collected from various sources and are split into training, validation, and test sets.

***Gradio Applications***
1. User Application
The first Gradio application is designed for users (patients). In this application, users can upload images of their skin lesions. The app then utilizes the trained CNN model to predict whether the lesion is benign or malignant. This quick and accessible tool allows users to get an initial assessment of their skin condition, which can prompt further medical consultation if necessary.

Features:
Image Upload: Users can easily upload images of their skin lesions.
Prediction: The app predicts whether the lesion is benign or malignant.
User Feedback: Users receive immediate feedback, helping them understand their skin health and whether they should seek further medical advice.

<img src="https://github.com/user-attachments/assets/f9897e23-d224-4831-b7e3-94093a7093a8" alt="Model Performance Metrics" width="600" height="800"/>

**2. Doctor Application**
The second Gradio application is tailored for doctors. This application enables medical professionals to review the images uploaded by their patients and provide a more detailed analysis. The doctor can examine the predicted results, and using a voice memo feature, they can record and send personalized feedback to the patient. This system streamlines the communication between patients and doctors, facilitating quicker and more efficient follow-up.

**Features:**
Image Review: Doctors can access and review the images uploaded by patients.
Voice Memo: Doctors can record a voice memo with their analysis and send it directly to the patient, providing a personal touch and detailed feedback.
Enhanced Communication: The app bridges the gap between patient self-assessment and professional medical consultation.

<img src="https://github.com/user-attachments/assets/10b9f8e5-a2f3-429e-b971-4ca7f4bb9b8f" alt="Physician's Assistant for Melanoma Detection" width="1000" height="800"/>

**Data Preprocessing**
Resizing: Images are resized to 300x300 pixels.
Normalization: Pixel values are scaled to the range [0, 1].
Data Augmentation: Techniques such as random rotation, translation, zoom, and flipping are applied to increase the diversity of the training set and improve the model’s accuracy.

**Model Building**
A custom CNN architecture was built from scratch using TensorFlow/Keras. The architecture includes multiple convolutional layers followed by max pooling, flattening, and fully connected layers. Regularization techniques like Dropout and L2 regularization were applied to prevent overfitting.

**Model Training**
<u>Optimizeu<u>: Adam optimizer
<u>Loss Function<u>: Categorical cross-entropy loss
<u>Training<u>: Performed over multiple epochs with early stopping and learning rate reduction callbacks to optimize performance.

**Making & Optimizing the Model**
Title: Making & Optimizing the Model
Subtitle: How We Refined Our CNN Model to Ensure High Accuracy & Effective Generalization, Preparing It for Real-World Applications

![Alt Text](https://github.com/sebastianfjrd/Project-3_Group-9/blob/main/Making%20%26%20Optimizing%20the%20Model.png)

This image outlines the steps taken in refining the CNN model, from defining the architecture to fine-tuning the model for optimal performance. The process included the use of various callbacks, training and model evaluation techniques, and fine-tuning with TensorFlow Keras optimizers to achieve a high accuracy score of 93%.

**Evaluation**
<u>Performance Metrics<u>: The model achieved a test accuracy of 92.01%, with corresponding precision, recall, F1 score, and AUC-ROC metrics.
<u>Visualizations<u>: ROC and Precision-Recall curves were plotted to assess model performance. Confusion matrix visualizations were also used.

**Model Performance Metrics Overview**
Title: Model Performance Metrics Overview
Subtitle: Evaluating Accuracy, Precision, and Loss Across Training Epochs

![Alt Text](https://github.com/sebastianfjrd/Project-3_Group-9/blob/main/model_performance_optimization.png)

This image showcases the model's performance metrics, including the ROC curve, Precision-Recall curve, training accuracy, and training loss over multiple epochs. The high AUC-ROC and precision-recall scores indicate strong model performance, while the accuracy and loss plots highlight the model's training and validation performance over time.

**Future Research & Improvements**
<u>Advanced Architectures<u>: Consider incorporating architectures like ResNet or EfficientNet, or applying transfer learning with pretrained models.
<u>Additional Data Augmentation<u>: Experiment with additional data augmentation techniques or synthetic data generation to improve model generalization.
<u>Integration of Advanced Technologies<u>: Potential to integrate technologies like PyTorch, Whisper (OpenAI’s automatic speech recognition system), or Gradio for enhanced user interaction.

**Gradio Application**: SkinLink, the Melanoma Detector

**Overview**
SkinLink is a user-friendly Gradio application that allows patients to upload images of their skin lesions for analysis. The application uses a trained CNN model to predict the likelihood of melanoma. The results can be reviewed by doctors, who can then provide feedback to the patients. The app includes a bonus feature: language translation, enabling doctors to communicate results to patients who may not speak English.

**Features**

Image Upload: Patients can upload images of their skin lesions.
Prediction: The model predicts whether the lesion is benign or malignant.
Doctor Feedback: Doctors can review the results and provide feedback.
Language Translation: Translation feature to help communicate with non-English speaking patients.


**Future Improvements for the Gradio App**
Enhanced Interactivity: Add more interactive features like voice input using Whisper.
Expanded Language Support: Include more languages for translation to cater to a broader audience.
Advanced Visualizations: Integrate advanced visualizations to better explain the predictions to the patients and doctors.



