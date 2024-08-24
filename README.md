# Melanoma /Skin Cancer Detection Using CNN Detection Project

## **SkinLink for Melanoma Detection**

## **Executive Summary**
This project focuses on detecting melanoma, a dangerous form of skin cancer, through image analysis using a Convolutional Neural Network (CNN). The dataset sourced from Kaggle used for training consisted of 10,605 images of skin lesions, categorized as benign or malignant. The model's performance is evaluated using accuracy, precision, recall, F1 score, AUC-ROC, and a confusion matrix. The goal is to provide an automated system that assists both patients and medical professionals in the early detection of melanoma, potentially improving survival rates.

![image](https://github.com/user-attachments/assets/007ed606-2f74-40d7-8cc8-0453c918c808)

Source: [Cancer Therapy Advisor](https://www.cancertherapyadvisor.com/features/american-cancer-society-2021-statistics-show-continuous-decline/)

Melanoma instances are growing and estimated to grow in the next decade. However mortality rate linked to melanoma is trending lower. According to American Cancer about 100,000 new melanoma incidents are diagnozed. Average annual mortality is around 8,000. 

![image](https://github.com/user-attachments/assets/b4b7754b-1522-4c6e-83aa-2253360fe287)
Source: [American Cancer Society](https://www.cancer.org/content/dam/cancer-org/research/cancer-facts-and-statistics/annual-cancer-facts-and-figures/2024/2024-cancer-facts-and-figures-acs.pdf)

![image](https://github.com/user-attachments/assets/410df380-57ba-4b37-b543-27926f14b73d)
Source: [American Cancer Society](https://www.cancer.org/content/dam/cancer-org/research/cancer-facts-and-statistics/annual-cancer-facts-and-figures/2024/2024-cancer-facts-and-figures-acs.pdf)

![image](https://github.com/user-attachments/assets/d21bdca7-abc3-4bde-8261-d6ea3f17e8dd)
Source: [American Cancer Society](https://www.cancer.org/content/dam/cancer-org/research/cancer-facts-and-statistics/annual-cancer-facts-and-figures/2024/2024-cancer-facts-and-figures-acs.pdf)

![image](https://github.com/user-attachments/assets/7fa34bc5-a21a-441a-bb00-d3b96fbe8aa3)
Source: [American Cancer Society](https://www.cancer.org/content/dam/cancer-org/research/cancer-facts-and-statistics/annual-cancer-facts-and-figures/2024/2024-cancer-facts-and-figures-acs.pdf)

**Early Detection is Key:**
Skin cancer is the most common type of cancer. Early detection is crucial for effective treatment. Visual examination by a dermatologist, followed by a biopsy, is the standard procedure for diagnosis. This project aims to provide a tool that helps in the early detection of melanoma by analyzing images of skin lesions.

**Benefits to Patients & Medical Professionals**
For Patients: Faster and more accurate diagnosis can lead to earlier intervention, reducing the risk of advanced disease and improving survival rates. Patients gain quicker access to diagnostic results, which can be life-saving.

**For Medical Professionals##: Improved diagnostic tools can reduce dermatologists' workload, allow for more efficient use of healthcare resources, and help in standardizing diagnosis across different settings. This can lead to cost savings and better patient outcomes.

**Project Summary**
Dataset
The project utilizes a melanoma skin cancer dataset containing 10,605 images labeled as benign and malignant. These images were collected from various sources and are split into training, validation, and test sets.

# Gradio Apps
## 1. Patient's AI Assistant App for Early Melanoma Detection
The first Gradio application is designed for users (patients). In this application, users can upload images of their skin lesions. The app then utilizes the trained CNN model to predict whether the lesion is benign or malignant. This quick and accessible tool allows users to get an initial assessment of their skin condition, which can prompt further medical consultation if necessary.

Features:
- Image Upload: Users can easily upload images of their skin lesions.
- User can choose a language in which they wish to recive the AI predicted results
- Users can also enter their ZIP which will be used to get local medical professional information from [NPI Registry](https://npiregistry.cms.hhs.gov/api/?version=2.1&number=&enumeration_type=NPI-1&taxonomy_description=Oncology&postal_code=10001&country_code=US&limit=5)
- User uploaded image will then be normalized and batch dimenstion added
- Normalized image will be fed into CNN Model and prediction will be received
- Predicted results text will be displayed in text output box in the language specified by the user
- Prediction: The app will predict whether the lesion is benign or malignant along with degree of confidence.
- User Feedback: Users receive immediate feedback, helping them understand their skin health and whether they should seek further medical advice.

<img src="https://github.com/user-attachments/assets/f9897e23-d224-4831-b7e3-94093a7093a8" alt="Model Performance Metrics" width="600" height="800"/>

## 2. Physician's AI Assistant App for Melanoma Detection
The second Gradio application is tailored for physicians. This application enables medical professionals to review the images uploaded by their patients and provide a more detailed analysis. The doctor can speak into the app and request a patient's results. App then will fetc examine the predicted results. This system streamlines the communication between patients and doctors, facilitating quicker and more efficient follow-up.

**Features:**
- Input: Doctors can speak in English or Spanish or French and request for results of a patient.
- App will the recognize the spoken text and get appropriate language (English, Spanish or French)
- If spoken language is Spanish or French, spoken text will be translated intoo English
- From this text just the patient's name will be extracted
- Using the patient's name App will then read the patient's image from Google Cloud Storage
- Image will then be normalized and batch dimenstion added
- App will read previously saved CNN Model from Google Cloud Storage 
- Normalized image will be fed into CNN Model and prediction will be received
- Predicted results text will be displayed in text output box in the original language that the user initially spoke.
- The app will predict whether the lesion is benign or malignant along with degree of confidence.

![image](https://github.com/user-attachments/assets/18ca0e56-9e12-47ca-aba5-6cc620868862)



**Data Preprocessing**
Resizing: Images are resized to 300x300 pixels.
Normalization: Pixel values are scaled to the range [0, 1].
Data Augmentation: Techniques such as random rotation, translation, zoom, and flipping are applied to increase the diversity of the training set and improve the model’s accuracy.
Dataset was split into train, validation and test datasets. Then they were saved in pickle files to be used later for building AI model. 

**Model Building**
Preprocessed image dataset pickle files were read. A custom CNN model was built using TensorFlow/Keras. The architecture includes multiple convolutional layers followed by max pooling, flattening, and fully connected layers. Batch normalization function also was added to help the model converge quicker. Regularization techniques like Dropout and L2 regularization were applied to prevent overfitting.

**Model Training**
<u>Optimizeu<u>: Adam optimizer
<u>Loss Function<u>: Categorical cross-entropy loss
<u>Training<u>: Performed over multiple epochs with early stopping and learning rate reduction callbacks to optimize performance.

**Making & Optimizing the Model**
Title: Making & Optimizing the Model
Subtitle: How We Refined Our CNN Model to Ensure High Accuracy & Effective Generalization, Preparing It for Real-World Applications

![Alt Text](https://github.com/sebastianfjrd/Project-3_Group-9/blob/main/Making%20%26%20Optimizing%20the%20Model.png)

This image outlines the steps taken in refining the CNN model, from defining the architecture to fine-tuning the model for optimal performance. The process included the use of various callbacks, training and model evaluation techniques, and fine-tuning with TensorFlow Keras optimizers to achieve a high accuracy score of 93%. Custom CNN Model keras file was saved in Google Cloud Storage for later retrieval. 

**Evaluation**
<u>Performance Metrics<u>: The model achieved a test accuracy of 92.01%, with corresponding precision, recall, F1 score, and AUC-ROC metrics.
<u>Visualizations<u>: ROC and Precision-Recall curves were plotted to assess model performance. Confusion matrix visualizations were also used.

**Model Performance Metrics Overview**
Title: Model Performance Metrics Overview
Subtitle: Evaluating Accuracy, Precision, and Loss Across Training Epochs

![Alt Text](https://github.com/sebastianfjrd/Project-3_Group-9/blob/main/model_performance_optimization.png)

This image showcases the model's performance metrics, including the ROC curve, Precision-Recall curve, training accuracy, and training loss over multiple epochs. The high AUC-ROC and precision-recall scores indicate strong model performance, while the accuracy and loss plots highlight the model's training and validation performance over time.

![image](https://github.com/user-attachments/assets/aeb9b964-ac57-4171-b78c-fb28d0ac57c1)

![image](https://github.com/user-attachments/assets/1a998b81-7a36-4f18-841b-53453cd80db4)


## CNN Model Predictions: 
Here is a plot of model predictions vs actual predictions done on some Test Data set images -

![image](https://github.com/user-attachments/assets/cec78831-3211-4b97-a4f4-786b9f70be61)
![image](https://github.com/user-attachments/assets/a2322ea3-2ec3-410a-a54b-14e24617aca3)


**Future Research & Improvements**
<u>Advanced Architectures<u>: Consider incorporating architectures like ResNet or EfficientNet, or applying transfer learning with pretrained models.
<u>Additional Data Augmentation<u>: Experiment with additional data augmentation techniques or synthetic data generation to improve model generalization.
<u>Integration of Advanced Technologies<u>: Add automatic voice recognition system to the App so that Doctors do not have to do four steps (record, speak, stop and submit) to get results from the App. Integrate Patient AI assistant and Doctor's AI assistant into one portal. 

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



