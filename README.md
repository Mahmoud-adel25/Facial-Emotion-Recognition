

Project Title: Emotion Recognition from Facial Expressions


Overview:

This project performs facial emotion recognition using images from the FER-2013 dataset. It includes preprocessing steps, data loading, and training of a model using TensorFlow/Keras. It also provides tools for testing predictions on custom images using upload interfaces in Google Colab.


 Key Features:

 Automatically downloads the FER-2013 dataset from Kaggle.
 Performs custom preprocessing, including Histogram Equalization, Normalization, Sharpening, and Non-Local Means Denoising.
 Organizes data into training and test sets.
 Prepares emotion labels such as angry, happy, sad, surprise, etc.
 Includes model training and evaluation (details visible in later notebook cells).
 Enables real-time image testing through upload widgets.
 Supports YOLO-based face detection for detecting and classifying multiple faces in one image.
 **Supports deployment with a web-based Gradio interface for accessible emotion detection on uploaded images.



 Requirements:

 Python 3.x
 Jupyter Notebook (Google Colab recommended)
 TensorFlow
 NumPy, matplotlib, scikit-learn, OpenCV
 ipywidgets
Kaggle API key (`kaggle.json` file)


 Setup Instructions:

1. Upload your `kaggle.json` to the notebook directory.
2. Ensure required libraries are installed:

   pip install kaggle tensorflow matplotlib scikit-learn opencv-python ipywidgets
3. Run the notebook cells in order to download data, preprocess images, and train the model.



 Testing Options:
 1. Image Upload Widget:

 Upload an image through an interactive GUI in the notebook.
The image is automatically preprocessed and passed to the trained model.
 The predicted emotion label is printed directly below the image.

#### 2. YOLO-Based Multi-Face Detection and Prediction:

 Upload a photo containing one or more faces.
 The YOLO model detects face regions.
 Each face is cropped, preprocessed, and passed through the emotion recognition model.
 The final image is displayed with bounding boxes and emotion labels over each face.

 3. Gradio Web Interface (NEW):

 A user-friendly web interface is provided using Gradio.
 Users can upload images directly in the browser without needing to run individual cells.
 The backend integrates YOLO for face detection and uses the trained emotion recognition model.
 Detected faces are annotated with bounding boxes and predicted emotion labels, and the result is displayed instantly.
 Gradio also supports optional public sharing via a temporary URL, making it easy to demo the model.


Dataset:

FER-2013 from Kaggle: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

Author:

Mahmoud Adel & Youssef Omran
