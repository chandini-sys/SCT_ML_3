# 🐱🐶 Cat vs Dog Image Classifier using SVM + HOG

This project is a machine learning pipeline that classifies images of cats and dogs using **Support Vector Machines (SVM)** and **Histogram of Oriented Gradients (HOG)** features.

---

## 📌 Project Overview

* 🔍 **Objective:** Classify whether an input image is a **cat** or a **dog**.
* 📷 **Technique:** Feature extraction using **HOG**, classification using **SVM (Linear Kernel)**.
* 💾 **Dataset:** [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
* 🧠 **Model:** Trained with 1000 cat and 1000 dog images (grayscale, 128x128 size).

---

## 🛠️ Technologies Used

* Python 3
* OpenCV
* scikit-learn
* scikit-image (for HOG)
* joblib
* NumPy

---

## 📁 Project Structure

```
📦 svm-cat-dog-classifier
├── svm dog vs cat.py          # Training script
├── test images.py             # Prediction and display script
├── svm_cat_dog_model_hog.pkl  # Saved trained model
├── README.md                  # Project documentation
└── PetImages/                 # Cat & Dog image folders
```

---

## 🚀 How to Run

### ✅ Step 1: Install Requirements

```bash
pip install numpy opencv-python scikit-learn scikit-image joblib
```

### ✅ Step 2: Prepare Dataset

Download and extract the dataset from Kaggle. Place images in the following structure:

```
PetImages/
├── Cat/
│   ├── 1.jpg
│   └── ...
└── Dog/
    ├── 1.jpg
    └── ...
```

> **Note**: The script currently uses 1000 images from each class.

### ✅ Step 3: Train the Model

Run the following script to extract HOG features and train an SVM:

```bash
python "svm dog vs cat.py"
```

This will:

* Train an SVM model
* Print accuracy and classification report
* Save the model as `svm_cat_dog_model_hog.pkl`

### ✅ Step 4: Test the Model

Update the image path in `test images.py` and run:

```bash
python "test images.py"
```

This will:

* Load the model
* Predict the label (Cat or Dog) for the given image
* Display the image with label overlay

---

## 📊 Sample Output

```
✅ Accuracy: 0.8450
```

And on prediction:

```
Predicted: Dog 🐶
```

![Prediction Screenshot](./sample_prediction.png)

---

## 🔮 Future Enhancements

* Try with CNN or transfer learning models (like VGG16)
* Automate data preprocessing for all formats
* Add support for batch prediction
* Create a simple web or GUI interface

---

## 📄 License

This project is licensed under the MIT License.
