<img src="https://capsule-render.vercel.app/api?type=waving&color=0:36BCF7,50:7F00FF,100:FF4B4B&height=250&section=header&text=Customer%20Churn%20Prediction&fontSize=55&fontAlignY=35&fontColor=ffffff&animation=fadeIn&desc=Powered%20by%20Deep%20Learning%20%7C%20TensorFlow%20%7C%20Keras&descAlignY=55&descSize=18" width="100%"/>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-ANN-red?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Preprocessing-yellow?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

<p align="center">
  An end-to-end Deep Learning project that predicts whether a bank customer is likely to leave the organization using an Artificial Neural Network (ANN).
</p>

---

## 📌 Project Overview

Customer churn is one of the major challenges faced by banks and financial institutions. Retaining existing customers is often more cost-effective than acquiring new ones.

This project implements a **Customer Churn Prediction System** using an Artificial Neural Network (ANN) built with TensorFlow and Keras.

The model analyzes customer demographics, financial information, and banking activity to estimate the probability of a customer leaving the bank.

The project covers the complete Deep Learning workflow, starting from data preprocessing and feature engineering to ANN training, evaluation, model persistence, and deployment through an interactive Streamlit application.

## 🎯 Project Objectives

- Analyze customer data to identify patterns associated with customer churn.
- Perform categorical encoding and numerical feature scaling.
- Develop an Artificial Neural Network for binary classification.
- Optimize model training using Early Stopping.
- Monitor training performance using TensorBoard.
- Evaluate model performance using a confusion matrix and classification report.
- Save trained models and preprocessing objects for future inference.
- Build an interactive application for real-time customer churn prediction.

## Live URL

Check out :- https://churn-prediction-dl-ann.streamlit.app/

<br>

## 📂 Dataset Description

The project uses the **Churn Modelling Dataset**, which contains customer-related information from a banking institution.

### Features

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Customer's country |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Number of years with the bank |
| Balance | Customer's account balance |
| NumOfProducts | Number of banking products owned |
| HasCrCard | Whether the customer owns a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Estimated annual salary |
| Exited | Target variable indicating customer churn |

**Target Variable: `Exited`**

- `0` – Customer stays with the bank.
- `1` – Customer leaves the bank.

The columns `RowNumber`, `CustomerId`, and `Surname` are removed because they are not required as predictive features.

## ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas & NumPy | Data manipulation and numerical operations |
| Scikit-learn | Data preprocessing and evaluation |
| TensorFlow & Keras | ANN development and training |
| Matplotlib | Data visualization |
| TensorBoard | Training monitoring and visualization |
| Streamlit | Interactive web application |
| Pickle | Saving preprocessing objects |

## 🔄 Project Workflow

The complete pipeline follows these stages:

<p align="center">
  <img src="https://img.shields.io/badge/01-Data%20Collection-2563EB?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/02-Data%20Cleaning-2563EB?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/03-Feature%20Selection-2563EB?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/04-Categorical%20Encoding-2563EB?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/05-Train%20Test%20Split-2563EB?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/06-Feature%20Scaling-2563EB?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/07-ANN%20Development-FF8C00?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/08-Model%20Training-FF8C00?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/09-Model%20Evaluation-16A34A?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/10-Model%20Persistence-16A34A?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/11-Streamlit%20Deployment-FF4B4B?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/12-Customer%20Churn%20Prediction-16A34A?style=for-the-badge" />
</p>

## 🧹 Data Preprocessing

The following preprocessing operations are performed before training the neural network.

### 1. Data Cleaning and Feature Selection

- Removed unnecessary columns such as `RowNumber`, `CustomerId`, and `Surname`.
- Separated independent features and the target variable `Exited`.

### 2. Label Encoding

The `Gender` column is converted into numerical values using Scikit-learn's `LabelEncoder`.

### 3. One-Hot Encoding

The `Geography` column is transformed using `OneHotEncoder` to represent categorical geographical information without introducing an artificial ordinal relationship.

The resulting features are:

- Geography_France
- Geography_Germany
- Geography_Spain

### 4. Train-Test Split

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

The split uses `random_state=42` to ensure reproducibility.

### 5. Feature Scaling

`StandardScaler` is used to standardize numerical features.

The scaler is fitted on the training data and subsequently applied to the testing data to maintain consistency and avoid data leakage during scaling.

All preprocessing objects are saved using Pickle for reuse during inference.

## 🧠 Artificial Neural Network Architecture

A feedforward Artificial Neural Network is implemented using TensorFlow and Keras.

The model consists of two hidden layers and one output layer.

### Network Architecture

<p align="center">
  <img src="https://img.shields.io/badge/INPUT%20LAYER-12%20FEATURES-2563EB?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/HIDDEN%20LAYER%201-64%20NEURONS%20%7C%20RELU-9333EA?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/HIDDEN%20LAYER%202-32%20NEURONS%20%7C%20RELU-9333EA?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/OUTPUT%20LAYER-1%20NEURON%20%7C%20SIGMOID-16A34A?style=for-the-badge" />
  <br>⬇️
  <br>
  <img src="https://img.shields.io/badge/PREDICTION-CHURN%20PROBABILITY-FF8C00?style=for-the-badge" />
</p>

### Model Configuration

| Layer | Neurons | Activation |
|---|---:|---|
| Input Layer | 12 | — |
| Hidden Layer 1 | 64 | ReLU |
| Hidden Layer 2 | 32 | ReLU |
| Output Layer | 1 | Sigmoid |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Crossentropy |
| Evaluation Metric | Accuracy |
| Maximum Epochs | 100 |
| Early Stopping Patience | 10 |
| Batch Size | Keras default (32) |
| Output Threshold | 0.5 |

**Early Stopping:** Monitors validation loss and stops training when no improvement is observed for 10 consecutive epochs. The best model weights are restored.

**TensorBoard:** Used to monitor training and validation metrics across epochs.

## 📊 Model Evaluation

The trained model is evaluated on the test dataset using:

- Confusion Matrix
- Precision
- Recall
- F1-Score
- Accuracy

The predicted probabilities are converted into binary classes using a threshold of 0.5.

```python
y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob >= 0.5).astype(int)
```

The confusion matrix and classification report are generated using Scikit-learn.

> Note: Evaluation metrics are intentionally not hardcoded. They should be updated using the actual results obtained from model evaluation.

## 💾 Model Persistence

To ensure that the trained model can be reused without retraining, the following artifacts are saved:

| File | Purpose |
|---|---|
| `model.h5` | Trained ANN model |
| `label_encoder.pkl` | Saved Gender LabelEncoder |
| `onehot_encoder.pkl` | Saved Geography OneHotEncoder |
| `standard_scaler.pkl` | Saved StandardScaler |

These artifacts are loaded by the prediction application to ensure that incoming customer data undergoes the same preprocessing operations used during model training.

## 🚀 Streamlit Web Application

An interactive web application is developed using Streamlit to make customer churn prediction accessible through a user-friendly interface.

Users can provide customer information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Account Balance
- Number of Products
- Credit Card Ownership
- Active Membership Status
- Estimated Salary

The application processes the inputs, applies the saved encoders and scaler, and passes the transformed features to the trained ANN.

The application displays:

- Predicted customer churn status.
- Estimated churn probability.

### Prediction Logic

The sigmoid output of the ANN represents the estimated probability of customer churn.

A threshold of 0.5 is used for binary classification:

- Probability ≥ 0.5 → Customer likely to churn.
- Probability < 0.5 → Customer unlikely to churn.

## 📁 Project Structure

```text
Deep Learning End to End Project/
│
├── app.py
│
├── Churn_Modelling.csv
│
├── model.h5
│
├── label_encoder.pkl
├── onehot_encoder.pkl
├── standard_scaler.pkl
│
├── logs/
│   └── fit/
│       └── TensorBoard Training Logs
│
├── requirements.txt
│
└── README.md
```

## 🛠️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd "Deep Learning End to End Project"
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install pandas numpy scikit-learn tensorflow streamlit matplotlib tensorboard h5py
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

### 6. Access the Application

Open the following URL in your browser:

```text
http://localhost:8501
```

## 📈 TensorBoard Visualization

TensorBoard is integrated into the training workflow to visualize model performance.

To launch TensorBoard independently:

```bash
python -m tensorboard.main --logdir logs/fit --port 6006
```

Open:

```text
http://localhost:6006
```

This allows visualization of training and validation metrics over the course of model training.

## 🔮 Future Improvements

- Hyperparameter tuning to optimize ANN architecture and performance.
- Experiment with dropout and batch normalization to improve generalization.
- Address class imbalance using class weights or resampling techniques.
- Compare ANN performance with traditional ML algorithms such as Random Forest, XGBoost, and CatBoost.
- Experiment with different classification thresholds to improve churn detection recall.
- Implement SHAP-based explainability to understand feature contributions.
- Deploy the application using Streamlit Community Cloud or another hosting platform.

<p align="center">
  <a href="https://github.com/Abhinav-frfr">
    <img src="https://img.shields.io/badge/GitHub-Abhinav--frfr-181717?style=for-the-badge&logo=github" />
  </a>
</p>

---

<p align="center">
  ⭐ If you find this project interesting, consider giving the repository a star!
</p>

<p align="center">
  <i>Building intelligent systems, one model at a time.</i>
</p>
