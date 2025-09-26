# Logistic Regression Classifier — Breast Cancer Dataset

## 📌 Project Overview
This project implements a **binary classification** model using **Logistic Regression** on the popular **Breast Cancer Wisconsin dataset** (from scikit-learn).  
The task was given as part of an **AI & ML Internship** and demonstrates practical usage of logistic regression for classification, model evaluation, and threshold tuning.

---

## 📊 Workflow
1. **Dataset Loading**
   - Used `sklearn.datasets.load_breast_cancer` (no manual downloads required).
   - Inspected dataset shape, features, and target balance.

2. **Exploratory Data Analysis (EDA)**
   - Checked class distribution (benign vs malignant).  
   - Plotted feature distributions.

3. **Preprocessing**
   - Train/test split with stratification to maintain class balance.  
   - Standardized features using `StandardScaler`.

4. **Model Training**
   - Logistic Regression (`sklearn.linear_model.LogisticRegression`).  
   - Solver: `liblinear`, with increased `max_iter` for convergence.

5. **Evaluation Metrics**
   - **Confusion Matrix**  
   - **Accuracy, Precision, Recall, F1-score**  
   - **ROC Curve & ROC-AUC**  
   - **Precision-Recall Curve & Average Precision Score**

6. **Threshold Tuning**
   - Compared precision, recall, and F1 at different thresholds.  
   - Showed how changing threshold affects trade-offs.

7. **Sigmoid Function**
   - Plotted sigmoid curve to explain how logistic regression converts logits → probabilities.  

8. **Hyperparameter Tuning**
   - Used `GridSearchCV` to optimize regularization strength `C`.  
   - Evaluated best model with cross-validation (ROC-AUC).

9. **Model Saving**
   - Saved trained model + scaler with `joblib` → `logistic_bc_model.joblib`.

---

## 📈 Results
- **Default Logistic Regression Model**
  - High accuracy (~95%+) on test set.  
  - ROC-AUC consistently > 0.98.  
- Precision/Recall trade-off explored with custom thresholds.

---

## 🛠️ Technologies Used
- **Python 3**
- **Google Colab**
- **scikit-learn**
- **pandas**
- **matplotlib**
- **joblib**

---

## 📂 Repository Structure
```
├── logistic_regression_task4.ipynb   # Google Colab notebook
├── README.md                         # Project documentation (this file)
├── logistic_bc_model.joblib          # Saved trained model (optional)
```

---

## 🚀 How to Run
1. Open [Google Colab](https://colab.research.google.com/).  
2. Upload `logistic_regression_task4.ipynb`.  
3. Run all cells step by step.  
4. (Optional) Save model using the provided code cell.

---

## 🎯 Learning Outcomes
- Difference between **Logistic vs Linear Regression**.  
- Role of the **Sigmoid Function** in probability estimation.  
- Understanding of **Precision vs Recall**.  
- Importance of **ROC-AUC** and **Confusion Matrix**.  
- Handling **imbalanced datasets**.  
- Adjusting the **classification threshold**.  
- Logistic Regression for **multi-class problems**.

---

## ✍️ Author
- **Your Name** (AI & ML Internship Participant)  
- Internship Task 4 — Logistic Regression Classifier  
