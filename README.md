# 📊 Customer Churn Prediction Dashboard

A **Machine Learning web application** that predicts whether a customer is likely to **churn (leave a service)** based on customer subscription and usage details. The model is trained using the **Decision Tree algorithm** and deployed through an interactive **Streamlit dashboard** for real-time predictions.

---

## 🚀 Features

* Predicts **customer churn probability** based on input features.
* Interactive **Streamlit dashboard** for real-time predictions.
* Displays **churn probability and prediction results instantly**.
* Visualizes **feature importance** to explain model decisions.
* Simple and user-friendly interface.

---

## 🧠 Machine Learning Model

The project uses a **Decision Tree Classifier** to analyze customer attributes and predict churn likelihood.

### Workflow

1. Data preprocessing and feature encoding
2. Train-test split of dataset
3. Model training using **Decision Tree algorithm**
4. Feature scaling using **StandardScaler**
5. Model serialization using **Pickle**
6. Real-time prediction via **Streamlit interface**

---

## 🛠 Tech Stack

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming Language      |
| Pandas       | Data preprocessing        |
| NumPy        | Numerical operations      |
| Scikit-learn | Machine Learning model    |
| Streamlit    | Web application framework |
| Matplotlib   | Data visualization        |

---

## 📂 Project Structure

```
Customer_Churn_Prediction
│
├── app.py
├── decision_tree_model.pkl
├── scaler.pkl
├── model_columns.pkl
├── customer_churn_dataset.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/rinkita23/Customer_Churn_Prediction.git
```

Navigate to the project folder:

```bash
cd Customer_Churn_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

* Deploy the application online.
* Improve model performance with advanced algorithms.
* Add more customer behavior features for better prediction accuracy.
* Build a more advanced analytics dashboard.

---

## 👩‍💻 Author

**Rinkita Ramrakhiyani**
