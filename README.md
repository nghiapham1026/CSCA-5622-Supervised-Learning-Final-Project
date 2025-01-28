# **Heart Disease Prediction using Machine Learning**

## **Project Overview**
Heart disease is one of the leading causes of death globally, making early detection and intervention essential. This project applies machine learning techniques to predict the likelihood of heart disease based on clinical, behavioral, and demographic data. By leveraging feature engineering, exploratory data analysis (EDA), and advanced modeling techniques, the project aims to provide actionable insights and accurate predictions to aid in early diagnosis and prevention.

---

## **Key Features**
- Comprehensive **data preprocessing** to handle missing values and scale numerical features.
- Advanced **feature engineering** (e.g., BMI categories, health risk index, sleep quality).
- Exploratory **visualizations** for insights into data distributions and feature relationships.
- Comparison of multiple machine learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model performance evaluation using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- Interpretability through **feature importance analysis** and visualizations.

---

## **Dataset**
- **Source**: The dataset was sourced from publicly available heart disease datasets.  
  *(Include a link to the dataset if publicly available.)*
- **Description**: Contains clinical, demographic, and behavioral features, including:
  - `BMI`: Body Mass Index.
  - `PhysicalHealthDays`: Number of poor physical health days in the past month.
  - `MentalHealthDays`: Number of poor mental health days in the past month.
  - `SleepHours`: Average hours of sleep per night.
  - Target variable: `HadHeartAttack` (Yes/No).

---

## **Getting Started**

### **Prerequisites**
- Install Python 3.8 or higher.
- Clone the repository and navigate to the project directory:
  ```bash
  git clone https://github.com/yourusername/heart-disease-prediction.git
  cd heart-disease-prediction
  ```

### **Install Dependencies**
Install the required libraries using:
```bash
pip install -r requirements.txt
```

### **Run the Project**
1. **Preprocess the Data**:
   Run the data preprocessing script:
   ```bash
   python scripts/data_preprocessing.py
   ```

2. **Perform Feature Engineering**:
   ```bash
   python scripts/feature_engineering.py
   ```

3. **Train Models**:
   ```bash
   python scripts/train_models.py
   ```

4. **Generate Visualizations**:
   Use the visualization script or notebooks to explore results:
   ```bash
   python scripts/visualize_results.py
   ```

---

## **Key Results**
- **Model Performance**:
  - Logistic Regression: ROC-AUC = 0.8852
  - Random Forest: ROC-AUC = 0.8800
  - XGBoost: ROC-AUC = 0.8866
- **Top Predictive Features**:
  - `HealthRiskIndex`
  - `BMI`
  - `PhysicalActivities`
  - `GeneralHealth`

---

## **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature/your-feature`).
3. Commit your changes.
4. Push to the branch and submit a pull request.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.