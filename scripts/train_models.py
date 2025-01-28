import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def load_and_split_data(filepath):
    """Load the dataset and split it into training and testing sets."""
    data = pd.read_csv(filepath)
    X = data.drop(columns=['HadHeartAttack'])
    y = data['HadHeartAttack']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with grid search."""
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(estimator=logreg, param_grid=param_grid, scoring='roc_auc', cv=5)
    grid.fit(X_train, y_train)
    return grid

def train_random_forest(X_train, y_train):
    """Train Random Forest with grid search."""
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='roc_auc', cv=5)
    grid.fit(X_train, y_train)
    return grid

def train_xgboost(X_train, y_train):
    """Train XGBoost with grid search."""
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
    grid = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='roc_auc', cv=5)
    grid.fit(X_train, y_train)
    return grid

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data("data/heart_disease_engineered.csv")
    
    # Train models
    logreg_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # Evaluate models
    for model_name, model in zip(['Logistic Regression', 'Random Forest', 'XGBoost'], [logreg_model, rf_model, xgb_model]):
        print(f"\n{model_name} Results:")
        y_pred = model.best_estimator_.predict(X_test)
        y_pred_proba = model.best_estimator_.predict_proba(X_test)[:, 1]
        print(classification_report(y_test, y_pred))
        print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba)}")
