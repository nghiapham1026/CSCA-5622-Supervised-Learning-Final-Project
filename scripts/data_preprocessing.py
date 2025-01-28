import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def handle_missing_values(data):
    """Handle missing values in the dataset."""
    # Fill numerical features with the median
    num_features = ['BMI', 'WeightInKilograms', 'SleepHours']
    for feature in num_features:
        data[feature].fillna(data[feature].median(), inplace=True)
    
    # Fill categorical features with the mode
    cat_features = ['GeneralHealth', 'HadHeartAttack']
    for feature in cat_features:
        data[feature].fillna(data[feature].mode()[0], inplace=True)
    
    # Drop columns with excessive missing values
    data.drop(columns=['TetanusLast10Tdap'], inplace=True, errors='ignore')
    return data

def scale_numerical_features(data):
    """Scale numerical features using Min-Max scaling."""
    scaler = MinMaxScaler()
    num_features = ['BMI', 'PhysicalHealthDays', 'MentalHealthDays', 'SleepHours']
    data[num_features] = scaler.fit_transform(data[num_features])
    return data

def preprocess_data(filepath):
    """Load, clean, and scale the dataset."""
    data = load_data(filepath)
    data = handle_missing_values(data)
    data = scale_numerical_features(data)
    return data

if __name__ == "__main__":
    data = preprocess_data("data/heart_disease_raw.csv")
    data.to_csv("data/heart_disease_cleaned.csv", index=False)
    print("Preprocessed data saved to 'data/heart_disease_cleaned.csv'")
