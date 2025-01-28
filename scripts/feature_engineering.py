import pandas as pd

def create_bmi_category(data):
    """Create BMI category feature."""
    def bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 25:
            return 'Normal'
        elif 25 <= bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    data['BMICategory'] = data['BMI'].apply(bmi_category)
    return data

def create_health_risk_index(data):
    """Create a composite health risk index."""
    data['HealthRiskIndex'] = data['PhysicalHealthDays'] + data['MentalHealthDays']
    return data

def create_sleep_quality(data):
    """Create sleep quality feature."""
    def sleep_quality(hours):
        if 7 <= hours <= 9:
            return 'Adequate'
        else:
            return 'Inadequate'
    
    data['SleepQuality'] = data['SleepHours'].apply(sleep_quality)
    return data

def engineer_features(filepath):
    """Perform feature engineering on the dataset."""
    data = pd.read_csv(filepath)
    data = create_bmi_category(data)
    data = create_health_risk_index(data)
    data = create_sleep_quality(data)
    data.to_csv("data/heart_disease_engineered.csv", index=False)
    print("Feature-engineered data saved to 'data/heart_disease_engineered.csv'")

if __name__ == "__main__":
    engineer_features("data/heart_disease_cleaned.csv")
