import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names, top_n=10, title="Feature Importance"):
    """Plot top N feature importances for a given model."""
    importances = model.best_estimator_.feature_importances_
    indices = importances.argsort()[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh([feature_names[i] for i in indices][::-1], importances[indices][::-1], color='skyblue')
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()
