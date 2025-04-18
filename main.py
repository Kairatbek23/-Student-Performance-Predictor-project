import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load dataset
    data = pd.read_csv("student_data.csv")

    # Preprocess
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop("final_grade", axis=1)
    y = data["final_grade"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)
    print("R^2 Score:", r2_score(y_test, predictions))

    # Feature importance plot
    importances = model.feature_importances_
    features = X.columns
    sns.barplot(x=importances, y=features)
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

if __name__ == "__main__":
    main()
