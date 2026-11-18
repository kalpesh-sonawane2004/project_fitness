import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
file_path = "bmi.csv"
df = pd.read_csv(file_path)

# Label encoding for BMI Class (Normal weight -> Fit, others -> Not Fit)
df['Fit_Status'] = df['BmiClass'].apply(lambda x: 'Not Fit' if x == 'Normal Weight' else 'Fit')
label_encoder = LabelEncoder()
df['Fit_Status'] = label_encoder.fit_transform(df['Fit_Status'])  # Encode Fit as 1, Not Fit as 0

# Selecting features and target
X = df[['Age', 'Height', 'Weight']]
y = df['Fit_Status']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(clf, "bmi_classifier.pkl")
print("Model saved as bmi_classifier.pkl")

# Function for user input classification
def predict_fitness(age, height, weight):
    prediction = clf.predict([[age, height, weight]])[0]
    return "Fit" if prediction == 1 else "Not Fit"

# Example usage
age = float(input("Enter Age: "))
height = float(input("Enter Height in meters: "))
weight = float(input("Enter Weight in kg: "))
print("Fitness Status:", predict_fitness(age, height, weight))
