import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle

# Load the data from CSV file
data = pd.read_csv(r'D:\HI448116_Santosh_Karpe\FY25\DOCS\III\Ass\P2\pronostico_dataset.csv', delimiter=';')

# Check for missing values
print(data.isnull().sum())

# Split the data into features (X) and target (y)
X = data[['age', 'systolic_bp', 'diastolic_bp', 'cholesterol']]  # Features
y = data['prognosis']  # Target variable

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build an SVM model
model = SVC(kernel='linear', random_state=42)  # Linear kernel is often a good choice, but you can experiment with others
model.fit(X_train, y_train)

# Save the model to a .pkl file
with open(r'D:\HI448116_Santosh_Karpe\FY25\DOCS\III\Ass\ASA - SK\svm2.pkl', 'wb') as f:
    pickle.dump(model, f)

# Optionally, save the scaler as well (useful when scaling future data)
with open(r'D:\HI448116_Santosh_Karpe\FY25\DOCS\III\Ass\ASA - SK\scaler2.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
