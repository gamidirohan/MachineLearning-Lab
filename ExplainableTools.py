import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier
import pickle
import os.path as path
from lime.lime_tabular import LimeTabularExplainer  # Import LimeTabularExplainer from LIME

# Load dataset from CSV file
csv_file = "D:\College\Sem_4\Machine Learning\Labs\Lab 7\GPT2\poems_data.csv"
data_df = pd.read_csv(csv_file)

# Extract features and labels
X = data_df.drop(columns=['label']).values  # Features
y = data_df['label'].values  # Labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply MinMax scaling to input data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model file path
model_file = "D:\College\Sem_4\Machine Learning\Labs\Lab 7\catboost_model.pkl"

# Check if model file exists
if path.exists(model_file):
    # Load trained model from file
    with open(model_file, 'rb') as f:
        catboost_clf = pickle.load(f)
else:
    # Initialize CatBoost classifier
    catboost_clf = CatBoostClassifier(logging_level='Silent')

    # Train CatBoost classifier
    catboost_clf.fit(X_train_scaled, y_train)

    # Save trained model as a .pkl file
    with open(model_file, 'wb') as f:
        pickle.dump(catboost_clf, f)

# Explain feature importances using ELI5 (removed)

# Initialize LimeTabularExplainer
explainer = LimeTabularExplainer(X_train_scaled, feature_names=data_df.columns.drop('label'), class_names=['0', '1'], discretize_continuous=True)

# Select a sample from the test set for explanation
sample_idx = 0  # You can change this to any index you want to explain
explanation = explainer.explain_instance(X_test_scaled[sample_idx], catboost_clf.predict_proba, num_features=len(data_df.columns.drop('label')))

# Print explanation
print("\nPrediction Explanation:")
print("===========================================================")
print(explanation.as_list())