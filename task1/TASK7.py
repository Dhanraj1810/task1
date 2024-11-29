import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
file_path = r'D:\Desktop\task1\credicard1.csv'
print("Checking directory contents:")
print(os.listdir(r'D:\Desktop\task1'))
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}. Please check the file name and path.")
    raise
print("Initial Data Preview:")
print(df.head())
print("Columns in DataFrame:")
print(df.columns)
df.columns = df.columns.str.strip()
if 'Class' not in df.columns:
    raise ValueError("'Class' column not found in DataFrame.")
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
iso_forest = IsolationForest(contamination=0.001, random_state=42)
iso_forest.fit(X_train)
X_train['Anomaly'] = iso_forest.predict(X_train)
X_train = X_train[X_train['Anomaly'] == 1]
y_train = y_train.loc[X_train.index]
X_train = X_train.drop('Anomaly', axis=1)
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=300, random_state=42)
}
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    if y_pred_prob is not None:
        print(f"{model_name} AUC-ROC Score:", roc_auc_score(y_test, y_pred_prob))
def make_prediction(data, model):
    """
    Real-time prediction for new transactions.
    :param data: A single transaction as a DataFrame (same format as training data)
    :param model: Trained model for prediction
    :return: Prediction (class label)
    """
    data = data.copy()
    data['Amount'] = scaler.transform(data[['Amount']])
    prediction = model.predict(data)
    return prediction
example_data = pd.DataFrame([X_test.iloc[0]])
logistic_model = models["Logistic Regression"]
print("\nReal-time Prediction for an example transaction:")
print(make_prediction(example_data, logistic_model))
