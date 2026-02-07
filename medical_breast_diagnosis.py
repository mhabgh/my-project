# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# load Dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
print('Number of samples and features:' ,X.shape)
print("Target labels', np.unique(y)")

# preprocessing (scaling)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# spit data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)

# feature scaling (Normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training set shape", X_train_scaled.shape)
print("Testing set shape", X_test_scaled.shape)

print("Ttrain features mean (approx):", X_train_scaled.mean(axis=0)[:5])
print("Train features std (approx):", X_test_scaled.std(axis=0)[:5])

# Logistic Regression Training
from sklearn.linear_model import LinearRegression
Logreg = LogisticRegression(max_iter=1000, random_state=42)
Logreg.fit(X_train_scaled, y_train)
y_pred_Logreg = Logreg.predict(X_test_scaled)
print("first 10 prediction:", y_pred_Logreg[:10])
print("first 10 true lables:", y_test[:10])

# Logestic Regression Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred_Logreg)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred_Logreg)
print("Confusion Matrix:\n", cm)
report = classification_report(y_test, y_pred_Logreg)
print("Classification Report:\n", report)

# Random Forest Training
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
print("First 10 prediction (Random Forest):", y_pred_rf[:10])
print("First 10 true Lables:", y_test[:10])

# Random Forest Evaluation
accuracy_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

# Put Results
results = {"\logistic comprison": accuracy, "Random Forest": accuracy_rf}
print("\nModel Comparisson:")
for model, acc in results.items(): print(f"(model): {acc:4f}")

# F1-Score for Random Forest
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred_rf)
print("Random Forest F1-Score:", f1)

from sklearn.metrics import classification_report
print ("Random Forest Classification Report:\n")
print(classification_report(y_test, y_pred_rf))

# Feature Importance (Random Forest)
import pandas as pd
importances = rf_model.feature_importances_
feature_names = dataset.feature_names
feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)
print(feat_imp_df.head(10))

# Plot Feature Importance
import matplotlib.pyplot as plt
top_features = feat_imp_df.head(10)
plt.figure()
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feattures Importance (Random Forest)")
plt.gca().invert_yaxis()
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _= roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# save the trained  Random Forest model
import joblib
joblib.dump(rf_model, "rf_breast_cancer_model.pk1")
print("model saved successfully")










