# RANDOM FOREST (Design Concept 1)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
 
 
 
# Load the CSV file
df = pd.read_csv("emails.csv")  
 
# Extract email content and labels
emails = df["text"]
labels = df["spam"]  # 1 = malicious, 0 = safe
 
# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)
 
# Create and train the model
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100))
model.fit(X_train, y_train)
 
# Test the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
# Print classification metrics: Precision, Recall, F1 Score, and Support
print(classification_report(y_test, predictions))
 
# Function to classify new emails
def classify_email(email):
    result = model.predict([email])
    return "Malicious Email" if result[0] == 1 else "Safe Email"
 
# Test with new emails
print(classify_email("URGENT: Your account has been compromised. Click here to reset your password immediately!"))

