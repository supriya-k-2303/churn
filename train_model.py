import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Loads training features into a DataFrame
X_train=pd.read_csv("data/X_train.csv")
X_test=pd.read_csv("data/X_test.csv")
y_train=pd.read_csv("data/y_train.csv").values.ravel()
y_test=pd.read_csv("data/y_test.csv").values.ravel()


model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

# make prediction
y_pred=model.predict(X_test)

# evaluate accurace
print("accuracy:",accuracy_score(y_test,y_pred))
print("\n Classification report:", classification_report(y_test,y_pred))

joblib.dump(model,"model/churn_model.pkl")
joblib.dump(X_train.columns.tolist(),"model/features.pkl")
print("model saved successfully")