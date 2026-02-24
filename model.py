
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = {
    "attendance": [90, 85, 70, 60, 50, 95, 40, 75],
    "score": [85, 78, 60, 50, 45, 90, 35, 65],
    "result": [1, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["attendance", "score"]]
y = df["result"]

model = LogisticRegression()
model.fit(X, y)

with open("student_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained successfully")