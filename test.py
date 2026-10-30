import pandas as pd
import joblib 
from sklearn.metrics import accuracy_score  
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model.pkl")

test_df = pd.read_csv(BASE_DIR / "Iris.csv")
test_df = test_df.drop("Id", axis=1)
X = test_df.drop("Species", axis=1)
y = test_df["Species"]
pred = model.predict(X)
accuracy = accuracy_score(pred,y)
print("Accuracy:", accuracy)
#assertions are some conditions
# part of software testing to verify that the code is working as expected
assert accuracy > 0.91
assert len(pred) == len(X)
print("test cases passed")