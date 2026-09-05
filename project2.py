# ==========================================================
# PROJECT 2: DATA CLASSIFICATION USING AI
# Iris Flower Classification using Decision Tree
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ----------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------

iris = load_iris()

data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

data["target"] = iris.target

# Add flower names
data["flower_name"] = data["target"].map(
    {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }
)


# ----------------------------------------------------------
# 2. UNDERSTAND THE DATASET
# ----------------------------------------------------------

print("=" * 50)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 50)

print("\nDataset shape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nFlower distribution:")
print(data["flower_name"].value_counts())


# ----------------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# ----------------------------------------------------------

X = data[iris.feature_names]
y = data["target"]


# ----------------------------------------------------------
# 4. SPLIT DATA INTO TRAINING AND TESTING
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ----------------------------------------------------------
# 5. CREATE CLASSIFICATION MODEL
# ----------------------------------------------------------

model = DecisionTreeClassifier(
    random_state=42
)


# ----------------------------------------------------------
# 6. TRAIN THE MODEL
# ----------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed successfully.")


# ----------------------------------------------------------
# 7. MAKE PREDICTIONS
# ----------------------------------------------------------

y_pred = model.predict(X_test)


# ----------------------------------------------------------
# 8. CALCULATE ACCURACY
# ----------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)


# ----------------------------------------------------------
# 9. CONFUSION MATRIX
# ----------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# ----------------------------------------------------------
# 10. DECISION TREE VISUALIZATION
# ----------------------------------------------------------

plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree Classification Model")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=300)
plt.show()


# ----------------------------------------------------------
# 11. FEATURE GRAPH
# ----------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    data["petal length (cm)"],
    data["petal width (cm)"],
    c=data["target"]
)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Flower Classification")
plt.tight_layout()

plt.savefig("iris_features.png", dpi=300)
plt.show()


# ----------------------------------------------------------
# 12. PREDICT A NEW FLOWER
# ----------------------------------------------------------

new_flower = [[
    5.1,   # Sepal length
    3.5,   # Sepal width
    1.4,   # Petal length
    0.2    # Petal width
]]

prediction = model.predict(new_flower)

predicted_flower = iris.target_names[prediction[0]]

print("\nNew Flower Prediction:")
print("Predicted Flower:", predicted_flower)


# ----------------------------------------------------------
# 13. FINAL RESULT
# ----------------------------------------------------------

print("\n" + "=" * 50)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 50)