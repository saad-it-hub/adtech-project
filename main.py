# 1. Import libraries

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
import joblib


# 2. Load both CSV files

train_data = pd.read_csv("messy_adtech_data_20k.csv")
test_data = pd.read_csv("messy_adtech_test_data_10k.csv")

print("Training Dataset:", train_data.shape)
print("Testing Dataset:", test_data.shape)


# 3. Display dataset information

print("\nTraining Dataset:")
print(train_data.head())

print("\nDataset Information:")
print(train_data.info())

print("\nMissing Values:")
print(train_data.isnull().sum())


# 4. Select input features

features = [
    "user_age",
    "time_on_page_sec",
    "historical_ctr",
    "bid_amount",
    "page_load_time_ms",
    "user_past_purchases",
    "scroll_depth_pct",
    "has_adblocker"
]


# 5. Clean input data

def clean_data(data):

    data = data.copy()

    for col in features:

        values = data[col].astype(str).str.strip()

        percentage = values.str.contains("%", regex=False)

        values = values.str.replace("$", "", regex=False)
        values = values.str.replace("%", "", regex=False)
        values = values.str.replace(",", "", regex=False)

        data[col] = pd.to_numeric(
            values,
            errors="coerce"
        )

        if col == "historical_ctr":
            data.loc[percentage, col] /= 100

    # Handle invalid values

    data.loc[
        (data["user_age"] < 13) |
        (data["user_age"] > 100),
        "user_age"
    ] = np.nan

    data.loc[
        data["page_load_time_ms"] <= 0,
        "page_load_time_ms"
    ] = np.nan

    return data[features]


# 6. Clean target column

def clean_target(data):

    return (
        data["clicked"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({
            "0": 0,
            "1": 1,
            "no": 0,
            "yes": 1
        })
    )


# 7. Prepare training data

X_train = clean_data(train_data)
y_train = clean_target(train_data)

# Remove rows with missing target values

valid_train = y_train.notna()

X_train = X_train[valid_train]
y_train = y_train[valid_train].astype(int)


# 8. Prepare testing data

X_test = clean_data(test_data)
y_test = clean_target(test_data)


# 9. Create Logistic Regression model

model = make_pipeline(

    SimpleImputer(strategy="median"),

    StandardScaler(),

    LogisticRegression(max_iter=1000)

)


# 10. Train the model

print("\nTraining Started...")

model.fit(X_train, y_train)

print("Training Completed Successfully!")


# 11. Test the model

print("\nTesting Started...")

predictions = model.predict(X_test)

print("Testing Completed Successfully!")


# 12. Check model accuracy

valid_test = y_test.notna()

actual = y_test[valid_test].astype(int)
predicted = predictions[valid_test]

print("\n========== MODEL EVALUATION ==========")

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Testing Records with Actual Labels:", len(actual))

if len(actual) > 0:

    print("\nAccuracy:")
    print(accuracy_score(actual, predicted))

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predicted))

    print("\nClassification Report:")
    print(classification_report(actual, predicted))

else:
    print("No actual test labels available.")


# 13. Save trained model

joblib.dump(model, "logistic_model.pkl")

print("\nTrained Model Saved Successfully!")


# 14. Save testing predictions

results = pd.DataFrame({

    "click_id": test_data["click_id"],

    "actual_clicked": test_data["clicked"],

    "predicted_clicked": predictions

})

results.to_csv(
    "testing_results.csv",
    index=False
)

print("\nTesting Results Saved Successfully!")

print("\nFirst 10 Predictions:")
print(results.head(10))

print("\n========== PROJECT COMPLETED ==========")

