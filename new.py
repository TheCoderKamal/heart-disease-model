from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.params import Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import joblib

app = FastAPI()

# Mount static files and set templates directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variable to store the best model and feature columns
best_model = None
feature_columns = []

# Define the order of models for tie-breaking
MODEL_ORDER = {
    "Logistic Regression": 7,
    "KNN": 6,
    "Naive Bayes": 5,
    "SVM": 4,
    "Decision Tree": 3,
    "Random Forest": 2,
    "Gradient Boosting": 1,
}

@app.get("/", response_class=HTMLResponse)
async def upload_file(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/train")
async def train_model(
    request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    hyperparameter_tuning: str = Form(...)
):
    global best_model, feature_columns

    # Strip whitespace and newlines from the target_column
    target_column = target_column.strip()

    # Save uploaded file
    file_location = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)  # Create data directory if it doesn't exist
    with open(file_location, "wb+") as f:
        f.write(file.file.read())

    # Read dataset
    df = pd.read_csv(file_location)
    os.remove(file_location)  # Delete file after reading
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)  # Impute numeric columns
    df.fillna(df.mode().iloc[0], inplace=True)  # Impute categorical columns


    # Check if the target column exists in the dataset
    if target_column not in df.columns:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": f"Target column '{target_column}' not found in the dataset."},
        )

    # Label encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    label_encoders = {col: LabelEncoder() for col in categorical_cols}
    for col in categorical_cols:
        df[col] = label_encoders[col].fit_transform(df[col])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Store feature columns for prediction form
    feature_columns = X.columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models to train with hyperparameter tuning
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=20000),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            }
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "params": {}
        },
        "SVM": {
            "model": SVC(),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(),
            "params": {
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        }
    }

    results = []
    for name, config in models.items():
        model = config["model"]
        params = config["params"]

        if hyperparameter_tuning == "yes" and params:
            # Perform hyperparameter tuning
            grid_search = GridSearchCV(model, params, cv=3, scoring="accuracy")
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            accuracy = grid_search.best_score_
        else:
            # Train without hyperparameter tuning
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

        # Get feature importances if the model has them
        feature_importances = (
            model.feature_importances_
            if hasattr(model, "feature_importances_")
            else np.zeros(len(X.columns))
        )
        results.append(
            {
                "model_name": name,
                "accuracy": accuracy,
                "feature_importances": sorted(
                    zip(X.columns, feature_importances),
                    key=lambda x: x[1],
                    reverse=True,
                ),
            }
        )

    # Sort results by accuracy in descending order
    # If accuracy is the same, use the predefined MODEL_ORDER for tie-breaking
    results.sort(key=lambda x: (-x["accuracy"], MODEL_ORDER.get(x["model_name"], 0)))

    # Save the best model
    best_model_name = results[0]["model_name"]
    best_model = models[best_model_name]["model"]
    best_model.fit(X_train, y_train)  # Retrain on the full training data
    joblib.dump(best_model, "best_model.pkl")

    return templates.TemplateResponse(
        "results.html",
        {"request": request, "results": results, "best_model": best_model_name},
    )

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    global feature_columns
    if not feature_columns:
        return templates.TemplateResponse("error.html", {"request": request, "error": "No model trained yet."})
    return templates.TemplateResponse("predict.html", {"request": request, "feature_columns": feature_columns})

# @app.post("/predict", response_class=HTMLResponse)
# async def make_prediction(
#     request: Request,
#     form_data: dict = Depends(Request.form)
# ):
#     global best_model, feature_columns

#     if not best_model:
#         return templates.TemplateResponse("error.html", {"request": request, "error": "No model trained yet."})

#     # Parse form data
#     data = await form_data.form()
#     input_data = {col: float(data[col]) for col in feature_columns}

#     # Convert input data to DataFrame
#     input_df = pd.DataFrame([input_data])

#     # Make predictions
#     prediction = best_model.predict(input_df)

#     return templates.TemplateResponse(
#         "predict_results.html",
#         {"request": request, "prediction": prediction[0]},
#     )
@app.post("/predict", response_class=HTMLResponse)
async def make_prediction(
    request: Request
):
    global best_model, feature_columns

    if not best_model:
        return templates.TemplateResponse("error.html", {"request": request, "error": "No model trained yet."})

    # Extract form data
    form_data = await request.form()

    try:
        # Convert input data to appropriate format
        input_data = {col: float(form_data[col]) for col in feature_columns}

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make predictions
        prediction = best_model.predict(input_df)

        return templates.TemplateResponse(
            "predict_results.html",
            {"request": request, "prediction": prediction[0]},
        )
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": f"Prediction error: {str(e)}"})
