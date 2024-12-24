from fastapi import FastAPI, Header, HTTPException, File, UploadFile
from typing import List, Annotated
from os import path, listdir
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pydantic import BaseModel

app = FastAPI()

MODELS_DIRECTORY = "models"

# Utility functions
def load_model(model_name: str):
    model_path = path.join(MODELS_DIRECTORY, f"{model_name}.pkl")
    if not path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model not found")
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)

def save_model(model: str, new_model_name: str):
    model_path = path.join(MODELS_DIRECTORY, f"{new_model_name}.pkl")
    with open(model_path, "wb") as model_file:
        return pickle.dump(model, model_file)

# Task 1 - continue-train endpoint
@app.post("/continue-train")
def continue_train(model_name: str, new_model_name: str, train_input: UploadFile = File(...)) -> str:
    # Mount the train input file
    contents = train_input.file
    df = pd.read_csv(contents)

    # Check for "target" column, for our project the column is "Diagnosis"
    if "Diagnosis" not in df.columns:
        raise HTTPException(status_code=400, detail="PANIC. Missing 'Diagnosis' column in the dataset")
    
    X_input_train = df.drop(columns=["Diagnosis"])
    y_input_train = df["Diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X_input_train, y_input_train, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Load the input model
    model = load_model(model_name)
    model.fit(X_train_scaled, y_train)
    save_model(model, new_model_name)

    # Evaluate the new model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    metric = f"{new_model_name} model's accuracy: {accuracy:.2f}"
    return metric

# Task 2
@app.post("/predict")
def predict(model_name: str, input: UploadFile = File(...)):
    # Mount the input file
    contents = input.file
    df = pd.read_csv(contents)

    # Check for "target" column, for our project the column is "Diagnosis"
    if "Diagnosis" in df.columns:
        raise HTTPException(status_code=400, detail="PANIC. 'Diagnosis' column in the dataset")

    # Load the input model
    model = load_model(model_name)

    # Scale the features
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(df)

    # Make predictions
    predictions = model.predict(input_scaled)
    result = {"predictions": predictions.tolist()}
    return result


# Task 3 - get models endpoint
@app.get("/models", response_model=List[str])
def get_models() -> List[str]:
    try:
        models = [model for model in listdir(MODELS_DIRECTORY) if model.endswith(".pkl")]
        models_name = [path.splitext(model)[0] for model in models]
        return models_name
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))