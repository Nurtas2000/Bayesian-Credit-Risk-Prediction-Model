import pymc3 as pm
import pandas as pd
import numpy as np
import joblib
from model import BayesianCreditRiskModel  # Импорт класса модели

def train_model(data_path, model_path):
    df = pd.read_csv(data_path)
    model = BayesianCreditRiskModel()
    
    X = df.drop('loan_status', axis=1).values
    y = df['loan_status'].values
    
    model.build_model(X, y)
    trace = model.fit_model(X, y)
    
    # Сохранение модели
    joblib.dump({
        'model': model,
        'trace': trace
    }, model_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/processed_data.csv")
    parser.add_argument("--model", default="models/trained_model.pkl")
    args = parser.parse_args()
    
    train_model(args.data, args.model)
