import pickle
import pandas as pd
import numpy as np

def load_artifacts(model_dir='models'):
    with open(f'{model_dir}/final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{model_dir}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{model_dir}/columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, scaler, columns


def preprocess(input_dict, columns):
    df = pd.DataFrame([input_dict])
   
    df = pd.get_dummies(df, drop_first=False)
    df = df.reindex(columns=columns, fill_value=0)
    return df


def predict_churn(input_dict, model_dir='models'):
    model, scaler, columns = load_artifacts(model_dir)
    df = preprocess(input_dict, columns)
    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0][1]
    return prob