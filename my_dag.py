import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob, os, joblib, requests
from airflow.decorators import dag, task

""" 

Purpose:
    Airflow DAG: extract new applicant batches, score them with the saved
    Random Forest pipeline, push results to Power BI streaming dataset.

Pseudocode:
    extract():
        - glob new CSV files in landing folder not yet processed
        - concat into one DataFrame, return it
        - move/mark processed files so they aren't re-scored next run
    score(df):
        - load joblib pipeline
        - predict charges on the six raw columns
        - attach predicted_charges + timestamp to df
        - return scored df
    load(scored_df):
        - for each row, POST as JSON to Power BI streaming Push URL
        - log success/failure per row

Schedule:
    Runs monthly, quartely, weekly or daily, simulating near-continuous intake.
"""

sim_dir = "data/sim_data"
processed_dir = "data/processed"
model_path = "models/rf_underwriting_pipeline.pkl"
push_url = ... #for power bi url

@dag(schedule="*/5 * * * *", start_date=(2026, 8, 17), catchup=False)
def underwriting_scoring():

    @task
    def extract():
        files = glob.glob(f"{sim_dir}/*.csv")
        if not files:
            return None
        df = pd.concat([pd.read_csv(f) for f in files])
        os.makedirs(processed_dir, exist_ok=True)
        for f in files:
            os.rename(f, f"{processed_dir}/{os.path.basename(f)}")
            return df.to_json()

    @task
    def score(df_json):
        if df_json is None:
            return None

        df = pd.read_json(df_json)
        pipe = joblib.load(model_path)
        df['predicted_charges'] = pipe.predict(df)
        df['scored_at'] = datetime.now().isoformat()
        return df.to_json()

    @task
    def load(df_json):
        if df_json is None:
            return None
        df = pd.read_json(df_json)
        for row in df.to_dict(orient='records'):
            requests.post(push_url, json=[row])

    load(score(extract()))

underwriting_scoring()