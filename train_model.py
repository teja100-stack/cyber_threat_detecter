"""
=============================================================================
AI Cyber Threat Detection and Intelligent Response - Model Training Script
=============================================================================
Project: Educational Prototype for Small Organizations & Academic Labs
Algorithm: Isolation Forest (Unsupervised Anomaly Detection)
Output File: cyber_threat_model.joblib
=============================================================================

HOW ISOLATION FOREST WORKS (EDUCATIONAL SUMMARY FOR HACKATHONS):
-----------------------------------------------------------------------------
1. Anomaly detection differs from supervised classification because real-world
   cyber attacks are rare, rapidly evolving, and rarely labeled.
2. Isolation Forest isolates anomalies by building random decision trees.
3. Normal instances cluster together and require MANY random cuts/splits to isolate.
4. Anomalies (e.g., 20 login attempts, unknown device at 3 AM, or 8,000 MB data transfer)
   stand far away from the norm and get isolated in VERY FEW splits (shallow path length).
5. The algorithm outputs:
   - Prediction: 1 for Normal, -1 for Anomaly
   - Decision Score: Average path length metric (lower/negative = more anomalous)
=============================================================================
"""

import os
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

MODEL_FILENAME = "cyber_threat_model.joblib"
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "login_attempts",
    "is_known_device",
    "data_transferred_mb",
    "is_off_hours"
]


def generate_synthetic_security_logs(n_normal=2200, n_anomalies=220, random_state=RANDOM_SEED):
    """
    Generates synthetic access logs representing common organizational activity.
    
    Features:
    - login_attempts: Number of authentication attempts during the event
    - is_known_device: 1 if device fingerprint is in organization registry, 0 if new/unknown
    - data_transferred_mb: Amount of data transferred in megabytes
    - is_off_hours: 1 if event occurred outside business hours (late night/weekend), 0 otherwise
    """
    np.random.seed(random_state)
    
    # -------------------------------------------------------------
    # 1. Normal Baseline Traffic (Typical Daily Staff/Student Use)
    # -------------------------------------------------------------
    # - Low login attempts (1-2 attempts typically, occasional typo up to 3)
    # - High known device percentage (~95%)
    # - Standard file/web browsing payloads (5 MB to 150 MB)
    # - Mostly business hours (~88% during day)
    normal_login = np.random.choice([1, 2, 3], size=n_normal, p=[0.82, 0.14, 0.04])
    normal_device = np.random.choice([1, 0], size=n_normal, p=[0.95, 0.05])
    normal_data = np.clip(np.random.lognormal(mean=3.2, sigma=0.75, size=n_normal), 0.5, 350.0)
    normal_hours = np.random.choice([0, 1], size=n_normal, p=[0.88, 0.12])
    
    df_normal = pd.DataFrame({
        "login_attempts": normal_login,
        "is_known_device": normal_device,
        "data_transferred_mb": np.round(normal_data, 2),
        "is_off_hours": normal_hours,
        "ground_truth_label": 0  # 0 = Normal
    })
    
    # -------------------------------------------------------------
    # 2. Suspicious Anomalies (Simulated Threat Vectors)
    # -------------------------------------------------------------
    # Scenario A: Brute Force / Credential Guessing
    n_brute = n_anomalies // 3
    brute_login = np.random.randint(5, 30, size=n_brute)
    brute_device = np.random.choice([0, 1], size=n_brute, p=[0.80, 0.20])
    brute_data = np.random.uniform(1.0, 80.0, size=n_brute)
    brute_hours = np.random.choice([0, 1], size=n_brute, p=[0.40, 0.60])
    
    df_brute = pd.DataFrame({
        "login_attempts": brute_login,
        "is_known_device": brute_device,
        "data_transferred_mb": np.round(brute_data, 2),
        "is_off_hours": brute_hours,
        "ground_truth_label": 1
    })
    
    # Scenario B: Large Data Exfiltration / Unusual Bulk Transfer
    n_exfil = n_anomalies // 3
    exfil_login = np.random.choice([1, 2, 3], size=n_exfil, p=[0.70, 0.20, 0.10])
    exfil_device = np.random.choice([0, 1], size=n_exfil, p=[0.60, 0.40])
    exfil_data = np.random.uniform(1500.0, 15000.0, size=n_exfil)  # Massive spike
    exfil_hours = np.random.choice([1, 0], size=n_exfil, p=[0.70, 0.30])
    
    df_exfil = pd.DataFrame({
        "login_attempts": exfil_login,
        "is_known_device": exfil_device,
        "data_transferred_mb": np.round(exfil_data, 2),
        "is_off_hours": exfil_hours,
        "ground_truth_label": 1
    })
    
    # Scenario C: Off-Hours Intrusion with Unknown Device
    n_intrusion = n_anomalies - n_brute - n_exfil
    intrusion_login = np.random.randint(3, 12, size=n_intrusion)
    intrusion_device = np.zeros(n_intrusion, dtype=int)  # 100% unknown device
    intrusion_data = np.random.uniform(400.0, 3000.0, size=n_intrusion)
    intrusion_hours = np.ones(n_intrusion, dtype=int)     # 100% off-hours / late night
    
    df_intrusion = pd.DataFrame({
        "login_attempts": intrusion_login,
        "is_known_device": intrusion_device,
        "data_transferred_mb": np.round(intrusion_data, 2),
        "is_off_hours": intrusion_hours,
        "ground_truth_label": 1
    })
    
    # Combine and shuffle
    df_combined = pd.concat([df_normal, df_brute, df_exfil, df_intrusion], ignore_index=True)
    df_shuffled = df_combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    return df_shuffled


def train_isolation_forest():
    """
    Trains the Isolation Forest model on synthetic data and saves the artifact.
    """
    print("=" * 65)
    print("  AI Cyber Threat Detection - Training Isolation Forest Model")
    print("=" * 65)
    
    print("[1/4] Generating synthetic security dataset...")
    df = generate_synthetic_security_logs(n_normal=2400, n_anomalies=240, random_state=RANDOM_SEED)
    print(f"      Total records generated: {len(df):,}")
    print(f"      - Normal baseline: {(df['ground_truth_label'] == 0).sum():,} records")
    print(f"      - Simulated threat anomalies: {(df['ground_truth_label'] == 1).sum():,} records")
    
    # Extract features for training
    X = df[FEATURE_COLUMNS]
    
    # Isolation Forest setup
    # contamination = estimated proportion of outliers in the data (~9-10%)
    contamination_rate = 0.09
    print(f"\n[2/4] Initializing IsolationForest (n_estimators=100, contamination={contamination_rate})...")
    
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination_rate,
        random_state=RANDOM_SEED,
        n_jobs=1
    )
    
    print("[3/4] Fitting model to activity distribution...")
    model.fit(X)
    
    # Quick sanity check predictions
    preds = model.predict(X)  # 1: normal, -1: anomaly
    detected_anomalies = (preds == -1).sum()
    print(f"      Model flagged {detected_anomalies} anomalies ({detected_anomalies/len(X):.1%}) in training data.")
    
    # Bundle artifact with metadata
    artifact = {
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "contamination": contamination_rate,
        "training_samples": len(df),
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0.0",
        "feature_stats": {
            "login_attempts_mean": float(df["login_attempts"].mean()),
            "data_transferred_mb_median": float(df["data_transferred_mb"].median()),
            "data_transferred_mb_95th": float(df["data_transferred_mb"].quantile(0.95))
        }
    }
    
    output_path = os.path.abspath(MODEL_FILENAME)
    print(f"\n[4/4] Serializing model bundle with Joblib to: {output_path}")
    joblib.dump(artifact, output_path)
    print("=" * 65)
    print("SUCCESS: Model trained and saved successfully!")
    print(f"File created: {MODEL_FILENAME}")
    print("You can now run the web interface with: streamlit run app.py")
    print("=" * 65)
    return artifact


if __name__ == "__main__":
    train_isolation_forest()
