# 🛡️ AI Cyber Threat Detection

An interactive web application that helps **colleges, small organizations, and small businesses** review manually entered activity patterns — such as repeated failed logins, access from an unknown device, or unusually high data transfers — estimate risk, explain the indicators behind the result, and suggest practical next steps for a person to review.

> **Disclosure:** This tool provides advisory, educational guidance only. An anomaly indicates *unusual statistical activity*, not proof of an attack. All recommendations are for a person to review — the application never automatically blocks accounts, devices, or network access.

---

## Architecture

| Layer | Technology |
|---|---|
| **Interface** | [Streamlit](https://streamlit.io/) 1.x (Python) — SOC dashboard styled with pure CSS/SVG |
| **Machine Learning** | Scikit-learn `IsolationForest` — unsupervised anomaly detection |
| **Data** | Pandas, NumPy — feature construction and tabular history |
| **Model Persistence** | Joblib — serialize and load the trained artifact |

**Why Streamlit?** The environment (Windows, Python 3.13) does not have Node.js, npm, or FastAPI installed. Streamlit with custom CSS delivers the full SOC dashboard experience without additional tooling.

---

## Features

- **Activity input form** — login attempts, device recognition status, data volume (MB), session time window, and an optional reference note
- **Isolation Forest anomaly detection** — flags unusual feature combinations as statistical outliers
- **Transparent threat scoring** — 0–100 heuristic score combining model output with observable telemetry rules
- **Visual SVG semi-circular gauge** — color-coded per risk level (cyan = low, amber = medium, red = high)
- **Factor breakdown panel** — explains every contributing indicator with point allocations
- **Recommended response panel** — practical, advisory next steps matched to the assessed risk level
- **Synthetic demonstration profiles** — 4 pre-built scenarios (Standard User, Repeated Logins, Data Transfer Spike, Off-Hours Unknown Device)
- **In-memory session history** — ephemeral log with CSV export; no data written to disk
- **Governance tab** — purpose, model methodology, privacy principles, and operational limitations
- **Responsive layout** — adapts to desktop and mobile widths

---

## Privacy & Safety Principles

- **Zero credential ingestion** — no passwords, tokens, usernames, API keys, or PII are ever collected
- **Ephemeral session memory** — all input data lives in browser session RAM only; never written to disk or transmitted
- **Advisory only** — no automated account lockouts, device blocks, packet sniffing, or offensive actions
- **Human-in-the-loop** — all outputs are suggestions for a person to review and decide

---

## Limitations

- The Isolation Forest model is trained on **synthetic probability distributions**, not live enterprise data. It is not calibrated for production network traffic.
- A flagged anomaly does not prove malicious intent. High data transfer may be a scheduled backup; multiple failed logins may be a forgotten password.
- The 0–100 threat score is an advisory heuristic, not a forensic finding.
- Always corroborate through secondary approved communication channels before initiating any institutional incident response.

---

## Setup & Running

### Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13
- A terminal (PowerShell, Command Prompt, or Bash)

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train the Isolation Forest model

```bash
python train_model.py
```

This generates `cyber_threat_model.joblib` in the project directory. You can also click **"Train Model Now (In-App)"** in the sidebar if the file is missing.

### 3 — Launch the application

```bash
python -m streamlit run app.py
```

The app opens at **http://localhost:8501** in your default browser.

---

## Project Files

```
├── app.py                        # Streamlit application — UI, CSS, scoring, results, history
├── train_model.py                # Synthetic dataset generator + Isolation Forest training
├── cyber_threat_model.joblib     # Serialized trained model artifact
├── requirements.txt              # Python package dependencies
├── .gitignore                    # Ignored cache, model files, virtual environments
└── README.md                     # This file
```

---

## License

Open-source. Intended for educational and organizational advisory use only.
