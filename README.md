# Global Asset Volatility Forecaster (TimeMixer)

[![Live Demo](https://img.shields.io/badge/Live_Dashboard-Ready-success?style=for-the-badge&logo=render)](https://global-asset-volatility-forecasting.onrender.com)

A production-grade, Full-Stack Machine Learning application designed to forecast financial market volatility across 40 global assets. Built upon the latest **TimeMixer** (arXiv:2410.09062) deep learning architecture, this system completely decouples GPU training from real-time CPU inference via a high-performance **FastAPI** web dashboard.

![Dashboard Preview](static/dashboard_preview.png) 
<img width="1898" height="856" alt="image" src="https://github.com/user-attachments/assets/79a59cd7-73b8-494b-ae28-576f21ed6b77" />


---

## 🚀 Features

- **TimeMixer Architecture**: Utilizes advanced multiscale mixing configurations to out-predict standard GARCH(1,1) baselines natively across 5 distinct horizons (12, 96, 192, 336, 720 days).
- **Global Asset Matrix**: Supports live, granular inference for 40 distinct tickers grouped into Equities, ETFs, Crypto, and Forex.
- **Instant Inference**: Fetches real-time price action via `yfinance`, dynamically computes the Yang-Zhang volatility estimator, and runs forward passes in milliseconds.
- **Glassmorphism Trading UI**: Features an interactive, pure JS/Chart.js frontend for strict date-selection highlighting and anomaly detection. 
- **Decoupled Architecture**: Training is isolated to `PyTorch Lighting/NeuralForecast`. The Web Application pulls statically-saved Pickled weights for immediate edge-deployment.

## 🛠 Tech Stack

- **Machine Learning**: `PyTorch`, `NeuralForecast`, `Arch`
- **Data Engineering**: `Pandas`, `Numpy`, `yfinance`
- **Backend / API**: `FastAPI`, `Uvicorn`
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism), `Chart.js`

---

## 🏃‍♂️ How to Run the App Locally

If you want to run the live dashboard on your own machine without a GPU, do the following:

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Web Server**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   *Note: Uvicorn automatically binds the static frontend files and API endpoints.*

3. **Open the Dashboard**
   Navigate to `http://localhost:8000` in any web browser to view the live TimeMixer Matrix.

---

## 🧠 Model Weights & Architecture

**"Where do the weights come from?"**
This repository permanently tracks the statically saved `h{horizon}.pkl` model weights in the `models/timemixer/` directory.

When the FastAPI server initiates an inference request via `POST /api/predict`, it parses the required asset class, ticker, and horizon, instantaneously loads the pre-trained weights from disk into CPU memory, runs the forward pass, and serves the JSON array. This is how the system handles fraction-of-a-second inference globally without requiring an active H100 inference cluster.

---

## 📊 Repository Structure & Engineering

Unlike standard data science repositories riddled with stale `.ipynb` files, this repository is generated algorithmically for strict reproducibility.

```text
├── Data/                   # Frozen historical data for reproducibility testing.
├── Notebooks/              # Algorithmically generated Jupyter pipelines.
├── scripts/                # Generators (ETL and Notebook scaffolding logic).
│   ├── gen_feat_nb.py      # Feature engineering scripts
│   ├── gen_structured_tm_nb.py  # Model scaffolding
│   └── live_demo.py        # Terminal-based CLI tester
├── models/                 # Pre-trained Pickled weights
├── src/                    # Core Model classes & Yang-Zhang Processors
├── static/                 # Front-end Web GUI assets
├── app.py                  # Production FastAPI web server
├── config.yaml             # Single-source-of-truth configuration
└── README.md
```

## 📝 Statistical Validation (Diebold-Mariano)
This implementation was rigorously validated via Wilcoxon signed-rank significance testing against symmetric GARCH(1,1) implementations, resulting in significant `sMAPE` differential superiority on extended forecasting horizons (>96 days). See Phase 5 logic across the `results/` matrix for raw tabular data.
