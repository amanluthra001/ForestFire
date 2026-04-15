# Forest Fire Alert System

A Python project for forest fire risk prediction, real-time alerts, and visualization (Kelowna dataset).
This repository helps monitor wildfire risk using weather, satellite, and sensor inputs, with model inference + dashboards.

## 🚀 Project overview

- Data sources:
  - Historical weather/station data (`stations/`)
  - Satellite burn data (`satellite-burn/`)
  - Region-level evaluation (`evaluation/`)
- Model artifacts:
  - Trained models and ensemble outputs (`models2/`)
- Utilities:
  - Data preparation (`build_dataset.py`, `make_kelowna_uniform_grid.py`)
  - Inference/prediction (`wildfire_predictor_optimized.py`, `check_alerts.py`)
  - Visualization dashboards (`forecast_dashboard.py`, `realtime_dashboard.py`, `plots.py`)

## 🧠 Features

- Risk classification and probability forecasting
- Ensemble model stacking (CatBoost + others)
- Region/zone evaluation CSV outputs
- Real-time alert generation
- Interactive plotting and map dashboards

## 📁 Repository structure

- `dataset.csv` – raw input dataset
- `models2/` – model artifacts, ensemble metadata
- `results_opt/` – out-of-fold metrics, prediction logs, plots
- `evaluation/` – region prediction result sheets
- `satellite-burn/`, `stations/` – supporting reference data

## 🛠️ Quickstart

1. Install dependencies:
   ```bash
   pip install -r libs.txt
   ```
2. Build dataset (if needed):
   ```bash
   python build_dataset.py
   ```
3. Train or run model:
   ```bash
   python wildfire_predictor_optimized.py
   ```
4. Run alert check:
   ```bash
   python check_alerts.py
   ```
5. Launch dashboards:
   ```bash
   python forecast_dashboard.py
   python realtime_dashboard.py
   ```

## 📊 Evaluation

- Check `results_opt/ensemble_final_metrics.csv`
- Explore per-region metrics in `evaluation/region_*.csv`
- Use `plots/` for validation graphs and calibration curves

## 💡 Notes

- Ensure CSVs in `stations/` and `satellite-burn/` are formatted with required columns before training.
- Adjust hyperparameters and thresholds in `wildfire_predictor_optimized.py` to fit local deployment needs.

## 🔧 Recommendations

- Add `.gitignore` for Python artifacts (`__pycache__/`, `.venv/`, `*.pyc`)
- Include `LICENSE` and `CODE_OF_CONDUCT` for open-source clarity
- Add unit tests in `tests/` for functions in `evaluation.py`, `metrics.py`, etc.

