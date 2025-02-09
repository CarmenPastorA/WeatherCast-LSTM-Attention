
# Weather Time Series Forecasting with LSTM and Attention

This project focuses on forecasting weather time series data using Long Short-Term Memory (LSTM) networks combined with attention mechanisms. The model architecture is designed to capture temporal dependencies in weather data and highlight important features that influence the forecast.

The implementation leverages PyTorch and PyTorch Lightning for efficient model training and evaluation.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used
The following libraries and tools are used in this project:

- **Python Libraries:**
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `torch`
  - `pytorch_lightning`

- **Additional Tools:**
  - `PyTorch Lightning` for model structuring and training.
  - `TensorBoardLogger` and `CSVLogger` for logging.

## Dataset
The dataset used in this project consists of historical weather data. It is preprocessed to include relevant features necessary for time series forecasting.

## Methodology
The project employs the following steps:

1. **Data Preprocessing:** Cleaning and scaling the data.
  - Normalization using `MinMaxScaler` and `StandardScaler`.
  - Train-test split using `train_test_split` from `scikit-learn`.
2. **Model Development:** Building LSTM models enhanced with attention mechanisms.
3. **Training:** Using `PyTorch Lightning` for structured and efficient training.
4. **Evaluation:** Comparing model performance against baseline model.

## Results
The model's performance is compared against baseline model to highlight improvements. Detailed results include:

- Baseline model result using traditional machine learning methods.
- LSTM and attention model results.
- Comparative analysis of different model performances.

![Model Results](images/results_table.png)
![Model Results](images/best_model.png)
