# Box Office Revenue Prediction ML Project

A machine learning project for predicting box office revenue using various algorithms including scikit-learn, LightGBM, and XGBoost. The project implements a complete ML pipeline from data preprocessing to model comparison.

## Features

- **Multiple ML Algorithms**: Compares Linear Regression, Random Forest, LightGBM, and XGBoost
- **Data Preprocessing**: Handles missing values, feature scaling, and encoding
- **Model Evaluation**: Comprehensive metrics including RMSE, MAE, and R²
- **Ready for Kaggle Datasets**: Designed to work with movie datasets from Kaggle

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- LightGBM
- XGBoost
- matplotlib
- seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alinapradhan/Box-office-revenue-prediction-ML-project-.git
cd Box-office-revenue-prediction-ML-project-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### With Sample Data

Run the project with a generated sample dataset:

```bash
python main.py
```

### With Kaggle Dataset

1. Download a movie dataset from Kaggle (e.g., TMDB Movie Dataset)
2. Place the CSV file in the project directory
3. Modify `main.py` to load your dataset:

```python
df = load_data('path/to/your/kaggle/dataset.csv')
```

### Expected Output

The script will:
1. Load and preprocess the data
2. Train four different models (Linear Regression, Random Forest, LightGBM, XGBoost)
3. Evaluate each model with RMSE, MAE, and R² metrics
4. Compare models and identify the best performer
5. Make a sample prediction with the best model

## Project Structure

```
Box-office-revenue-prediction-ML-project-/
├── main.py              # Main script with ML pipeline
├── requirements.txt     # Project dependencies
├── README.md           # Project documentation
├── .gitignore          # Git ignore rules
└── LICENSE             # License file
```

## Models Implemented

1. **Linear Regression**: Baseline model for comparison
2. **Random Forest**: Ensemble method with decision trees
3. **LightGBM**: Gradient boosting framework optimized for speed
4. **XGBoost**: Gradient boosting with regularization

## Dataset Features

The model expects the following features (can be customized):
- `budget`: Movie production budget
- `runtime`: Movie duration in minutes
- `vote_average`: Average user rating
- `vote_count`: Number of user votes
- `popularity`: Popularity metric
- `genre_*`: Binary genre indicators
- `release_month`: Month of release
- `has_sequel`: Whether the movie has a sequel

## Customization

### Adding New Features

Modify the `preprocess_data()` function in `main.py` to add custom feature engineering.

### Using Different Models

Add new models in the `main()` function following the existing pattern:

```python
models['Your Model'] = your_training_function(X_train, y_train)
```

### Adjusting Hyperparameters

Edit the model training functions (`train_lightgbm`, `train_xgboost`, etc.) to tune hyperparameters.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under MIT.

## Acknowledgments

- Dataset sources: Kaggle
- Libraries: pandas, scikit-learn, LightGBM, XGBoost
