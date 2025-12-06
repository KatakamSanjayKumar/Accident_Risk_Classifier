# Accident_Risk_Classifier

This project provides a comprehensive notebook for predicting accident risk levels on road segments using machine learning and deep learning methods. The notebook walks through the entire data science pipeline — including data loading, preprocessing, feature engineering, model selection, evaluation, and hyperparameter tuning — with an emphasis on interpretability and flexibility.

## Dataset

The notebook utilizes a synthetic road accident dataset (`synthetic_road_accidents.csv`) containing the following (sample) features:
- Road attributes (type, number of lanes, curvature, etc.)
- Environmental attributes (lighting, weather, time_of_day)
- Safety indicators (road signs, public road, holiday, school_season)
- Accident statistics (`num_reported_accidents`, `accident_risk`)

## Notebook Highlights

- **Data Preprocessing:** Handles categorical, boolean, and numerical variables using appropriate encodings (e.g., one-hot encoding for categoricals).
- **Feature Engineering:** Demonstrates how to derive new features, such as `speed_ratio` (vehicle_speed vs. speed_limit) or flags for adverse weather conditions.
- **Model Training and Selection:** Implements logistic regression as a baseline, with grid search hyperparameter optimization. Also compares performance with advanced models like XGBoost and LightGBM, including cross-validation and comprehensive metric reporting.
- **Deep Learning:** Contains a template for training a simple neural network on the dataset using Keras/TensorFlow.
- **Evaluation:** Uses accuracy and classification reports to compare models and select the optimal approach.

## Usage

1. Place `synthetic_road_accidents.csv` in the same directory as the notebook or update the path as needed.
2. Run through the code cells sequentially to process data, fit models, and evaluate performance.
3. Modify or extend the hyperparameter grids, add new models, or apply further feature engineering as desired.

## Requirements

- Python 3.x
- pandas, numpy, scikit-learn
- xgboost, lightgbm (for advanced models)
- tensorflow, keras (for deep learning)
- Jupyter Notebook or Google Colab

Install dependencies via:

```sh
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow keras
```

## Notes

- The notebook is modular; you can run only the sections you need to quickly benchmark or compare algorithms.
- You may customize the risk threshold, target variable encoding (e.g., binary thresholding at 0.5), or add additional feature transformations.

## License

Feel free to use, modify, or extend the notebook for your own accident-risk modeling experiments or educational purposes.
