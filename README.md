# Real Estate House Price Prediction with Deep Learning

This project implements a deep learning model for predicting real estate house prices using a Feedforward Neural Network (FNN). The project incorporates advanced techniques for hyperparameter tuning, cross-validation, and model evaluation, making it suitable for professional real-world applications in 2024.

## Project Overview

The goal of this project is to predict house prices based on various features of real estate data. The model is built using TensorFlow and Keras, with Keras Tuner used for hyperparameter optimization. The project includes steps for data preprocessing, outlier handling, feature engineering, and model training. The final model is evaluated using cross-validation to ensure its robustness and generalizability.

## Features

- **Outlier Handling**: Outliers are handled using the Interquartile Range (IQR) method.
- **Feature Engineering**: Additional features are created to improve model performance.
- **Data Preprocessing**: Data is standardized using `StandardScaler`.
- **Hyperparameter Tuning**: Hyperparameters are optimized using Keras Tuner's `RandomSearch`.
- **Cross-Validation**: The model is validated using k-fold cross-validation to ensure generalizability.
- **Advanced Logging**: Detailed logging is implemented for tracking model performance and tuning results.
- **Learning Rate Scheduling**: `ReduceLROnPlateau` is used to dynamically adjust the learning rate during training.

## Installation

To run this project, you'll need to have Python installed along with the required packages. You can install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
