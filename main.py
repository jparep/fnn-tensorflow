import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Logging Configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

def load_data(file_path):
    """Loads data from a specified CSV file path."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data successfully loaded from {file_path}")
        return df
    except FileNotFoundError as e:
        logging.error(f'File not found: {file_path}')
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f'No data found in file: {file_path}')
        raise e
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        raise e

def handle_outliers(df, columns):
    """Handles outliers using the 1st and 3rd quartiles (IQR method)."""
    for col in columns:
        if (col in df.columns):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
            logging.info(f"Outliers handled for column: {col}")
    return df

def feature_engineering(df):
    """Performs feature engineering on the DataFrame."""
    if 'Avg. Area Number of Rooms' in df.columns and 'Avg. Area Number of Bedrooms' in df.columns:
        df['RoomBedroom_Ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']
        logging.info("Feature RoomBedroom_Ratio created.")
    return df

def preprocess_data(df):
    """Prepares the data for training."""
    X = df.drop(['Price', 'Address'], axis=1, errors='ignore')
    y = df['Price']
    
    # Standardize the feature matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def build_model(input_shape):
    """Builds and compiles the neural network model."""
    model = Sequential([
        Dense(64, input_shape=(input_shape,), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    logging.info("Neural Network model built and compiled.")
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Trains the neural network model."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)
    logging.info("Model training completed.")
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluates the model performance on the test set."""
    y_pred = model.predict(X_test).flatten()
    
    eval_metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    logging.info(f"Model Evaluation: {eval_metrics}")
    return eval_metrics

def main():
    # Load Data
    file_path = './data/USA_Housing.csv'
    df = load_data(file_path)
    
    # Handle Outliers
    outlier_columns = ['Avg. Area Income', 'Avg. Area House Age']
    df = handle_outliers(df, outlier_columns)
    
    # Feature Engineering
    df = feature_engineering(df)
    
    # Preprocessing Data
    X_processed, y = preprocess_data(df)
    
    # Split Data into Training, Validation, and Testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y, test_size=0.3, random_state=12)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=12)
    
    # Build the Neural Network Model
    input_shape = X_train.shape[1]
    model = build_model(input_shape)
    
    # Train the Model
    model, history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate the Model
    eval_metrics_best_model = evaluate_model(model, X_test, y_test)
    
    # Print Best Model evaluation metrics
    print("Neural Network Best Model Metrics:")
    for metric, value in eval_metrics_best_model.items():
        print(f"    - {metric}: {value:,.4f}")

if __name__ == "__main__":
    main()
