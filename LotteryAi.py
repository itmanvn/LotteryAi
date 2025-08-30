import socket
import sys 
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from keras import layers
from art import text2art 

def guard(*args, **kwargs):
    """Raises an exception to prevent any network connections."""
    raise Exception("Internet access is disabled for security or offline operation.")
socket.socket = guard

def print_intro():
    try:
        ascii_art = text2art("LotteryAi")
        print("=" * 60)
        print("LotteryAi")
        print("Lottery prediction artificial intelligence")
        print("=" * 60)
        print("Starting...")
        print("=" * 60)
    except Exception as e:
        print(f"Error displaying introduction: {str(e)}")
        sys.exit(1)

# --- Data Loading and Preprocessing ---
def load_data():
    """Loads lottery data from 'data.txt', preprocesses it, and splits it into training and validation sets."""
    try:
        if not tf.io.gfile.exists('data.txt'):
            raise FileNotFoundError("Error: 'data.txt' not found in the current directory.")

        # Load data from 'data.txt' using numpy's genfromtxt.
        data = np.genfromtxt('data-dacbiet.txt', delimiter=',', dtype=int)

        # Check if the loaded data array is empty
        if data.size == 0:
            raise ValueError("Error: 'data.txt' is empty or contains improperly formatted data.")

        # If data is 1D, reshape it to 2D with 1 feature per sample
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            print("Data reshaped from 1D to 2D format (samples, 1 feature)")

        # Optional: Replace any placeholder values (like -1) with 0.
        data[data == -1] = 0

        # Determine the size of the training set (80% of the total data)
        train_size = int(0.8 * len(data))
        if train_size == 0:
            raise ValueError("Error: Dataset is too small to split into training and validation sets (needs at least 5 rows).")

        # Split the data into training and validation sets
        train_data = data[:train_size]
        val_data = data[train_size:]

        # Find the maximum lottery number value in the entire dataset.
        max_value = np.max(data)

        return train_data, val_data, max_value
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        sys.exit(1)
    except ValueError as val_error:
        print(val_error)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {str(e)}")
        sys.exit(1)

# --- Model Creation ---
def create_model(max_value):
    """Creates and compiles the Keras Sequential model for lottery prediction."""
    try:
        model = keras.Sequential([
            # Embedding layer: Convert lottery numbers to dense vectors
            layers.Embedding(input_dim=max_value + 1, output_dim=128),
            
            # LSTM layers for sequence learning
            layers.LSTM(256, return_sequences=True),
            layers.LSTM(128),
            
            # Dense layers for feature extraction
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer: Predict probability for each possible number
            layers.Dense(max_value + 1, activation='softmax')
        ])

        model.compile(
            loss='sparse_categorical_crossentropy',  # Better for integer labels
            optimizer='adam',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        print(f"Error creating the neural network model: {str(e)}")
        sys.exit(1)

# --- Model Training ---
def train_model(model, train_data, val_data):
    """Trains the Keras model using the provided training and validation data."""
    try:
        print("Starting model training...")
        
        # Prepare data for sparse categorical crossentropy
        # We'll use the lottery numbers as both input and target
        x_train = train_data
        y_train = train_data.flatten()  # Flatten for sparse categorical
        
        x_val = val_data
        y_val = val_data.flatten()
        
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=50,  # Reduced epochs for faster training
            verbose=1,
            batch_size=4
        )
        print("Model training completed.")
        return history
    except Exception as e:
        print(f"An error occurred during model training: {str(e)}")
        sys.exit(1)

# --- Number Prediction ---
def predict_numbers(model, input_data):
    """Uses the trained model to predict lottery numbers based on input data."""
    try:
        print("Generating predictions...")
        predictions = model.predict(input_data)

        # Get the top 5 highest probability numbers
        top_k = 5
        indices = np.argsort(predictions, axis=1)[:, -top_k:]
        
        # Convert indices to 3-digit numbers (100-999)
        # Ensure the numbers are within valid 3-digit range
        predicted_numbers = np.clip(indices + 100, 100, 999)
        
        print("Prediction generation finished.")
        return predicted_numbers
    except Exception as e:
        print(f"An error occurred during number prediction: {str(e)}")
        sys.exit(1)

# --- Output Printing ---
def print_predicted_numbers(predicted_numbers):
    """Prints the predicted lottery numbers."""
    try:
        print("-" * 60)
        print("Predicted Numbers (Top 5 choices based on model output):")

        if predicted_numbers.size > 0:
            # Handle both 1D and 2D arrays safely
            if predicted_numbers.ndim == 1:
                # If 1D array, just print the first few predictions
                print(', '.join(map(str, predicted_numbers[:5])))
            else:
                # If 2D array, print the first row
                print(', '.join(map(str, predicted_numbers[0])))
        else:
            print("No predictions were generated or available to display.")

        print("=" * 60)
    except Exception as e:
        print(f"An error occurred while printing the predictions: {str(e)}")
        sys.exit(1)

# --- Main Execution Block ---
def main():
    """Main function to orchestrate the loading, training, and prediction process."""
    try:
        print_intro()

        print("Loading and preparing data...")
        train_data, val_data, max_value = load_data()
        print(f"Data loaded. Max lottery number found: {max_value}")
        print(f"Training set size: {train_data.shape[0]}, Validation set size: {val_data.shape[0]}")

        # Ensure data has at least 2 dimensions
        if train_data.ndim < 2:
            raise ValueError("Training data must have at least 2 dimensions (samples, features). Check 'data.txt' format.")
        
        num_features = train_data.shape[1]
        print(f"Detected {num_features} numbers per draw.")

        print("Creating the neural network model...")
        model = create_model(max_value)
        print("Model created successfully.")

        _ = train_model(model, train_data, val_data)

        print("Using the last sequence from validation data as input for prediction demonstration.")
        prediction_input = val_data[-1:]
        predicted_numbers = predict_numbers(model, prediction_input)

        print_predicted_numbers(predicted_numbers)
        print("LotteryAi finished.")

    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Fatal Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"A fatal error occurred in the main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
