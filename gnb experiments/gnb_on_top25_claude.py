# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# Import the dataset loading function
from dataset_for_nb import load_data

# Implementation of Gaussian Naïve Bayes from scratch
class CustomGaussianNB:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.mean = {}
        self.var = {}
        self.n_features = None
        self.epsilon = 1e-9  # To avoid division by zero
        
    def fit(self, X, y):
        """
        Fit the Gaussian Naïve Bayes classifier to the training data.
        
        Parameters:
        - X: Training features (numpy array or pandas DataFrame)
        - y: Target labels (numpy array or pandas Series)
        """
        # Convert inputs to numpy arrays if they're pandas objects
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.classes = np.unique(y)
        n_samples, self.n_features = X.shape
        
        # Calculate class priors and feature statistics for each class
        for c in self.classes:
            # Extract samples of class c
            X_c = X[y == c]
            
            # Store class prior probability
            self.class_priors[c] = X_c.shape[0] / n_samples
            
            # Calculate mean and variance for each feature
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + self.epsilon
        
        return self
    
    def _calculate_log_likelihood(self, x, mean, var):
        """
        Calculate the log likelihood of sample x given the class parameters.
        
        Parameters:
        - x: Sample feature vector
        - mean: Mean vector for class
        - var: Variance vector for class
        
        Returns:
        - Log likelihood
        """
        # Log of Gaussian PDF: -0.5 * ((x - mean)² / var + log(2π * var))
        exponent = -0.5 * np.sum(np.square(x - mean) / var)
        log_coefficient = -0.5 * np.sum(np.log(2 * np.pi * var))
        return log_coefficient + exponent
    
    def predict_proba(self, X):
        """
        Calculate class probabilities for each sample in X.
        
        Parameters:
        - X: Samples to predict (numpy array or pandas DataFrame)
        
        Returns:
        - Matrix of class probabilities for each sample
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, len(self.classes)))
        
        for i, x in enumerate(X):
            # Calculate log probabilities for each class
            log_probs = {}
            for c in self.classes:
                # Log prior probability
                log_prior = np.log(self.class_priors[c])
                # Log likelihood
                log_likelihood = self._calculate_log_likelihood(x, self.mean[c], self.var[c])
                # Posterior log probability = prior + likelihood
                log_probs[c] = log_prior + log_likelihood
            
            # Convert log probabilities to actual probabilities
            # Shift values to avoid numerical issues
            max_log_prob = max(log_probs.values())
            exp_probs = {c: np.exp(log_prob - max_log_prob) for c, log_prob in log_probs.items()}
            
            # Normalize probabilities to sum to 1
            total_prob = sum(exp_probs.values())
            normalized_probs = {c: exp_prob / total_prob for c, exp_prob in exp_probs.items()}
            
            # Store in output matrix
            for j, c in enumerate(self.classes):
                probs[i, j] = normalized_probs[c]
        
        return probs
    
    def predict(self, X):
        """
        Predict the class for each sample in X.
        
        Parameters:
        - X: Samples to predict (numpy array or pandas DataFrame)
        
        Returns:
        - Predicted class labels
        """
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        """
        Calculate the accuracy of the model on test data.
        
        Parameters:
        - X: Test features
        - y: True labels
        
        Returns:
        - Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate model performance and display metrics.
    
    Parameters:
    - model: Trained model with predict method
    - X_test: Test features
    - y_test: True labels
    - class_names: List of class names (optional)
    
    Returns:
    - Dictionary with performance metrics
    """
    # Predict classes
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get classification report
    if class_names is not None:
        report = classification_report(y_test, y_pred, target_names=class_names)
    else:
        report = classification_report(y_test, y_pred)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Prediction time: {prediction_time:.4f} seconds")
    print("\nClassification Report:")
    print(report)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'prediction_time': prediction_time
    }

# Function to visualize confusion matrix
def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix as a heatmap.
    
    Parameters:
    - cm: Confusion matrix
    - class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# Function to get probabilities for a single sample
def get_genre_probabilities(model, sample, class_names):
    """
    Get probability predictions for a music sample.
    
    Parameters:
    - model: Trained model with predict_proba method
    - sample: Feature vector
    - class_names: List of class names
    
    Returns:
    - Dictionary of genre probabilities
    """
    # Ensure sample is in correct format
    if isinstance(sample, pd.DataFrame):
        X = sample
    else:
        X = pd.DataFrame([sample])
    
    # Get probability predictions
    probabilities = model.predict_proba(X)[0]
    
    # Create a dictionary with genre names and their probabilities
    genre_probs = {genre: prob for genre, prob in zip(class_names, probabilities)}
    
    # Sort by probability in descending order
    genre_probs = {k: v for k, v in sorted(genre_probs.items(), key=lambda item: item[1], reverse=True)}
    
    return genre_probs

# Define the top 25 features
def get_top_features():
    """
    Return the list of top 25 features and their feature numbers.
    
    Returns:
    - feature_names: List of descriptive feature names
    - feature_indices: List of feature indices (0-based)
    """
    top_25_features = [
        'mfcc6_var', 'chroma_stft_var', 'mfcc15_mean', 'mfcc12_mean', 'mfcc7_var',
        'spectral_centroid_var', 'mfcc13_mean', 'mfcc8_mean', 'rolloff_var', 'mfcc3_mean',
        'rms_var', 'mfcc5_mean', 'perceptr_var', 'mfcc9_mean', 'zero_crossing_rate_mean',
        'mfcc7_mean', 'mfcc6_mean', 'mfcc2_mean', 'mfcc4_mean', 'rms_mean',
        'mfcc1_mean', 'spectral_centroid_mean', 'chroma_stft_mean', 'spectral_bandwidth_mean', 'rolloff_mean'
    ]
    
    # Feature numbers as provided (1-indexed to match the feature names)
    feature_indices = [
        29-1, 2-1, 46-1, 40-1, 31-1,
        6-1, 42-1, 32-1, 10-1, 22-1, 
        4-1, 26-1, 16-1, 34-1, 11-1,
        30-1, 28-1, 20-1, 24-1, 3-1, 
        18-1, 5-1, 1-1, 7-1, 9-1
    ]
    
    return top_25_features, feature_indices

# Main function to run the entire process
def main():
    # Load data using provided function
    print("Loading and preprocessing data...")
    X_train_full, X_test_full, y_train, y_test = load_data()
    
    # Get top 25 features
    feature_names, feature_indices = get_top_features()
    
    # Filter dataset to include only top 25 features
    if isinstance(X_train_full, pd.DataFrame):
        # If DataFrame, filter by column indices
        X_train = X_train_full.iloc[:, feature_indices]
        X_test = X_test_full.iloc[:, feature_indices]
        # Rename columns for better readability
        X_train.columns = feature_names
        X_test.columns = feature_names
    else:
        # If numpy array, filter by column indices
        X_train = X_train_full[:, feature_indices]
        X_test = X_test_full[:, feature_indices]
    
    # Check the dataset
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Using top 25 features: {feature_names}")
    
    # Map numeric labels back to genre names for better visualization
    genre_mapping = {
        0: 'blues',
        1: 'classical',
        2: 'country',
        3: 'disco',
        4: 'hiphop',
        5: 'jazz',
        6: 'metal',
        7: 'pop',
        8: 'reggae',
        9: 'rock'
    }
    
    class_names = [genre_mapping[i] for i in range(10)]
    
    # 1. Train and evaluate scikit-learn's GaussianNB
    print("\n--- Scikit-learn Gaussian Naïve Bayes (Top 25 Features) ---")
    sklearn_gnb = GaussianNB()
    start_time = time.time()
    sklearn_gnb.fit(X_train, y_train)
    sklearn_training_time = time.time() - start_time
    print(f"Training time: {sklearn_training_time:.4f} seconds")
    
    sklearn_results = evaluate_model(sklearn_gnb, X_test, y_test, class_names)
    
    # 2. Train and evaluate our custom GaussianNB
    print("\n--- Custom Gaussian Naïve Bayes Implementation (Top 25 Features) ---")
    custom_gnb = CustomGaussianNB()
    start_time = time.time()
    custom_gnb.fit(X_train, y_train)
    custom_training_time = time.time() - start_time
    print(f"Training time: {custom_training_time:.4f} seconds")
    
    custom_results = evaluate_model(custom_gnb, X_test, y_test, class_names)
    
    # Plot confusion matrix for scikit-learn model
    print("\nConfusion Matrix for Scikit-learn GaussianNB:")
    plot_confusion_matrix(sklearn_results['confusion_matrix'], class_names)
    
    # Plot confusion matrix for custom model
    print("\nConfusion Matrix for Custom GaussianNB:")
    plot_confusion_matrix(custom_results['confusion_matrix'], class_names)
    
    # Example: Get probabilities for a test sample
    sample_idx = 0
    sample = X_test.iloc[sample_idx:sample_idx+1] if isinstance(X_test, pd.DataFrame) else X_test[sample_idx:sample_idx+1]
    true_genre = genre_mapping[y_test.iloc[sample_idx]] if isinstance(y_test, pd.Series) else genre_mapping[y_test[sample_idx]]
    
    print(f"\nExample: Predicting genre probabilities for test sample (true genre: {true_genre})")
    
    # Get probabilities from scikit-learn model
    sklearn_probs = get_genre_probabilities(sklearn_gnb, sample, class_names)
    print("\nSciki-learn GaussianNB Probabilities:")
    for genre, prob in sklearn_probs.items():
        print(f"{genre}: {prob:.4f}")
    
    # Get probabilities from custom model
    custom_probs = get_genre_probabilities(custom_gnb, sample, class_names)
    print("\nCustom GaussianNB Probabilities:")
    for genre, prob in custom_probs.items():
        print(f"{genre}: {prob:.4f}")
    
    return sklearn_gnb, custom_gnb, class_names, X_test, y_test

# Function to run a comparison between multiple test samples
def compare_predictions(sklearn_model, custom_model, X_test, y_test, class_names, n_samples=5):
    """
    Compare predictions between sklearn and custom GNB models.
    
    Parameters:
    - sklearn_model: Trained sklearn GaussianNB model
    - custom_model: Trained custom GaussianNB model
    - X_test: Test features
    - y_test: Test labels
    - class_names: List of class names
    - n_samples: Number of samples to compare
    """
    # Create genre mapping
    genre_mapping = {i: name for i, name in enumerate(class_names)}
    
    print(f"\nComparing predictions for {n_samples} test samples:")
    print("-" * 60)
    
    for i in range(min(n_samples, len(X_test))):
        if isinstance(X_test, pd.DataFrame):
            sample = X_test.iloc[i:i+1]
            true_genre = genre_mapping[y_test.iloc[i]]
        else:
            sample = X_test[i:i+1]
            true_genre = genre_mapping[y_test[i]]
        
        # Get predictions
        sklearn_pred = genre_mapping[sklearn_model.predict(sample)[0]]
        custom_pred = genre_mapping[custom_model.predict(sample)[0]]
        
        # Get top probability from each model
        sklearn_probs = get_genre_probabilities(sklearn_model, sample, class_names)
        custom_probs = get_genre_probabilities(custom_model, sample, class_names)
        
        sklearn_top_prob = list(sklearn_probs.items())[0]
        custom_top_prob = list(custom_probs.items())[0]
        
        print(f"Sample {i+1} (True genre: {true_genre})")
        print(f"  Sklearn GNB: {sklearn_pred} (confidence: {sklearn_top_prob[1]:.4f})")
        print(f"  Custom GNB:  {custom_pred} (confidence: {custom_top_prob[1]:.4f})")
        print("-" * 60)

# If running as a script
if __name__ == "__main__":
    sklearn_model, custom_model, class_names, X_test, y_test = main()
    
    # Compare predictions between the two models
    compare_predictions(sklearn_model, custom_model, X_test, y_test, class_names)