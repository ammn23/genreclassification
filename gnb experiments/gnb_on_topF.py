import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # Load the CSV file with header
    df = pd.read_csv("C:\\Users\\User\\Desktop\\Spring 2025\\ml\\genreclassification\\dataset\\features_3_sec.csv")
    
    # Drop the filename column
    if 'filename' in df.columns:
        df = df.drop(columns=['filename'])
    else:
        first_col = df.columns[0]
        df = df.drop(columns=[first_col])
    
    # The last column should be the genre label
    last_col = df.columns[-1]
    print(f"Last column (expected to be genre): {last_col}")
    
    # Convert genres to numeric labels
    converter = LabelEncoder()
    y = converter.fit_transform(df[last_col])
    
    # Get feature columns (all except the genre column)
    X = df.drop(columns=[last_col])
    
    # Scale the features
    min_max_scaler = MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled, columns=X.columns)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

class GaussianNaiveBayes:
    def __init__(self, top_features=None):
        self.classes = None
        self.class_priors = None
        self.means = None
        self.variances = None
        self.n_features = None
        self.n_classes = None
        self.genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                             'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.top_features = top_features
    
    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes model using only the top features
        
        Parameters:
        -----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
            Training data
        y : Series or ndarray of shape (n_samples,)
            Target values (class labels)
        """
        # Filter data to only use the top features
        if self.top_features is not None:
            X = X[self.top_features]
        
        X = np.array(X)
        y = np.array(y)
        
        # Get unique classes and count
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        
        # Calculate class priors (probability of each class)
        self.class_priors = np.zeros(self.n_classes)
        self.means = np.zeros((self.n_classes, self.n_features))
        self.variances = np.zeros((self.n_classes, self.n_features))
        
        # For each class
        for i, c in enumerate(self.classes):
            # Get samples of this class
            X_c = X[y == c]
            
            # Calculate prior probability
            self.class_priors[i] = X_c.shape[0] / X.shape[0]
            
            # Calculate mean and variance for each feature
            self.means[i, :] = X_c.mean(axis=0)
            self.variances[i, :] = X_c.var(axis=0) + 1e-9  # Add epsilon to variance to prevent division by zero
        
        return self
    
    def _calculate_likelihood(self, x, mean, var):
        """Calculate Gaussian probability density function"""
        exponent = np.exp(-0.5 * ((x - mean) ** 2) / var)
        return (1 / np.sqrt(2 * np.pi * var)) * exponent
    
    def _calculate_class_probability(self, x):
        """Calculate posterior probability for each class"""
        posteriors = []
        
        # For each class
        for i in range(self.n_classes):
            # Start with the prior
            posterior = np.log(self.class_priors[i])
            
            # Add the log-likelihood for each feature
            for j in range(self.n_features):
                likelihood = self._calculate_likelihood(x[j], self.means[i, j], self.variances[i, j])
                posterior += np.log(likelihood + 1e-10)
            
            posteriors.append(posterior)
        
        # Convert log probabilities back to probabilities
        log_prob_sum = np.sum(posteriors)
        posteriors = np.exp(posteriors - log_prob_sum)
        
        return posteriors / np.sum(posteriors)
    
    def predict(self, X):
        """
        Predict class labels for samples in X using the trained model
        
        Parameters:
        -----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        ndarray of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Parameters:
        -----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        ndarray of shape (n_samples, n_classes)
            Class probabilities for each sample
        """
        X = np.array(X)
        proba = np.zeros((X.shape[0], self.n_classes))
        
        for i, x in enumerate(X):
            proba[i] = self._calculate_class_probability(x)
            
        return proba
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix as heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.genre_labels, 
                    yticklabels=self.genre_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        X_test : DataFrame or ndarray of shape (n_samples, n_features)
            Test samples
        y_test : Series or ndarray of shape (n_samples,)
            True class labels for test samples
            
        Returns:
        --------
        dict
            Dictionary with accuracy and classification report
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.genre_labels)
        
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }

# Main execution code
if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Use the correct top 25 features as per the new image
    top_25_features = [
        'mfcc6_var', 'chroma_stft_var', 'mfcc15_mean', 'mfcc12_mean', 'mfcc7_var',
        'spectral_centroid_var', 'mfcc13_mean', 'mfcc8_mean', 'rolloff_var', 'mfcc3_mean',
        'rms_var', 'mfcc5_mean', 'perceptr_var', 'mfcc9_mean', 'zero_crossing_rate_mean',
        'mfcc7_mean', 'mfcc6_mean', 'mfcc2_mean', 'mfcc4_mean', 'rms_mean', 
        'mfcc1_mean', 'spectral_centroid_mean', 'chroma_stft_mean', 'spectral_bandwidth_mean', 'rolloff_mean'
    ]
    
    # Initialize and train the model with top 25 features
    gnb = GaussianNaiveBayes(top_features=top_25_features)
    gnb.fit(X_train, y_train)
    
    # Evaluate model
    results = gnb.evaluate(X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot confusion matrix
    gnb.plot_confusion_matrix(results['confusion_matrix'])
