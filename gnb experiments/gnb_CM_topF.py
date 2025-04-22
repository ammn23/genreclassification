import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_for_nb import load_data

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.means = None
        self.variances = None
        self.n_features = None
        self.n_classes = None
        self.genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                            'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes model
        
        Parameters:
        -----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
            Training data
        y : Series or ndarray of shape (n_samples,)
            Target values (class labels)
        """
        # Convert to numpy arrays if they're not already
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
            # Add small epsilon to variance to prevent division by zero
            self.variances[i, :] = X_c.var(axis=0) + 1e-9
        
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
                # Use log to avoid numerical underflow
                # Add small epsilon to prevent log(0)
                posterior += np.log(likelihood + 1e-10)
            
            posteriors.append(posterior)
        
        # Convert log probabilities back to probabilities
        log_prob_sum = np.sum(posteriors)
        posteriors = np.exp(posteriors - log_prob_sum)
        
        # Normalize to ensure they sum to 1
        return posteriors / np.sum(posteriors)
    
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
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
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
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
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
    
    def plot_feature_importance(self):
        """Plot feature importance based on variance difference between classes"""
        feature_names = [f'Feature {i}' for i in range(self.n_features)]
        
        # Calculate feature importance as the variance of class means
        importance = np.var(self.means, axis=0)
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        top_n = 25  # Show top 20 features
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importance[indices][:top_n], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices][:top_n])
        plt.xlabel('Importance (Variance of Means)')
        plt.title('Top Features by Importance')
        plt.tight_layout()
        plt.show()
        # """Plot feature importance based on variance difference between classes"""
        # # Use the actual column names from the dataset
        # feature_names = X_train.columns.tolist()  # Assuming you are working with pandas DataFrame for X_train
        
        # # Calculate feature importance as the variance of class means
        # importance = np.var(self.means, axis=0)
        
        # # Sort features by importance
        # indices = np.argsort(importance)[::-1]
        # top_n = 25  # Show top 20 features
        
        # plt.figure(figsize=(12, 8))
        # plt.barh(range(top_n), importance[indices][:top_n], align='center')
        # plt.yticks(range(top_n), [feature_names[i] for i in indices][:top_n])  # Use feature names
        # plt.xlabel('Importance (Variance of Means)')
        # plt.title('Top Features by Importance')
        # plt.tight_layout()
        # plt.show()



# Main execution code
if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Initialize and train the model
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gnb.predict(X_test)
    
    # Get probabilities for each genre
    proba = gnb.predict_proba(X_test)
    
    # Print sample probabilities for first test example
    print("\nProbabilities for first test example:")
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    for i, genre in enumerate(genres):
        print(f"{genre}: {proba[0][i]:.4f}")
    
    # Evaluate model
    results = gnb.evaluate(X_test, y_test)
    
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot confusion matrix
    gnb.plot_confusion_matrix(results['confusion_matrix'])
    
    # Plot feature importance
    gnb.plot_feature_importance()
    
    # Test with specific examples
    def predict_genre(gnb, X_example):
        """Predict genre for a specific example and print probabilities"""
        proba = gnb.predict_proba(X_example.reshape(1, -1))[0]
        predicted_class = np.argmax(proba)
        
        print("\nPredicted Genre Probabilities:")
        for i, genre in enumerate(genres):
            print(f"{genre}: {proba[i]:.4f}" + (" ← PREDICTED" if i == predicted_class else ""))
    
    # Example: predict for the first test example
    example = X_test.iloc[0].values
    predict_genre(gnb, example)