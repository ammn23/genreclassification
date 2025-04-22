import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from dataset_for_mlp import load_data
from gnb_complex_64_final import EnhancedGaussianNB, preprocess_data
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler

# epoch change

# Activation functions and their derivatives
class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.power(np.tanh(x), 2)

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        sig = Activations.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def softmax(x):
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class MLP:
    def __init__(self, layer_sizes, activations, random_state=42):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1  # Number of weight matrices needed
        self.activations = activations

        # Activation function mapping
        self.activation_funcs = {
            'relu': (Activations.relu, Activations.relu_derivative),
            'tanh': (Activations.tanh, Activations.tanh_derivative),
            'sigmoid': (Activations.sigmoid, Activations.sigmoid_derivative),
            'softmax': (Activations.softmax, None)  # Softmax derivative is handled specially in backprop
        }

        # Initialize random number generator
        self.rng = np.random.RandomState(random_state)

        # Initialize weights and biases with He initialization
        self.weights = []
        self.biases = []

        for i in range(self.n_layers):
            # He initialization for weights: sqrt(2/n_in)
            scale = np.sqrt(2 / layer_sizes[i])
            self.weights.append(self.rng.normal(0, scale, size=(layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def forward(self, X):
        activations = [X]  # Store activations for each layer, starting with input
        pre_activations = []  # Store pre-activation values for backpropagation

        # Process all layers except the output layer
        for i in range(self.n_layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)

            # Apply activation function
            activation_func = self.activation_funcs[self.activations[i]][0]
            a = activation_func(z)
            activations.append(a)

        # Process output layer separately for softmax
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        pre_activations.append(z_out)

        # Apply output activation (softmax for multi-class classification)
        a_out = Activations.softmax(z_out)
        activations.append(a_out)

        return activations, pre_activations

    def backpropagation(self, X, y, learning_rate=0.01, batch_size=32, n_epochs=1000,
                        early_stopping=True, patience=10, validation_data=None,
                        verbose=True):
        n_samples = X.shape[0]
        n_classes = self.layer_sizes[-1]

        # Convert y to one-hot encoding
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1

        # Initialize history
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # Initialize early stopping variables
        best_val_loss = float('inf')
        no_improvement_count = 0
        best_weights = None
        best_biases = None

        # Training loop
        for epoch in range(n_epochs):
            # Shuffle training data
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)
            X_shuffled = X[indices]
            y_onehot_shuffled = y_onehot[indices]

            # Mini-batch gradient descent
            losses = []
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_onehot_shuffled[i:i + batch_size]

                # Forward pass
                activations, pre_activations = self.forward(X_batch)
                y_pred = activations[-1]

                # Compute batch loss (cross-entropy)
                batch_loss = -np.sum(y_batch * np.log(np.clip(y_pred, 1e-10, 1.0))) / X_batch.shape[0]
                losses.append(batch_loss)

                # Backpropagation
                # For output layer, delta is (y_pred - y_true)
                delta = y_pred - y_batch

                # Update weights for output layer
                self.weights[-1] -= learning_rate * np.dot(activations[-2].T, delta) / X_batch.shape[0]
                self.biases[-1] -= learning_rate * np.mean(delta, axis=0, keepdims=True)

                # Backpropagate error through hidden layers
                for j in range(self.n_layers - 2, -1, -1):
                    # Compute delta for current layer
                    delta = np.dot(delta, self.weights[j + 1].T)

                    # Apply derivative of activation function
                    activation_derivative = self.activation_funcs[self.activations[j]][1]
                    delta *= activation_derivative(pre_activations[j])

                    # Update weights and biases
                    self.weights[j] -= learning_rate * np.dot(activations[j].T, delta) / X_batch.shape[0]
                    self.biases[j] -= learning_rate * np.mean(delta, axis=0, keepdims=True)

            # Compute epoch metrics
            activations, _ = self.forward(X)
            y_pred = activations[-1]
            train_loss = -np.sum(y_onehot * np.log(np.clip(y_pred, 1e-10, 1.0))) / n_samples
            train_accuracy = np.mean(np.argmax(y_pred, axis=1) == y)

            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)

            # Compute validation metrics if validation data is provided
            if validation_data is not None:
                X_val, y_val = validation_data
                n_val_samples = X_val.shape[0]

                # Convert validation y to one-hot encoding
                y_val_onehot = np.zeros((n_val_samples, n_classes))
                y_val_onehot[np.arange(n_val_samples), y_val] = 1

                # Forward pass on validation data
                val_activations, _ = self.forward(X_val)
                val_y_pred = val_activations[-1]

                # Compute validation loss and accuracy
                val_loss = -np.sum(y_val_onehot * np.log(np.clip(val_y_pred, 1e-10, 1.0))) / n_val_samples
                val_accuracy = np.mean(np.argmax(val_y_pred, axis=1) == y_val)

                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                # Early stopping check
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improvement_count = 0
                        # Save best weights
                        best_weights = [w.copy() for w in self.weights]
                        best_biases = [b.copy() for b in self.biases]
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        # Restore best weights
                        self.weights = best_weights
                        self.biases = best_biases
                        break

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                status = f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
                if validation_data is not None:
                    status += f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
                print(status)

        return history

    def predict_proba(self, X):
        """Predict class probabilities"""
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


# Function to create and preprocess data for the enhanced MLP
def prepare_data_with_gnb(X_train_orig, X_test_orig, y_train, y_test,
                          apply_feature_selection=False, n_features=40,
                          apply_transformation=True, epsilon=1e-3,
                          class_weight_adjustment=1.2):

    #  Feature Selection (to try)
    if apply_feature_selection:
        print(f"Applying SelectKBest to find top {n_features} features...")
        k = min(n_features, X_train_orig.shape[1])
        feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_train = feature_selector.fit_transform(X_train_orig, y_train)
        X_test = feature_selector.transform(X_test_orig)
        print(f"Selected {X_train.shape[1]} features based on mutual information.")
    else:
        X_train = X_train_orig.copy()
        X_test = X_test_orig.copy()

    # --- Apply Preprocessing ---
    print("Preprocessing data...")
    X_train_basic, X_test_basic = preprocess_data(
        X_train, X_test,
        apply_scaling=True,
        apply_transformation=apply_transformation,
        transformation_method='quantile'
    )

    # --- Train GNB for enhanced features ---
    print(f"Training GNB for enhanced features...")
    gnb = EnhancedGaussianNB(epsilon=epsilon, class_weight_adjustment=class_weight_adjustment)
    gnb.fit(X_train_basic, y_train)

    # --- Get probability predictions from GNB ---
    gnb_train_proba = gnb.predict_proba(X_train_basic)
    gnb_test_proba = gnb.predict_proba(X_test_basic)

    # --- Concatenate original features with GNB probabilities ---
    X_train_enhanced = np.hstack((X_train_basic, gnb_train_proba))
    X_test_enhanced = np.hstack((X_test_basic, gnb_test_proba))

    print(f"Original data shape: {X_train_basic.shape}")
    print(f"Enhanced data shape (with GNB probas): {X_train_enhanced.shape}")

    return X_train_basic, X_test_basic, X_train_enhanced, X_test_enhanced

def apply_kernel_mapping(X_train,X_test,gamma=0.1,n_components=100):
    rbf_feature = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
    X_train_mapped = rbf_feature.fit_transform(X_train)
    X_test_mapped = rbf_feature.transform(X_test)
    return X_train_mapped, X_test_mapped

def evaluate_model(model, X_test, y_test, model_name="Model", class_names=None):
    """
    Evaluate model performance

    Parameters:
    -----------
    model : MLP
        Trained MLP model
    X_test : array-like
        Test data
    y_test : array-like
        True labels
    model_name : str
        Name of the model for display
    class_names : list
        List of class names for visualization

    Returns:
    --------
    results : dict
        Dictionary with evaluation results
    """
    print(f"\n--- Evaluating {model_name} ---")
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Prediction time: {prediction_time:.4f} seconds")
    print("\nClassification Report:")

    if class_names is not None:
        report = classification_report(y_test, y_pred,
                                       target_names=class_names,
                                       labels=np.arange(len(class_names)))
    else:
        report = classification_report(y_test, y_pred)

    print(report)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'prediction_time': prediction_time,
        'y_pred': y_pred
    }


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_training_history(history, title='Training History'):
    """Plot training history"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history and history['val_accuracy']:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    # 1. Load data
    print("Loading data...")
    X_train_full, X_test_full, y_train, y_test = load_data()

    # 2. Define genre mapping
    genre_mapping = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
                     5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
    class_names = [genre_mapping[i] for i in range(len(genre_mapping))]

    # 3. Prepare data for both baseline MLP and enhanced MLP
    X_train_basic, X_test_basic, X_train_enhanced, X_test_enhanced = prepare_data_with_gnb(
        X_train_full, X_test_full, y_train, y_test,
        apply_feature_selection=True,
        n_features=40
    )

    # Create a small validation set for early stopping
    X_train_basic, X_val_basic, y_train_split, y_val = train_test_split(
        X_train_basic, y_train, test_size=0.15, random_state=42)

    X_train_enhanced, X_val_enhanced = train_test_split(
        X_train_enhanced, test_size=0.15, random_state=42)[0:2]

    n_features_basic = X_train_basic.shape[1]
    n_features_enhanced = X_train_enhanced.shape[1]
    n_classes = len(genre_mapping)

    # 4. Define network architecture for baseline MLP
    # We'll use ReLU for hidden layers and Softmax for output layer
    # The baseline MLP takes only the original features
    baseline_layers = [n_features_basic, 128, 64, n_classes]
    baseline_activations = ['relu', 'relu', 'softmax']

    print("\n--- Training Baseline MLP ---")
    baseline_mlp = MLP(baseline_layers, baseline_activations)

    baseline_history = baseline_mlp.backpropagation(
        X_train_basic, y_train_split,
        learning_rate=0.001,
        batch_size=32,
        early_stopping=True,
        patience=10,
        validation_data=(X_val_basic, y_val),
        verbose=True
    )

    # 5. Define network architecture for enhanced MLP
    # The enhanced MLP takes original features + GNB probabilities
    enhanced_layers = [n_features_enhanced, 128, 64, n_classes]
    enhanced_activations = ['relu', 'relu', 'softmax']

    print("\n--- Training Enhanced MLP (with GNB probabilities) ---")
    enhanced_mlp = MLP(enhanced_layers, enhanced_activations)

    enhanced_history = enhanced_mlp.backpropagation(
        X_train_enhanced, y_train_split,
        learning_rate=0.001,
        batch_size=32,
        early_stopping=True,
        patience=10,
        validation_data=(X_val_enhanced, y_val),
        verbose=True
    )

    # 6. Evaluate models
    baseline_results = evaluate_model(
        baseline_mlp, X_test_basic, y_test,
        model_name="Baseline MLP",
        class_names=class_names
    )

    enhanced_results = evaluate_model(
        enhanced_mlp, X_test_enhanced, y_test,
        model_name="Enhanced MLP (with GNB probabilities)",
        class_names=class_names
    )

    print("\n--- Training scikit-learn MLPClassifier(basic) ---")

    # Initialize sklearn MLP (comparable architecture)
    sklearn_mlp_basic = MLPClassifier(hidden_layer_sizes=(128, 64),
                                activation='relu',
                                solver='adam',
                                max_iter=300,
                                early_stopping=True,
                                random_state=42,
                                verbose=True)

    # Fit on the same enhanced features for fair comparison
    sklearn_mlp_basic.fit(X_train_basic, y_train_split)

    # Predict and evaluate
    sklearn_y_pred_basic = sklearn_mlp_basic.predict(X_test_basic)
    sklearn_accuracy_basic = accuracy_score(y_test, sklearn_y_pred_basic)
    sklearn_cm_basic = confusion_matrix(y_test, sklearn_y_pred_basic)
    sklearn_report_basic = classification_report(y_test, sklearn_y_pred_basic, target_names=class_names)

    print(f"Accuracy: {sklearn_accuracy_basic:.4f}")
    print("\nClassification Report:")
    print(sklearn_report_basic)

    plot_confusion_matrix(sklearn_cm_basic, class_names, title='Scikit-learn MLP Confusion Matrix(basic)')

    # 10. Train Kernel-Enhanced MLP (using RBF features)
    print("\n--- Training Kernel-Enhanced MLP (with GNB + RBF kernel) ---")

    # Apply kernel transformation on enhanced features
    X_train_kernel, X_test_kernel = apply_kernel_mapping(X_train_enhanced, X_test_enhanced,
                                                         gamma=0.01, n_components=100)

    # Split for validation
    X_train_kernel, X_val_kernel = train_test_split(
        X_train_kernel, test_size=0.15, random_state=42)

    kernel_layers = [X_train_kernel.shape[1], 64, n_classes]
    kernel_activations = ['relu', 'softmax']

    kernel_mlp = MLP(kernel_layers, kernel_activations)

    kernel_history = kernel_mlp.backpropagation(
        X_train_kernel, y_train_split,
        learning_rate=0.0005,
        batch_size=32,
        early_stopping=True,
        patience=15,
        validation_data=(X_val_kernel, y_val),
        verbose=True
    )

    kernel_results = evaluate_model(
        kernel_mlp, X_test_kernel, y_test,
        model_name="Kernel-Enhanced MLP",
        class_names=class_names
    )

    plot_training_history(kernel_history, title='Kernel-Enhanced MLP Training History')
    plot_confusion_matrix(kernel_results['confusion_matrix'], class_names, title='Kernel-Enhanced MLP Confusion Matrix')


    print("\n--- Training scikit-learn MLPClassifier(enhanced) ---")

    # Initialize sklearn MLP (comparable architecture)
    sklearn_mlp = MLPClassifier(hidden_layer_sizes=(128, 64),
                                activation='relu',
                                solver='adam',
                                max_iter=300,
                                early_stopping=True,
                                random_state=42,
                                verbose=True)

    # Fit on the same enhanced features for fair comparison
    sklearn_mlp.fit(X_train_enhanced, y_train_split)

    # Predict and evaluate
    sklearn_y_pred = sklearn_mlp.predict(X_test_enhanced)
    sklearn_accuracy = accuracy_score(y_test, sklearn_y_pred)
    sklearn_cm = confusion_matrix(y_test, sklearn_y_pred)
    sklearn_report = classification_report(y_test, sklearn_y_pred, target_names=class_names)

    print(f"Accuracy: {sklearn_accuracy:.4f}")
    print("\nClassification Report:")
    print(sklearn_report)

    plot_confusion_matrix(sklearn_cm, class_names, title='Scikit-learn MLP Confusion Matrix')

    # 7. Plot results
    plot_training_history(baseline_history, title='Baseline MLP Training History')
    plot_training_history(enhanced_history, title='Enhanced MLP Training History')

    plot_confusion_matrix(
        baseline_results['confusion_matrix'],
        class_names,
        title='Baseline MLP Confusion Matrix'
    )

    plot_confusion_matrix(
        enhanced_results['confusion_matrix'],
        class_names,
        title='Enhanced MLP Confusion Matrix'
    )

    # 8. Compare results
    print("\n--- Model Comparison ---")
    print(f"Baseline MLP Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"Enhanced MLP Accuracy: {enhanced_results['accuracy']:.4f}")
    print(f"Kernel-Enhanced MLP Accuracy: {kernel_results['accuracy']:.4f}")
    print(f"Scikit-learn MLP Accuracy(basic): {sklearn_accuracy_basic:.4f}")
    print(f"Scikit-learn MLP Accuracy(enhanced): {sklearn_accuracy:.4f}")
    print(f"Improvement: {enhanced_results['accuracy'] - baseline_results['accuracy']:.4f}")

    return {
        'baseline_mlp': baseline_mlp,
        'enhanced_mlp': enhanced_mlp,
        'baseline_results': baseline_results,
        'enhanced_results': enhanced_results,
        'baseline_history': baseline_history,
        'enhanced_history': enhanced_history,
        'sklearn_mlp': sklearn_mlp,
        'sklearn_accuracy': sklearn_accuracy,
        'sklearn_report': sklearn_report
    }


if __name__ == "__main__":
    results = main()

