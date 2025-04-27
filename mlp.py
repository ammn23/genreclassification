from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, \
    f1_score, roc_auc_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import SVC
import time
import seaborn as sns
from sklearn.preprocessing import StandardScaler, label_binarize

from dataset_for_mlp import load_data
from gnb_simple_52_final import CustomGaussianNB
from gnb_complex_64_final import preprocess_data
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier



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

        # Check dimensions
        if len(activations) != self.n_layers:
            raise ValueError(f"Got {len(activations)} activation functions for {self.n_layers} layers")

        # Activation function mapping
        self.activation_funcs = {
            'relu': (Activations.relu, Activations.relu_derivative),
            'tanh': (Activations.tanh, Activations.tanh_derivative),
            'sigmoid': (Activations.sigmoid, Activations.sigmoid_derivative),
            'softmax': (Activations.softmax, None)  # Softmax derivative is handled specially
        }

        # Initialize random number generator
        self.rng = np.random.RandomState(random_state)

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(self.n_layers):
            # He initialization for ReLU: sqrt(2/n_in)
            if self.activations[i] == 'relu':
                scale = np.sqrt(2 / layer_sizes[i])
            else:
                scale = np.sqrt(1 / layer_sizes[i])

            self.weights.append(self.rng.normal(0, scale, size=(layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def forward(self, X):
        activations = [X]  # Start with input layer
        pre_activations = []

        # Process all layers
        for i in range(self.n_layers):
            # Linear transformation: z = X*W + b
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)

            # Apply activation function
            if i == self.n_layers - 1 and self.activations[i] == 'softmax':
                # Special case for softmax output layer
                a = Activations.softmax(z)
            else:
                activation_func = self.activation_funcs[self.activations[i]][0]
                a = activation_func(z)

            activations.append(a)

        return activations, pre_activations

    def backpropagation(self, X, y, learning_rate=0.01, batch_size=32, n_epochs=1000,
                        early_stopping=True, patience=10, validation_data=None,
                        verbose=True):
        n_samples = X.shape[0]
        n_classes = self.layer_sizes[-1]

        if len(y) != n_samples:
            raise ValueError(f"X has {n_samples} samples but y has {len(y)} samples")

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

        # Early stopping variables
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
                batch_size_actual = X_batch.shape[0]  # May be smaller than batch_size at the end

                # Forward pass
                activations, pre_activations = self.forward(X_batch)
                y_pred = activations[-1]

                # Compute batch loss (cross-entropy)
                epsilon = 1e-10  # For numerical stability
                batch_loss = -np.sum(y_batch * np.log(np.clip(y_pred, epsilon, 1.0))) / batch_size_actual
                losses.append(batch_loss)

                # Backward pass
                # Initialize gradients
                dW = [np.zeros_like(w) for w in self.weights]
                db = [np.zeros_like(b) for b in self.biases]

                # Output layer delta: for softmax + cross-entropy, delta = (y_pred - y_true)
                delta = y_pred - y_batch  # shape: (batch_size, n_classes)

                # Backpropagate through the network
                for l in range(self.n_layers - 1, -1, -1):
                    # Compute gradients for this layer
                    dW[l] = np.dot(activations[l].T, delta) / batch_size_actual
                    db[l] = np.sum(delta, axis=0, keepdims=True) / batch_size_actual

                    # Backpropagate delta to previous layer (if not the input layer)
                    if l > 0:
                        delta = np.dot(delta, self.weights[l].T)
                        # Apply activation derivative
                        activation_derivative = self.activation_funcs[self.activations[l - 1]][1]
                        delta *= activation_derivative(pre_activations[l - 1])

                # Update weights and biases using gradients
                for l in range(self.n_layers):
                    self.weights[l] -= learning_rate * dW[l]
                    self.biases[l] -= learning_rate * db[l]

            # Compute epoch metrics
            activations, _ = self.forward(X)
            y_pred = activations[-1]
            train_loss = -np.sum(y_onehot * np.log(np.clip(y_pred, 1e-10, 1.0))) / n_samples
            train_accuracy = np.mean(np.argmax(y_pred, axis=1) == y)

            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)

            # Compute validation metrics
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
        """Predict class probabilities for input X"""
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X):
        """Predict class labels for input X"""
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
    gnb = CustomGaussianNB()
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


def evaluate_model(model, X_test, y_test, model_name="Model", class_names=None):
    print(f"\n--- Evaluating {model_name} ---")
    start_time = time.time()

    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)
    except (AttributeError, NotImplementedError):
        print(f"Warning: {model_name} does not support predict_proba. ROC curves will not be available.")
        y_proba = None

    prediction_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)

    # Per-class precision
    precision_per_class = precision_score(y_test, y_pred, average=None)
    precision_dict = {class_names[i]: precision_per_class[i] for i in range(len(class_names))}

    # F1 score (macro and per-class)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_per_class = f1_score(y_test, y_pred, average=None)
    f1_dict = {class_names[i]: f1_per_class[i] for i in range(len(class_names))}
    cm = confusion_matrix(y_test, y_pred)

    # Compute ROC AUC - needs one-hot encoded y_test
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    try:
        roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
    except ValueError:
        # Handle potential dimension mismatch
        print("Warning: Could not compute ROC AUC. Continuing with other metrics.")
        roc_auc = None

    # Calculate per-class ROC AUC
    roc_auc_per_class = {}
    for i in range(n_classes):
        try:
            roc_auc_per_class[class_names[i]] = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        except (ValueError, IndexError):
            print(f"Warning: Could not compute ROC AUC for class {class_names[i]}.")
            roc_auc_per_class[class_names[i]] = None

    # Get classification report
    if class_names is not None:
        report = classification_report(y_test, y_pred, target_names=class_names)
    else:
        report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Prediction time: {prediction_time:.4f} seconds")
    print("\nClassification Report:")

    if roc_auc is not None:
        print(f"Macro ROC AUC: {roc_auc:.4f}")

    if class_names is not None:
        report = classification_report(y_test, y_pred,
                                       target_names=class_names,
                                       labels=np.arange(len(class_names)))
    else:
        report = classification_report(y_test, y_pred)

    print(report)

    print("\nPrecision for each class:")
    for genre, prec in precision_dict.items():
        print(f"{genre}: {prec:.4f}")

    print("\nF1 Score for each class:")
    for genre, f1 in f1_dict.items():
        print(f"{genre}: {f1:.4f}")

    if roc_auc is not None:
        print("\nROC AUC for each class:")
        for genre, auc_val in roc_auc_per_class.items():
            if auc_val is not None:
                print(f"{genre}: {auc_val:.4f}")

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_per_class': precision_dict,
        'f1_per_class': f1_dict,
        'confusion_matrix': cm,
        'classification_report': report,
        'prediction_time': prediction_time,
        'roc_auc': roc_auc,
        'roc_auc_per_class': roc_auc_per_class,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Plot absolute numbers
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Absolute Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # Plot percentages
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized by True Class)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_test, y_proba, class_names):

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(12, 10))
    colors = cycle(['blue', 'red', 'green', 'navy', 'turquoise',
                    'darkorange', 'cornflowerblue', 'teal', 'purple', 'gold'])

    for i, color, name in zip(range(n_classes), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{name} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Music Genre')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



# Function to get probabilities for a single sample
def get_genre_probabilities(model, sample, class_names):

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


def plot_per_class_metrics(precision_dict, f1_dict, roc_auc_dict=None):

    class_names = list(precision_dict.keys())
    precision_values = list(precision_dict.values())
    f1_values = list(f1_dict.values())

    if roc_auc_dict:
        roc_auc_values = [roc_auc_dict.get(name, 0) for name in class_names]
        width = 0.25  # Width of bars

        plt.figure(figsize=(14, 8))
        x = np.arange(len(class_names))

        plt.bar(x - width, precision_values, width, label='Precision', color='skyblue')
        plt.bar(x, f1_values, width, label='F1 Score', color='lightgreen')
        plt.bar(x + width, roc_auc_values, width, label='ROC AUC', color='salmon')

        plt.xlabel('Genre')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend(loc='lower right')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    else:
        width = 0.35  # Width of bars

        plt.figure(figsize=(14, 8))
        x = np.arange(len(class_names))

        plt.bar(x - width / 2, precision_values, width, label='Precision', color='skyblue')
        plt.bar(x + width / 2, f1_values, width, label='F1 Score', color='lightgreen')

        plt.xlabel('Genre')
        plt.ylabel('Score')
        plt.title('Per-Class Precision and F1 Scores')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend(loc='lower right')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
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
    # Assuming load_data() returns X_train_full, X_test_full, y_train, y_test
    X_train_full, X_test_full, y_train_full, y_test = load_data()  # Use y_train_full initially

    # 2. Define genre mapping
    genre_mapping = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
                     5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
    class_names = [genre_mapping[i] for i in range(len(genre_mapping))]
    n_classes = len(class_names)

    # 3. Prepare data for both baseline MLP and enhanced MLP
    X_train_basic_full, X_test_basic, X_train_enhanced_full, X_test_enhanced = prepare_data_with_gnb(
        X_train_full, X_test_full, y_train_full, y_test,
        apply_feature_selection=True,
        n_features=40,
        apply_transformation=True  # Keep settings from original code
    )

    # 4. Create a consistent training/validation split *once*
    # Use stratified split if classes might be imbalanced
    print("\nSplitting data into training and validation sets...")
    X_train_basic, X_val_basic, y_train, y_val = train_test_split(
        X_train_basic_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full)

    # Apply the *same* split to the enhanced features
    X_train_enhanced, X_val_enhanced, _, _ = train_test_split(
        X_train_enhanced_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full)

    print(f"Shapes after split:")
    print(f"  X_train_basic: {X_train_basic.shape}, y_train: {y_train.shape}")
    print(f"  X_val_basic: {X_val_basic.shape}, y_val: {y_val.shape}")
    print(f"  X_train_enhanced: {X_train_enhanced.shape}")
    print(f"  X_val_enhanced: {X_val_enhanced.shape}")

    n_features_basic = X_train_basic.shape[1]
    n_features_enhanced = X_train_enhanced.shape[1]

    # 5. Define network architecture for baseline MLP
    baseline_layers = [n_features_basic, 128, 64, n_classes]
    baseline_activations = ['relu', 'relu', 'softmax']

    print("\n--- Training Baseline MLP ---")
    baseline_mlp = MLP(baseline_layers, baseline_activations, random_state=42)

    baseline_history = baseline_mlp.backpropagation(
        X_train_basic, y_train,
        learning_rate=0.001,
        batch_size=32,
        early_stopping=True,
        patience=15,  # Increased patience slightly
        validation_data=(X_val_basic, y_val),
        verbose=True
    )

    # 6. Define network architecture for enhanced MLP
    enhanced_layers = [n_features_enhanced, 128, 64, n_classes]
    enhanced_activations = ['relu', 'relu', 'softmax']

    print("\n--- Training Enhanced MLP (with GNB probabilities) ---")
    enhanced_mlp = MLP(enhanced_layers, enhanced_activations, random_state=42)

    enhanced_history = enhanced_mlp.backpropagation(
        X_train_enhanced, y_train,
        learning_rate=0.001,
        batch_size=32,
        early_stopping=True,
        patience=15,  # Increased patience slightly
        validation_data=(X_val_enhanced, y_val),
        verbose=True
    )

    # 7. Evaluate custom models
    baseline_results = evaluate_model(
        baseline_mlp, X_test_basic, y_test,
        model_name="Baseline MLP (Custom)",
        class_names=class_names
    )

    enhanced_results = evaluate_model(
        enhanced_mlp, X_test_enhanced, y_test,
        model_name="Enhanced MLP (Custom, with GNB probabilities)",
        class_names=class_names
    )

    # Plot training history
    plot_training_history(baseline_history, title='Baseline MLP (Custom) Training History')
    plot_training_history(enhanced_history, title='Enhanced MLP (Custom) Training History')

    # Plot per-class metrics for custom models
    plot_per_class_metrics(
        baseline_results['precision_per_class'],
        baseline_results['f1_per_class'],
        baseline_results['roc_auc_per_class']
    )

    plot_per_class_metrics(
        enhanced_results['precision_per_class'],
        enhanced_results['f1_per_class'],
        enhanced_results['roc_auc_per_class']
    )

    # Plot ROC curves for custom models if y_proba is available
    if baseline_results['y_proba'] is not None:
        plot_roc_curves(baseline_results['y_test'], baseline_results['y_proba'], class_names)

    if enhanced_results['y_proba'] is not None:
        plot_roc_curves(enhanced_results['y_test'], enhanced_results['y_proba'], class_names)

    # --- Scikit-learn Models ---

    print("\n--- Training scikit-learn MLPClassifier (basic) ---")
    sklearn_mlp_basic = MLPClassifier(hidden_layer_sizes=(128, 64),
                                      activation='relu',
                                      solver='adam',
                                      alpha=0.0001,  # Default L2
                                      batch_size='auto',
                                      learning_rate_init=0.001,  # Default
                                      max_iter=500,  # Increased max_iter
                                      early_stopping=True,
                                      validation_fraction=0.1,  # Uses 10% of training data for validation
                                      n_iter_no_change=15,  # Similar to patience
                                      random_state=42,
                                      verbose=False)  # Quieter training

    # Fit on the basic training data
    print("Fitting sklearn MLP (basic)...")
    sklearn_mlp_basic.fit(X_train_basic, y_train)
    print("Fitting done.")

    # Evaluate basic sklearn MLP
    sklearn_basic_results = evaluate_model(
        sklearn_mlp_basic, X_test_basic, y_test,
        model_name="Scikit-learn MLP (basic)",
        class_names=class_names
    )

    print("\n--- Training scikit-learn MLPClassifier (enhanced) ---")
    sklearn_mlp_enhanced = MLPClassifier(hidden_layer_sizes=(128, 64),
                                         activation='relu',
                                         solver='adam',
                                         alpha=0.0001,
                                         batch_size='auto',
                                         learning_rate_init=0.001,
                                         max_iter=500,
                                         early_stopping=True,
                                         validation_fraction=0.1,
                                         n_iter_no_change=15,
                                         random_state=42,
                                         verbose=False)

    # Fit on the enhanced training data
    print("Fitting sklearn MLP (enhanced)...")
    sklearn_mlp_enhanced.fit(X_train_enhanced, y_train)
    print("Fitting done.")

    # Evaluate enhanced sklearn MLP
    sklearn_enhanced_results = evaluate_model(
        sklearn_mlp_enhanced, X_test_enhanced, y_test,
        model_name="Scikit-learn MLP (enhanced)",
        class_names=class_names
    )

    # Plot per-class metrics for sklearn models
    plot_per_class_metrics(
        sklearn_basic_results['precision_per_class'],
        sklearn_basic_results['f1_per_class'],
        sklearn_basic_results['roc_auc_per_class']
    )

    plot_per_class_metrics(
        sklearn_enhanced_results['precision_per_class'],
        sklearn_enhanced_results['f1_per_class'],
        sklearn_enhanced_results['roc_auc_per_class']
    )

    # Plot ROC curves for sklearn models if y_proba is available
    if sklearn_basic_results['y_proba'] is not None:
        plot_roc_curves(sklearn_basic_results['y_test'], sklearn_basic_results['y_proba'], class_names)

    if sklearn_enhanced_results['y_proba'] is not None:
        plot_roc_curves(sklearn_enhanced_results['y_test'], sklearn_enhanced_results['y_proba'], class_names)

    # ---Moving to SVM---
    print("\n--- Training scikit-learn MLPClassifier+library SVM ---")

    def relu(X):
        return np.maximum(0, X)

    def extract_mlp_features(X, model):
        # For scikit-learn MLP
        if hasattr(model, 'coefs_'):
            W1, W2 = model.coefs_[0], model.coefs_[1]
            b1, b2 = model.intercepts_[0], model.intercepts_[1]
            h1 = relu(np.dot(X, W1) + b1)
            return relu(np.dot(h1, W2) + b2)
        else:

            activations, _ = model.forward(X)
            return activations[-2]  # Return the second-to-last layer activations


    print("Extracting MLP hidden features...")
    X_train_mlp_features_baslib = extract_mlp_features(X_train_basic, sklearn_mlp_basic)
    X_test_mlp_features_baslib = extract_mlp_features(X_test_basic, sklearn_mlp_basic)

    scaler = StandardScaler()
    X_train_mlp_features_baslib = scaler.fit_transform(X_train_mlp_features_baslib)
    X_test_mlp_features_baslib = scaler.transform(X_test_mlp_features_baslib)


    svm_lib = SVC(kernel='rbf', C=10, gamma='scale', probability=True)  # Added probability=True for ROC curves
    svm_lib.fit(X_train_mlp_features_baslib, y_train)

    print("\n--- SVM on Basic MLP Hidden Features ---")
    svm_results_basic_lib = evaluate_model(
        svm_lib, X_test_mlp_features_baslib, y_test,
        model_name="SVM on Basic MLP Hidden Features",
        class_names=class_names
    )

    print("\n--- Training SVM on Enhanced MLP Features ---")


    X_train_mlp_features_enhlib = extract_mlp_features(X_train_enhanced, sklearn_mlp_enhanced)
    X_test_mlp_features_enhlib = extract_mlp_features(X_test_enhanced, sklearn_mlp_enhanced)

    scaler = StandardScaler()
    X_train_mlp_features_enhlib = scaler.fit_transform(X_train_mlp_features_enhlib)
    X_test_mlp_features_enhlib = scaler.transform(X_test_mlp_features_enhlib)


    svm_enhanced_lib = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm_enhanced_lib.fit(X_train_mlp_features_enhlib, y_train)


    print("\n--- SVM on Enhanced MLP Hidden Features ---")
    svm_results_enhanced_lib = evaluate_model(
        svm_enhanced_lib, X_test_mlp_features_enhlib, y_test,
        model_name="SVM on Enhanced MLP Hidden Features",
        class_names=class_names
    )


    plot_per_class_metrics(
        svm_results_basic_lib['precision_per_class'],
        svm_results_basic_lib['f1_per_class'],
        svm_results_basic_lib['roc_auc_per_class']
    )

    plot_per_class_metrics(
        svm_results_enhanced_lib['precision_per_class'],
        svm_results_enhanced_lib['f1_per_class'],
        svm_results_enhanced_lib['roc_auc_per_class']
    )

    # Plot ROC curves for SVM models if y_proba is available
    if svm_results_basic_lib['y_proba'] is not None:
        plot_roc_curves(svm_results_basic_lib['y_test'], svm_results_basic_lib['y_proba'], class_names)

    if svm_results_enhanced_lib['y_proba'] is not None:
        plot_roc_curves(svm_results_enhanced_lib['y_test'], svm_results_enhanced_lib['y_proba'], class_names)


    try:
        print("\n--- Training SVM on Custom MLP Features ---")

        # Extract features from custom MLPs
        X_train_mlp_features_custom_basic = extract_mlp_features(X_train_basic, baseline_mlp)
        X_test_mlp_features_custom_basic = extract_mlp_features(X_test_basic, baseline_mlp)

        X_train_mlp_features_custom_enhanced = extract_mlp_features(X_train_enhanced, enhanced_mlp)
        X_test_mlp_features_custom_enhanced = extract_mlp_features(X_test_enhanced, enhanced_mlp)

        # Normalize
        scaler_custom_basic = StandardScaler()
        X_train_mlp_features_custom_basic = scaler_custom_basic.fit_transform(X_train_mlp_features_custom_basic)
        X_test_mlp_features_custom_basic = scaler_custom_basic.transform(X_test_mlp_features_custom_basic)

        scaler_custom_enhanced = StandardScaler()
        X_train_mlp_features_custom_enhanced = scaler_custom_enhanced.fit_transform(
            X_train_mlp_features_custom_enhanced)
        X_test_mlp_features_custom_enhanced = scaler_custom_enhanced.transform(X_test_mlp_features_custom_enhanced)

        # Train SVMs
        svm_custom_basic = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        svm_custom_basic.fit(X_train_mlp_features_custom_basic, y_train)

        svm_custom_enhanced = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        svm_custom_enhanced.fit(X_train_mlp_features_custom_enhanced, y_train)

        # Evaluate
        svm_results_custom_basic = evaluate_model(
            svm_custom_basic, X_test_mlp_features_custom_basic, y_test,
            model_name="SVM on Custom Basic MLP Features",
            class_names=class_names
        )

        svm_results_custom_enhanced = evaluate_model(
            svm_custom_enhanced, X_test_mlp_features_custom_enhanced, y_test,
            model_name="SVM on Custom Enhanced MLP Features",
            class_names=class_names
        )

        # Plot for custom MLP + SVM models
        plot_per_class_metrics(
            svm_results_custom_basic['precision_per_class'],
            svm_results_custom_basic['f1_per_class'],
            svm_results_custom_basic['roc_auc_per_class']
        )

        plot_per_class_metrics(
            svm_results_custom_enhanced['precision_per_class'],
            svm_results_custom_enhanced['f1_per_class'],
            svm_results_custom_enhanced['roc_auc_per_class']
        )

        if svm_results_custom_basic['y_proba'] is not None:
            plot_roc_curves(svm_results_custom_basic['y_test'], svm_results_custom_basic['y_proba'], class_names)

        if svm_results_custom_enhanced['y_proba'] is not None:
            plot_roc_curves(svm_results_custom_enhanced['y_test'], svm_results_custom_enhanced['y_proba'], class_names)

    except Exception as e:
        print(f"Error with custom MLP feature extraction: {e}")
        print("Skipping SVM on custom MLP features evaluation.")

    # Plot confusion matrices for all models
    plot_confusion_matrix(baseline_results['confusion_matrix'], class_names,
                          title='Baseline MLP (Custom) Confusion Matrix')
    plot_confusion_matrix(enhanced_results['confusion_matrix'], class_names,
                          title='Enhanced MLP (Custom) Confusion Matrix')
    plot_confusion_matrix(sklearn_basic_results['confusion_matrix'], class_names,
                          title='Scikit-learn MLP (basic) Confusion Matrix')
    plot_confusion_matrix(sklearn_enhanced_results['confusion_matrix'], class_names,
                          title='Scikit-learn MLP (enhanced) Confusion Matrix')
    plot_confusion_matrix(svm_results_basic_lib['confusion_matrix'], class_names,
                          title='SVM on Basic MLP Features Confusion Matrix')
    plot_confusion_matrix(svm_results_enhanced_lib['confusion_matrix'], class_names,
                          title='SVM on Enhanced MLP Features Confusion Matrix')

    # # Optionally inspect raw probabilities for a few samples
    # print("\n--- Sample Genre Probabilities ---")
    # # Take first test sample as an example
    # sample_idx = 0
    # print(f"\nSample {sample_idx} (True class: {class_names[y_test[sample_idx]]})")
    #
    # print("Baseline MLP probabilities:")
    # print(get_genre_probabilities(baseline_mlp, X_test_basic[sample_idx:sample_idx + 1], class_names))
    #
    # print("\nEnhanced MLP probabilities:")
    # print(get_genre_probabilities(enhanced_mlp, X_test_enhanced[sample_idx:sample_idx + 1], class_names))
    #
    # print("\nScikit-learn Basic MLP probabilities:")
    # print(get_genre_probabilities(sklearn_mlp_basic, X_test_basic[sample_idx:sample_idx + 1], class_names))
    #
    # print("\nScikit-learn Enhanced MLP probabilities:")
    # print(get_genre_probabilities(sklearn_mlp_enhanced, X_test_enhanced[sample_idx:sample_idx + 1], class_names))

    # Final Accuracy Comparison Table
    print("\n--- Final Model Comparison ---")
    results_table = {
        "Model": [
            "Baseline MLP (Custom)",
            "Enhanced MLP (Custom)",
            "Scikit-learn MLP (basic)",
            "Scikit-learn MLP (enhanced)",
            "SVM on Basic MLP Features",
            "SVM on Enhanced MLP Features"
        ],
        "Accuracy": [
            baseline_results['accuracy'],
            enhanced_results['accuracy'],
            sklearn_basic_results['accuracy'],
            sklearn_enhanced_results['accuracy'],
            svm_results_basic_lib['accuracy'],
            svm_results_enhanced_lib['accuracy']
        ],
        "F1 (macro)": [
            baseline_results['f1_macro'],
            enhanced_results['f1_macro'],
            sklearn_basic_results['f1_macro'],
            sklearn_enhanced_results['f1_macro'],
            svm_results_basic_lib['f1_macro'],
            svm_results_enhanced_lib['f1_macro']
        ],
        "ROC AUC": [
            baseline_results['roc_auc'],
            enhanced_results['roc_auc'],
            sklearn_basic_results['roc_auc'],
            sklearn_enhanced_results['roc_auc'],
            svm_results_basic_lib['roc_auc'],
            svm_results_enhanced_lib['roc_auc']
        ],
        "Prediction Time (s)": [
            baseline_results['prediction_time'],
            enhanced_results['prediction_time'],
            sklearn_basic_results['prediction_time'],
            sklearn_enhanced_results['prediction_time'],
            svm_results_basic_lib['prediction_time'],
            svm_results_enhanced_lib['prediction_time']
        ]
    }

    # Creating a DataFrame for better display
    try:
        results_df = pd.DataFrame(results_table)
        print(results_df.to_string(index=False))
    except:

        for i in range(len(results_table["Model"])):
            model = results_table["Model"][i]
            acc = results_table["Accuracy"][i]
            f1 = results_table["F1 (macro)"][i]
            roc = results_table["ROC AUC"][i]
            time = results_table["Prediction Time (s)"][i]
            print(f"{model}: Acc={acc:.4f}, F1={f1:.4f}, ROC AUC={roc:.4f}, Time={time:.4f}s")

    # Return dictionary with key results
    return {
        'baseline_mlp_custom': baseline_mlp,
        'enhanced_mlp_custom': enhanced_mlp,
        'sklearn_mlp_basic': sklearn_mlp_basic,
        'sklearn_mlp_enhanced': sklearn_mlp_enhanced,
        'svm_basic': svm_lib,
        'svm_enhanced': svm_enhanced_lib,
        'baseline_results': baseline_results,
        'enhanced_results': enhanced_results,
        'sklearn_basic_results': sklearn_basic_results,
        'sklearn_enhanced_results': sklearn_enhanced_results,
        'svm_basic_results': svm_results_basic_lib,
        'svm_enhanced_results': svm_results_enhanced_lib,
        'class_names': class_names
    }


if __name__ == "__main__":
    results = main()