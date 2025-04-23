import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import SVC
import time
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from dataset_for_mlp import load_data
from gnb_simple_52_final import CustomGaussianNB
from gnb_complex_64_final import preprocess_data
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


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
            # Xavier initialization for sigmoid/tanh: sqrt(1/n_in)
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

    # Plot custom model histories
    plot_training_history(baseline_history, title='Baseline MLP (Custom) Training History')
    plot_training_history(enhanced_history, title='Enhanced MLP (Custom) Training History')

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

    # Done with scikit-learn MLP

    # ---Moving to SVM---
    print("\n--- Training scikit-learn MLPClassifier+library SVM ---")

    def relu(X):
        return np.maximum(0, X)

    def extract_mlp_features(X, model):
        W1, W2, W3 = model.coefs_[0], model.coefs_[1], model.coefs_[2]
        b1, b2, b3 = model.intercepts_[0], model.intercepts_[1], model.intercepts_[2]

        h1 = relu(np.dot(X, W1) + b1)
        h2 = relu(np.dot(h1, W2) + b2)
        return h2

    # Step 1: Extract features from MLP hidden layer
    print("Extracting MLP hidden features...")
    X_train_mlp_features_baslib = extract_mlp_features(X_train_basic, sklearn_mlp_basic)
    X_test_mlp_features_baslib = extract_mlp_features(X_test_basic, sklearn_mlp_basic)

    # Normalize hidden outputs
    scaler = StandardScaler()
    X_train_mlp_features_baslib = scaler.fit_transform(X_train_mlp_features_baslib)
    X_test_mlp_features_baslib = scaler.transform(X_test_mlp_features_baslib)

    # Step 2: Train SVM on hidden features
    svm_lib = SVC(kernel='rbf', C=10, gamma='scale')
    svm_lib.fit(X_train_mlp_features_baslib, y_train)

    print("\n--- SVM on MLP Hidden Features ---")
    svm_results_basic_lib = evaluate_model(
        svm_lib, X_test_mlp_features_baslib, y_test,
        model_name="SVM on MLP Hidden Features",
        class_names=class_names
    )

    print("\n--- Training scikit-learn MLPClassifier+SVM+NB ---")

    # Step 1: Extract features from MLP hidden layer
    X_train_mlp_features_enhlib = extract_mlp_features(X_train_enhanced, sklearn_mlp_enhanced)
    X_test_mlp_features_enhlib = extract_mlp_features(X_test_enhanced, sklearn_mlp_enhanced)

    # (Optional) Normalize hidden outputs
    scaler = StandardScaler()
    X_train_mlp_features_enhlib = scaler.fit_transform(X_train_mlp_features_enhlib)
    X_test_mlp_features_enhlib = scaler.transform(X_test_mlp_features_enhlib)

    # Step 2: Train SVM on hidden features
    svm_enhanced_lib = SVC(kernel='rbf', C=10, gamma='scale')
    svm_enhanced_lib.fit(X_train_mlp_features_enhlib, y_train)

    # Step 3: Predict and evaluate
    print("\n--- SVM on MLP Hidden Features ---")
    svm_results_enhanced_lib = evaluate_model(
        svm_enhanced_lib, X_test_mlp_features_enhlib, y_test,
        model_name="SVM on MLP Hidden Features+NB",
        class_names=class_names
    )

    # print("\n--- Training MLPClassifier+SVM ---")
    #
    # # Step 1: Extract features from MLP hidden layer
    # X_train_mlp_features_basmlp = extract_mlp_features(X_train_basic, baseline_mlp)
    # X_test_mlp_features_basmlp = extract_mlp_features(X_test_basic, baseline_mlp)
    #
    # # (Optional) Normalize hidden outputs
    # scaler = StandardScaler()
    # X_train_mlp_features_basmlp = scaler.fit_transform(X_train_mlp_features_basmlp)
    # X_test_mlp_features_basmlp = scaler.transform(X_test_mlp_features_basmlp)
    #
    # # Step 2: Train SVM on hidden features
    # svm = SVC(kernel='rbf', C=10, gamma='scale')
    # svm.fit(X_train_mlp_features_basmlp, y_train)
    #
    # # Step 3: Predict and evaluate
    # print("\n--- SVM on MLP Hidden Features ---")
    # svm_results_basic_mlp = evaluate_model(
    #     svm, X_test_mlp_features_basmlp, y_test,
    #     model_name="SVM on MLP Hidden Features+NB",
    #     class_names=class_names
    # )
    #
    # print("\n--- Training MLPClassifier+SVM+NB ---")
    #
    # # Step 1: Extract features from MLP hidden layer
    # X_train_mlp_features_enhmlp = extract_mlp_features(X_train_basic, enhanced_mlp)
    # X_test_mlp_features_enhmlp = extract_mlp_features(X_test_basic, enhanced_mlp)
    #
    # # (Optional) Normalize hidden outputs
    # scaler = StandardScaler()
    # X_train_mlp_features_enhmlp = scaler.fit_transform(X_train_mlp_features_enhmlp)
    # X_test_mlp_features_enhmlp = scaler.transform(X_test_mlp_features_enhmlp)
    #
    # # Step 2: Train SVM on hidden features
    # svm_enhanced = SVC(kernel='rbf', C=10, gamma='scale')
    # svm_enhanced.fit(X_train_mlp_features_enhmlp, y_train)
    #
    # # Step 3: Predict and evaluate
    # print("\n--- SVM on MLP Hidden Features ---")
    # svm_results_enhanced_mlp = evaluate_model(
    #     svm_enhanced, X_test_mlp_features_enhmlp, y_test,
    #     model_name="SVM on MLP Hidden Features+NB",
    #     class_names=class_names
    # )





    # Final Comparison
    print("\n--- Final Model Comparison ---")
    print(f"Baseline MLP (Custom) Accuracy:                {baseline_results['accuracy']:.4f}")
    print(f"Enhanced MLP (Custom) Accuracy:                {enhanced_results['accuracy']:.4f}")
    print("-" * 50)
    print(f"Scikit-learn MLP (basic) Accuracy:             {sklearn_basic_results['accuracy']:.4f}")
    print(f"Scikit-learn MLP (enhanced) Accuracy:          {sklearn_enhanced_results['accuracy']:.4f}")
    #svm
    print(f"Scikit-learn MLP (basic) SVM Accuracy:          {svm_results_basic_lib['accuracy']:.4f}")
    print(f"Scikit-learn MLP (enhanced) SVM Accuracy:          {svm_results_enhanced_lib['accuracy']:.4f}")
    # print(f" MLP (basic) SVM Accuracy:          {svm_results_basic_mlp['accuracy']:.4f}")
    # print(f"MLP (enhanced) SVM Accuracy:          {svm_results_enhanced_mlp['accuracy']:.4f}")


    # Plot confusion matrices for key models
    plot_confusion_matrix(baseline_results['confusion_matrix'], class_names,
                          title='Baseline MLP (Custom) Confusion Matrix')
    plot_confusion_matrix(enhanced_results['confusion_matrix'], class_names,
                          title='Enhanced MLP (Custom) Confusion Matrix')
    plot_confusion_matrix(sklearn_basic_results['confusion_matrix'], class_names,
                          title='Scikit-learn MLP (basic) Confusion Matrix')
    plot_confusion_matrix(sklearn_enhanced_results['confusion_matrix'], class_names,
                          title='Scikit-learn MLP (enhanced) Confusion Matrix')

    plot_confusion_matrix(svm_results_basic_lib['confusion_matrix'], class_names,
                          title='Scikit-learn MLP (basic) SVM Confusion Matrix')
    plot_confusion_matrix(svm_results_enhanced_lib['confusion_matrix'], class_names,
                          title='Scikit-learn MLP (enhanced) SVM Confusion Matrix')
    # plot_confusion_matrix(svm_results_basic_mlp['confusion_matrix'], class_names,
    #                       title='MLP (basic) SVM Confusion Matrix')
    # plot_confusion_matrix(svm_results_enhanced_mlp['confusion_matrix'], class_names,
    #                       title='MLP (enhanced) SVM Confusion Matrix')


    # Return dictionary with key results (optional)
    results_dict = {
        'baseline_mlp_custom': baseline_mlp,
        'enhanced_mlp_custom': enhanced_mlp,
        'sklearn_mlp_basic': sklearn_mlp_basic,
        'sklearn_mlp_enhanced': sklearn_mlp_enhanced,
        'baseline_results': baseline_results,
        'enhanced_results': enhanced_results,
        'sklearn_basic_results': sklearn_basic_results,
        'sklearn_enhanced_results': sklearn_enhanced_results,
    }
    return results_dict
if __name__ == "__main__":
    results = main()

