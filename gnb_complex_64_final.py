# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
import time
from dataset_for_nb import load_data # Assuming this file exists and works correctly

# --- UPDATED Variables for Experimentation ---

# 1. Feature Selection / Dimensionality Reduction
APPLY_FEATURE_SELECTION = True
N_FEATURES_TO_SELECT = 40

APPLY_PCA = True
N_PCA_COMPONENTS = 35

# 2. Preprocessing
APPLY_SCALING = True
APPLY_TRANSFORMATION = True
TRANSFORMATION_METHOD = 'quantile'
QUANTILE_N_QUANTILES = 2000

# 3. GNB Parameters
CUSTOM_EPSILON = 1e-3

# 4. Class weight adjustment
CLASS_WEIGHT_ADJUSTMENT = 1.2


# Function to apply scaling and feature transformation
def preprocess_data(X_train, X_test, apply_scaling=True, apply_transformation=True, 
                   transformation_method='quantile', n_quantiles=2000):
    """
    Applies transformation and scaling with enhanced options.
    """
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    transform_applied_msg = "No transformation applied."

    # Apply feature transformation
    if apply_transformation:
        if transformation_method == 'log':
            min_train_val = X_train_processed.min().min()
            if min_train_val <= 0:
                 shift = 1 - min_train_val
                 print(f"Data has non-positive values (min={min_train_val}). Shifting data by {shift:.4f} before log.")
                 X_train_processed += shift
                 X_test_processed += shift
            X_train_processed = np.log1p(X_train_processed)
            X_test_processed = np.log1p(X_test_processed)
            transform_applied_msg = "Applied log1p transformation."
        elif transformation_method in ['box-cox', 'yeo-johnson']:
            method = transformation_method
            # Ensure X_train_processed is numpy array for PowerTransformer checks
            if isinstance(X_train_processed, pd.DataFrame):
                 X_train_np = X_train_processed.values
            else:
                 X_train_np = X_train_processed

            if method == 'box-cox' and X_train_np.min() <= 0:
                print("Warning: Box-Cox requires positive data. Switching to Yeo-Johnson.")
                method = 'yeo-johnson'

            pt = PowerTransformer(method=method, standardize=False) # Don't standardize here
            try:
                X_train_processed = pt.fit_transform(X_train_processed)
                X_test_processed = pt.transform(X_test_processed)
                transform_applied_msg = f"Applied {method} transformation."
            except ValueError as e:
                 print(f"Error during PowerTransformation ({method}): {e}. Skipping transformation.")
                 # Keep original data if transform fails
                 X_train_processed = X_train.copy()
                 X_test_processed = X_test.copy()
        elif transformation_method == 'quantile':
            # Enhanced: Use passed n_quantiles parameter
            n_quantiles = min(n_quantiles, X_train_processed.shape[0] // 5)  # More aggressive heuristic
            if n_quantiles < 10: n_quantiles = X_train_processed.shape[0]

            qt = QuantileTransformer(output_distribution='normal', n_quantiles=n_quantiles, random_state=42)
            try:
                X_train_processed = qt.fit_transform(X_train_processed)
                X_test_processed = qt.transform(X_test_processed)
                transform_applied_msg = f"Applied Quantile transformation (output=normal, n_quantiles={n_quantiles})."
            except ValueError as e:
                 print(f"Error during Quantile Transformation: {e}. Skipping transformation.")
                 X_train_processed = X_train.copy()
                 X_test_processed = X_test.copy()
        elif transformation_method == 'none':
             pass # Do nothing
        else:
            print(f"Unknown transformation method: {transformation_method}. Skipping transformation.")

    print(transform_applied_msg)

    # Apply feature scaling
    if apply_scaling:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train_processed)
        X_test_processed = scaler.transform(X_test_processed)
        print("Applied StandardScaler.")
    else:
        print("No scaling applied.")

    return X_train_processed, X_test_processed


# Enhanced Gaussian Naive Bayes class with improved stability
class EnhancedGaussianNB:
    def __init__(self, epsilon=1e-9, class_weight_adjustment=1.0):
        self.classes = None
        self.class_priors = {}
        self.mean = {}
        self.var = {}
        self.n_features = None
        self.epsilon = epsilon
        self.class_weights = None
        self.class_weight_adjustment = class_weight_adjustment  # Factor to adjust class weights

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.classes = np.unique(y)
        n_samples, self.n_features = X.shape

        unique_classes, class_counts = np.unique(y, return_counts=True)
        print("\nClass distribution in training data:")
        for cls, count in zip(unique_classes, class_counts):
             print(f"  Class {cls}: {count} samples")
        is_imbalanced = len(set(class_counts)) > 1

        if is_imbalanced:
            print(f"Dataset appears imbalanced. Applying balanced class weights with adjustment factor {self.class_weight_adjustment}.")
            self.class_weights = compute_class_weight('balanced', classes=self.classes, y=y)
            
            # Apply additional adjustment to increase the importance of minority classes
            minority_classes = [c for c, count in zip(unique_classes, class_counts) 
                              if count < np.mean(class_counts)]
            
            class_weight_dict = dict(zip(self.classes, self.class_weights))
            
            # Increase weights for minority classes
            for c in minority_classes:
                class_weight_dict[c] *= self.class_weight_adjustment
                
            # Renormalize weights
            weight_sum = sum(class_weight_dict.values())
            for c in self.classes:
                class_weight_dict[c] /= weight_sum
                class_weight_dict[c] *= len(self.classes)  # Scale back to average of 1.0
        else:
            print("Dataset appears balanced. Using uniform class weights.")
            class_weight_dict = {c: 1.0 for c in self.classes}

        for c in self.classes:
            X_c = X[y == c]
            if X_c.shape[0] == 0:
                 print(f"Warning: No samples found for class {c} in the training data.")
                 self.class_priors[c] = 0
                 self.mean[c] = np.zeros(self.n_features)
                 self.var[c] = np.full(self.n_features, self.epsilon)
                 continue

            self.class_priors[c] = (X_c.shape[0] / n_samples) * class_weight_dict[c]
            self.mean[c] = np.mean(X_c, axis=0)
            
            # Improved variance calculation with smoothing technique
            raw_var = np.var(X_c, axis=0)
            
            # Apply a smoothing function to very small variances to avoid numerical issues
            min_var_threshold = 1e-10
            raw_var = np.maximum(raw_var, min_var_threshold)
            
            # Apply epsilon as a relative factor rather than additive
            self.var[c] = raw_var * (1 + self.epsilon)

        prior_sum = sum(self.class_priors.values())
        if prior_sum > 0:
            for c in self.classes:
                self.class_priors[c] /= prior_sum
        else:
            print("Warning: Sum of priors is zero. Predictions may be unreliable.")

        return self

    def _calculate_log_likelihood(self, x, mean, var):
        # More numerically stable computation
        log_prob = -0.5 * np.sum(np.log(2. * np.pi * var))
        
        # Use a more stable calculation for the squared difference term
        diff = x - mean
        scaled_diff = diff / np.sqrt(var)
        log_prob -= 0.5 * np.sum(scaled_diff ** 2)
        
        return log_prob

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, len(self.classes)))

        for i, x in enumerate(X):
            for j, c in enumerate(self.classes):
                 if self.class_priors[c] == 0:
                      log_probs[i, j] = -np.inf
                      continue
                 log_prior = np.log(self.class_priors[c])
                 log_likelihood = self._calculate_log_likelihood(x, self.mean[c], self.var[c])
                 log_probs[i, j] = log_prior + log_likelihood

        # Improved numerical stability in softmax calculation
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        max_log_probs[np.isneginf(max_log_probs)] = 0
        
        # Subtract max for numerical stability
        exp_log_probs = np.exp(log_probs - max_log_probs)
        
        sum_exp_log_probs = np.sum(exp_log_probs, axis=1, keepdims=True)
        probs = np.divide(exp_log_probs, sum_exp_log_probs, 
                          out=np.zeros_like(exp_log_probs), where=sum_exp_log_probs!=0)
        
        # Handle zero probability cases more intelligently by using priors
        zero_prob_rows = np.where(sum_exp_log_probs[:, 0] == 0)[0]
        if len(zero_prob_rows) > 0:
            print(f"Warning: {len(zero_prob_rows)} samples had zero probability for all classes. Using priors.")
            for row in zero_prob_rows:
                for j, c in enumerate(self.classes):
                    probs[row, j] = self.class_priors[c]
        
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


# Main function revised
def main():
    # Global declarations
    global APPLY_FEATURE_SELECTION, APPLY_PCA, N_FEATURES_TO_SELECT, N_PCA_COMPONENTS
    global APPLY_SCALING, APPLY_TRANSFORMATION, TRANSFORMATION_METHOD, QUANTILE_N_QUANTILES
    global CUSTOM_EPSILON, CLASS_WEIGHT_ADJUSTMENT

    print("Loading data...")
    X_train_full, X_test_full, y_train, y_test = load_data()

    if not isinstance(X_train_full, pd.DataFrame): X_train_full = pd.DataFrame(X_train_full)
    if not isinstance(X_test_full, pd.DataFrame): X_test_full = pd.DataFrame(X_test_full)
    if isinstance(y_train, pd.DataFrame): y_train = y_train.squeeze()
    if isinstance(y_test, pd.DataFrame): y_test = y_test.squeeze()

    # --- 1. Apply Feature Selection first ---
    feature_selector = None
    if APPLY_FEATURE_SELECTION:
        print(f"\nApplying SelectKBest to find top {N_FEATURES_TO_SELECT} features...")
        # Ensure N_FEATURES_TO_SELECT is not more than available features
        k = min(N_FEATURES_TO_SELECT, X_train_full.shape[1])
        if k < N_FEATURES_TO_SELECT:
            print(f"Warning: Requested {N_FEATURES_TO_SELECT} features, but only {X_train_full.shape[1]} available. Using k={k}.")

        # Try mutual_info_classif instead of f_classif for potentially better results
        feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        try:
            X_train = feature_selector.fit_transform(X_train_full, y_train)
            X_test = feature_selector.transform(X_test_full)
            print(f"Selected {X_train.shape[1]} features based on mutual information.")
            
            # Report top features if dataframe
            if isinstance(X_train_full, pd.DataFrame):
                selected_indices = feature_selector.get_support(indices=True)
                selected_features = X_train_full.columns[selected_indices]
                print("Top 10 selected features:", selected_features[:10].tolist())
        except Exception as e:
            print(f"Error during SelectKBest: {e}. Using all features.")
            APPLY_FEATURE_SELECTION = False
            X_train = X_train_full.copy()
            X_test = X_test_full.copy()
    else:
        # Use all features if not applying selection
        X_train = X_train_full.copy()
        X_test = X_test_full.copy()

    print(f"\nUsing {X_train.shape[1]} features after feature selection, before preprocessing.")
    
    # --- 2. Apply Preprocessing ---
    print("\nPreprocessing data...")
    X_train_processed, X_test_processed = preprocess_data(
        X_train, X_test,
        apply_scaling=APPLY_SCALING,
        apply_transformation=APPLY_TRANSFORMATION,
        transformation_method=TRANSFORMATION_METHOD,
        n_quantiles=QUANTILE_N_QUANTILES
    )

    # --- 3. Apply PCA after preprocessing ---
    pca_transformer = None
    if APPLY_PCA:
        print(f"\nApplying PCA to reduce to {N_PCA_COMPONENTS} components...")
        # Ensure N_PCA_COMPONENTS is not more than available features
        n_components = min(N_PCA_COMPONENTS, X_train_processed.shape[1])
        if n_components < N_PCA_COMPONENTS:
             print(f"Warning: Requested {N_PCA_COMPONENTS} components, but only {X_train_processed.shape[1]} features available. Using n_components={n_components}.")

        if n_components > 0:
            pca_transformer = PCA(n_components=n_components, random_state=42)
            try:
                X_train_processed = pca_transformer.fit_transform(X_train_processed)
                X_test_processed = pca_transformer.transform(X_test_processed)
                print(f"Applied PCA. New shape: {X_train_processed.shape}")
                explained_variance = pca_transformer.explained_variance_ratio_.sum()
                print(f"Explained variance ratio by {n_components} components: {explained_variance:.4f}")
            except Exception as e:
                print(f"Error during PCA: {e}. Skipping PCA.")
                APPLY_PCA = False
                # No need to revert - X_train_processed is already preprocessed

    # --- 4. Model Training and Evaluation ---
    genre_mapping = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 
                    5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
    class_names = [genre_mapping[i] for i in range(len(genre_mapping))]

    # Use the final processed data (potentially feature selected/PCA'd)
    final_X_train = X_train_processed
    final_X_test = X_test_processed

    # Train Custom EnhancedGaussianNB
    print(f"\n--- Training Custom EnhancedGaussianNB (epsilon={CUSTOM_EPSILON}, class_weight_adj={CLASS_WEIGHT_ADJUSTMENT}) ---")
    start_fit_time = time.time()
    custom_gnb = EnhancedGaussianNB(epsilon=CUSTOM_EPSILON, class_weight_adjustment=CLASS_WEIGHT_ADJUSTMENT)
    custom_gnb.fit(final_X_train, y_train)
    fit_time = time.time() - start_fit_time
    print(f"Fit time: {fit_time:.4f} seconds")

    custom_results = evaluate_model(custom_gnb, final_X_test, y_test, "Custom GNB", class_names)
    print("\nConfusion Matrix for Custom GaussianNB:")
    plot_confusion_matrix(custom_results['confusion_matrix'], class_names, 
                       title=f'Custom GNB (epsilon={CUSTOM_EPSILON}, adj={CLASS_WEIGHT_ADJUSTMENT})')

    return custom_results


# Evaluate function
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
        labels = np.unique(np.concatenate((y_test, y_pred)))
        target_names_filtered = [class_names[i] for i in labels if i < len(class_names)]
        if len(target_names_filtered) < len(np.unique(y_test)):
             print("Warning: Some classes were not present in predictions or true labels for the report.")
        try:
            report = classification_report(y_test, y_pred, target_names=class_names, 
                                        labels=np.arange(len(class_names)), zero_division=0)
        except ValueError:
             print("Error generating classification report with specified target names/labels. Using default.")
             report = classification_report(y_test, y_pred, zero_division=0)
    else:
        report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'prediction_time': prediction_time
    }

# Plot function
def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    custom_results = main()
    # You can further analyze the results stored in this variable