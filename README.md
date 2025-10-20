# Music Genre Classification: A Comparative Study of Classical and Deep Learning Methods

This repository contains the code and experiments for the paper "Music Genre Classification Using Classical and Deep Learning Methods: A Comparative Study" by Rauan Arstangaliyev, Kamila Spanova, Moldir Azhimukhanbet, Dilyara Arynova, and Amina Aimuratova from Nazarbayev University.

## Abstract

We explore different machine learning approaches for music genre classification. The study begins by combining Naive Bayes (NB) with a Multilayer Perceptron (MLP) and further enhances this by incorporating kernel methods (SVM) to capture complex patterns. Additionally, deep convolutional recurrent neural networks (CRNNs) are applied to two types of features: raw Mel-spectrograms and engineered statistical summaries of audio properties. Using the GTZAN dataset, models are compared based on accuracy, F1 score, and ROC-AUC. Our experimental results show that the best performance was achieved by combining MLP and SVM, as well as CRNNs trained on engineered features. Considering model complexity, we conclude that the MLP and SVM combination offers a practical and effective solution for this task.

**Index Terms:** Music Genre Classification, Naive Bayes, Multilayer Perceptron, Kernel Methods, SVM, CNN, CRNN, Classifier Combination, GTZAN.

## Key Findings


<img width="1060" height="181" alt="image" src="https://github.com/user-attachments/assets/dc898322-4e8c-4036-88c0-49fa9518cb74" />
<img width="740" height="170" alt="image" src="https://github.com/user-attachments/assets/27ff43ec-4f76-4d40-bba6-b929094bd7ff" />


*   **MLP + SVM:** Achieved approximately 90% accuracy and F1-score.
*   **CRNN on Engineered Features:** Also achieved ~90% accuracy and F1-score. 
*   **CRNN on Raw Mel-spectrograms:** Performed respectably (around 82-85% accuracy).
*   **GNB as a Feature Enhancer:** Showed modest improvements, especially when SVM was not yet added.

## Repository Structure

This repository contains the following key scripts and files:

*   `extract_features.py`: Script for extracting audio features (e.g., MFCCs, Chroma, Spectral Centroid, etc.) from the GTZAN dataset.
*   `dataset.py`: General dataset loading and preprocessing utilities.
*   `dataset_for_mlp.py`: Specific dataset preparation tailored for MLP models.
*   `dataset_for_nb.py`: Specific dataset preparation tailored for Naive Bayes models.
*   `gnb_simple_52_final.py`: Implementation and evaluation of a simpler Gaussian Naive Bayes model.
*   `gnb_complex_64_final.py`: Implementation and evaluation of an enhanced/complex Gaussian Naive Bayes model (likely involving feature transformations like Box-Cox/Yeo-Johnson and PCA as described in the paper).
*   `mlp.py`: Implementation and evaluation of Multilayer Perceptron models, potentially including the MLP+SVM combination.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.


## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Moldier/genreclassification.git
    cd genreclassification
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    We recommend creating a `requirements.txt` file. Based on the paper, common libraries would include:
    ```
    numpy
    scipy
    scikit-learn
    librosa
    matplotlib
    # Add pandas if used for data handling
    # Add TensorFlow or PyTorch if used for CRNNs
    ```
    Install them using:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Dataset:**
    *   Download the GTZAN dataset. Available at: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
    *   Ensure the dataset is placed in a location accessible by the scripts (e.g., a `data/gtzan/` directory) or update the paths within the scripts accordingly.

## Usage

1.  **Feature Extraction:**
    Run `extract_features.py` to process the GTZAN audio files and save the engineered features.
    ```bash
    python extract_features.py
    ```
    
2.  **Running Models:**
    Execute the individual model scripts to train and evaluate them:
    ```bash
    python gnb_simple_52_final.py
    python gnb_complex_64_final.py
    python mlp.py
    # Add commands for other models/experiments as needed
    ```
    The scripts will load pre-extracted features (or extract them if designed that way) and output performance metrics (accuracy, F1-score, and confusion matrices).

## Authors

*   Rauan Arstangaliyev (rauan.arstangaliyev@nu.edu.kz)
*   Kamila Spanova (kamila.spanova@nu.edu.kz)
*   Moldir Azhimukhanbet (moldir.azhimukhanbet@nu.edu.kz)
*   Dilyara Arynova (dilyara.arynova@nu.edu.kz)
*   Amina Aimuratova (amina.aimuratova@nu.edu.kz)

School of Engineering and Digital Sciences, Nazarbayev University, Astana, Kazakhstan.
