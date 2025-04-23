import os
import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings('ignore')

def extract_features(file_path, label=None):
    try:
        y, sr = librosa.load(file_path, sr=None)
        filename = os.path.basename(file_path)
        length = len(y)
        
        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_var = np.var(chroma_stft)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_var = np.var(spectral_centroid)
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_var = np.var(spectral_bandwidth)
        
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_var = np.var(rolloff)
        
        # Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_var = np.var(zero_crossing_rate)
        
        # Harmony and Perceptr (formerly Contrast)
        harmony, perceptr = librosa.effects.hpss(y)
        harmony_mean = np.mean(harmony)
        harmony_var = np.var(harmony)
        perceptr_mean = np.mean(perceptr)
        perceptr_var = np.var(perceptr)
        
        # Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # MFCCs (20 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_vars = np.var(mfccs, axis=1)
        
        # Create feature dictionary
        features = {
            'filename': filename,
            'length': length,
            'chroma_stft_mean': chroma_stft_mean,
            'chroma_stft_var': chroma_stft_var,
            'rms_mean': rms_mean,
            'rms_var': rms_var,
            'spectral_centroid_mean': spectral_centroid_mean,
            'spectral_centroid_var': spectral_centroid_var,
            'spectral_bandwidth_mean': spectral_bandwidth_mean,
            'spectral_bandwidth_var': spectral_bandwidth_var,
            'rolloff_mean': rolloff_mean,
            'rolloff_var': rolloff_var,
            'zero_crossing_rate_mean': zero_crossing_rate_mean,
            'zero_crossing_rate_var': zero_crossing_rate_var,
            'harmony_mean': harmony_mean,
            'harmony_var': harmony_var,
            'perceptr_mean': perceptr_mean,
            'perceptr_var': perceptr_var,
            'tempo': tempo
        }
        
        # Add MFCCs
        for i, (mean, var) in enumerate(zip(mfcc_means, mfcc_vars), 1):
            features[f'mfcc{i}_mean'] = mean
            features[f'mfcc{i}_var'] = var
        
        # Add label if provided
        if label is not None:
            features['label'] = label
            
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def extract_features_from_directory(directory_path, label=None):
    features_list = []
    
    for file in os.listdir(directory_path):
        # Check if file is an audio file
        if file.endswith(('.wav', '.mp3', '.ogg')):
            file_path = os.path.join(directory_path, file)
            features = extract_features(file_path, label)
            if features:
                features_list.append(features)
    
    # Create DataFrame
    if features_list:
        return pd.DataFrame(features_list)
    else:
        return None

def extract_features_from_file_list(file_list, labels=None):
    """
    Extract features from a list of individual files
    
    Parameters:
    file_list (list): List of file paths
    labels (list): Optional list of labels corresponding to each file
    
    Returns:
    pd.DataFrame: DataFrame containing features for all files
    """
    features_list = []
    
    for i, file_path in enumerate(file_list):
        label = labels[i] if labels and i < len(labels) else None
        features = extract_features(file_path, label)
        if features:
            features_list.append(features)
    
    # Create DataFrame
    if features_list:
        return pd.DataFrame(features_list)
    else:
        return None

def main():
    audio_directories = {
        'blues': '/Users/dilyaraarynova/MLProject/exp_data/blues',
        'jazz': '/Users/dilyaraarynova/MLProject/exp_data/jazz',
        # ...etc
    }
    
    individual_files = {
        'blues': [
            '/Users/dilyaraarynova/MLProject/exp_data/blues/blues.00000.wav',
            '/Users/dilyaraarynova/MLProject/exp_data/blues/blues.00001.wav'
        ],
        'jazz': [
            '/Users/dilyaraarynova/MLProject/exp_data/jazz/jazz.00000.wav'
        ]
    }
    
    all_features = []
    
    print("Processing individual files...")
    for label, files in individual_files.items():
        file_features = []
        for file_path in files:
            print(f"Extracting features from {os.path.basename(file_path)}...")
            features = extract_features(file_path, label)
            if features:
                file_features.append(features)
        
        if file_features:
            features_df = pd.DataFrame(file_features)
            all_features.append(features_df)
    
    print("\nProcessing directories...")
    for label, directory in audio_directories.items():
        print(f"Extracting features from {label} directory...")
        features_df = extract_features_from_directory(directory, label)
        if features_df is not None:
            all_features.append(features_df)
    
    if all_features:
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Save features to CSV
        combined_df.to_csv('extracted_audio_features.csv', index=False)
        print(f"Features saved to extracted_audio_features.csv")
    else:
        print("No features were extracted.")

# for a single file
def process_single_file(file_path, label=None, output_csv=None):

    print(f"Extracting features from {os.path.basename(file_path)}...")
    features = extract_features(file_path, label)
    
    if features:
        df = pd.DataFrame([features])
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Features saved to {output_csv}")
            return None
        else:
            return df
    else:
        print("No features were extracted.")
        return None

if __name__ == "__main__":
    
    main()
    
    # Option 2: Process a single file
    # file_path = '/Users/dilyaraarynova/MLProject/exp_data/blues/blues.00000.wav'
    # process_single_file(file_path, label='blues', output_csv='single_file_features.csv')
