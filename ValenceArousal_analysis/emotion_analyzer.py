import os
import numpy as np
import soundfile as sf
import librosa
# from sklearn.preprocessing import MinMaxScaler

class EmotionAnalyzer:
    def __init__(self):
        pass
    
    def extract_features(self, audio_data, sr):
        features = []
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features.extend(zcr.mean(axis=1))
        # Spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        features.extend(spec_cent.mean(axis=1))
        # Spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
        features.extend(spec_bw.mean(axis=1))
        # Spectral roll-off
        spec_roll = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        features.extend(spec_roll.mean(axis=1))
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        features.extend(chroma.mean(axis=1))
        # RMS energy
        rms = librosa.feature.rms(y=audio_data)
        features.extend(rms.mean(axis=1))
        # MFCCs (13 coefficients by default)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        features.extend(mfcc.mean(axis=1))
        features.extend(mfcc.std(axis=1))
        return np.array(features)
    
    def analyze_emotion(self, audio_path):
        try:
            # Load audio file
            audio_data, sr = sf.read(audio_path)
            
            # Extract features
            features = self.extract_features(audio_data, sr)
            
            # Use raw features directly
            # Refined mapping:
            # valence: mean of chroma (indices 4:16), spectral centroid (1), mean of first 5 MFCCs (indices 21:26)
            # arousal: mean of zcr (0), spectral bandwidth (2), RMS (16), std of first 5 MFCCs (indices 34:39)
            chroma = features[4:16]
            centroid = features[1]
            mfcc_mean = features[21:26]
            valence = np.mean(np.concatenate([chroma, [centroid], mfcc_mean]))
            zcr = features[0]
            bandwidth = features[2]
            rms = features[16]
            mfcc_std = features[34:39]
            arousal = np.mean(np.concatenate([[zcr, bandwidth, rms], mfcc_std]))
            print(f"Features for {audio_path}: {features.flatten()}")
            print(f"Valence: {valence}, Arousal: {arousal}")
            return valence, arousal
        except Exception as e:
            print(f"Error analyzing {audio_path}: {str(e)}")
            return None, None

def rename_files_with_nuanced_valence_arousal(directory):
    analyzer = EmotionAnalyzer()
    features_list = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.wav'):
            file_path = os.path.join(directory, filename)
            # Extract all features for each file
            try:
                audio_data, sr = sf.read(file_path)
                # Ensure mono
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                features = analyzer.extract_features(audio_data, sr).flatten()
                # Only include files with correct feature length
                if len(features) == 43:
                    features_list.append(features)
                    filenames.append(filename)
                else:
                    print(f"Skipping {filename}: unexpected feature vector length {len(features)}")
            except Exception as e:
                print(f"Error extracting features from {filename}: {e}")
    if not features_list:
        print("No valid audio files found.")
        return
    features_arr = np.stack(features_list)
    # Normalize each feature dimension across all files (z-score)
    features_norm = (features_arr - features_arr.mean(axis=0)) / (features_arr.std(axis=0) + 1e-8)
    # Feature indices (based on extract_features):
    # 0: zcr, 1: centroid, 2: bandwidth, 3: rolloff, 4-15: chroma, 16: rms, 17-29: mfcc mean, 30-42: mfcc std
    valence_features = np.concatenate([
        features_norm[:,4:16],   # chroma
        features_norm[:,1:2],    # centroid
        features_norm[:,3:4],    # rolloff
        features_norm[:,17:30],  # mfcc mean
    ], axis=1)
    arousal_features = np.concatenate([
        features_norm[:,0:1],    # zcr
        features_norm[:,2:3],    # bandwidth
        features_norm[:,3:4],    # rolloff
        features_norm[:,16:17],  # rms
        features_norm[:,30:43],  # mfcc std
    ], axis=1)
    valence = valence_features.mean(axis=1)
    arousal = arousal_features.mean(axis=1)
    # Normalize valence and arousal to [-1, 1] across the dataset
    v_min, v_max = valence.min(), valence.max()
    a_min, a_max = arousal.min(), arousal.max()
    v_norm = 2 * (valence - v_min) / (v_max - v_min) - 1 if v_max > v_min else np.zeros_like(valence)
    a_norm = 2 * (arousal - a_min) / (a_max - a_min) - 1 if a_max > a_min else np.zeros_like(arousal)
    for i, fname in enumerate(filenames):
        ext = os.path.splitext(fname)[1]
        new_filename = f"Reference_v{v_norm[i]:.3f}_a{a_norm[i]:.3f}{ext}"
        src = os.path.join(directory, fname)
        dst = os.path.join(directory, new_filename)
        # Avoid overwriting
        if src != dst:
            if not os.path.exists(dst):
                os.rename(src, dst)
                print(f"Renamed {fname} to {new_filename}")
            else:
                print(f"Skipping rename for {fname}: {new_filename} already exists")

if __name__ == "__main__":
    import sys
    # Accept directory path as a command-line argument, or fallback to input
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        print(f"Using directory from command-line: {directory}")
    else:
        directory = input("Enter the directory path: ")
    rename_files_with_nuanced_valence_arousal(directory)