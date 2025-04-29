# Emotion-TTS Project

## Overview
Emotion-TTS is a research and analysis project focused on the extraction, analysis, and synthesis of emotional characteristics from speech audio. The project leverages the valence-arousal model to quantify and visualize emotions in audio files and provides tools for creating and analyzing emotional speech datasets.

## Folder Structure

- `Emotion_Research_Analysis.ipynb`: Jupyter notebook for analyzing the distribution of emotional valence and arousal in labeled speech audio files. Includes data visualization (scatter plots, density plots, boxplots), statistical summaries, and conclusions about the dataset's emotional diversity.
- `EmotionalTTS.ipynb`: Jupyter notebook guiding the creation of an emotional Text-to-Speech (TTS) system. Walks through recording, labeling, and synthesizing speech with varying emotions using reference audio samples.
- `Reference_Audio/`: Contains example reference audio files labeled with valence and arousal values (e.g., `Reference_v-0.042_a-0.217.wav`).
- `ValenceArousal_analysis/`: Contains the main scripts and data for valence-arousal feature extraction and analysis.
  - `emotion_analyzer.py`: Python script for extracting acoustic features from audio files, mapping them to valence and arousal, and renaming files according to their computed values. Includes a class `EmotionAnalyzer` for feature extraction and emotion analysis.
  - `requirements.txt`: Python dependencies for running the analysis scripts (librosa, numpy, scikit-learn, soundfile).
  - `Reference_Audio/` & `Reference_Audio-Updated/`: Audio subfolders with WAV files labeled or renamed based on computed valence and arousal.

## How It Works

### 1. Feature Extraction & Labeling
- The `emotion_analyzer.py` script extracts features (zero crossing rate, spectral centroid, chroma, MFCCs, etc.) from each WAV file.
- Features are mapped to valence (pleasantness) and arousal (energy) scores.
- Files are renamed to encode their normalized valence and arousal in the filename (e.g., `Reference_v-0.445_a1.000.wav`).

### 2. Dataset Analysis
- `Emotion_Research_Analysis.ipynb` loads the labeled audio files, parses valence/arousal from filenames, and creates visualizations:
  - **Scatter Plot:** Shows the spread and clustering of emotions in the dataset.
  - **Density & Boxplots:** Display the distribution and variability of valence/arousal.
  - **Statistical Summaries:** Quantify the diversity and balance of emotions.

### 3. Emotional TTS System
- `EmotionalTTS.ipynb` demonstrates how to use your own voice samples to create a TTS system that can synthesize speech with different emotional tones by varying reference audio.

## How to Run

### Requirements
- Python 3.8+
- Install dependencies from `ValenceArousal_analysis/requirements.txt`:
  ```bash
  pip install -r ValenceArousal_analysis/requirements.txt
  ```

### Feature Extraction & File Renaming
- Run `emotion_analyzer.py` to process a directory of WAV files:
  ```bash
  python ValenceArousal_analysis/emotion_analyzer.py <directory_path>
  ```
  - If no directory is provided, you will be prompted to enter one.
  - The script will output new filenames reflecting valence and arousal.

### Dataset Analysis
- Open `Emotion_Research_Analysis.ipynb` in Jupyter or Colab. Run all cells to see visualizations and analysis of your audio dataset.

### Emotional TTS
- Open `EmotionalTTS.ipynb` in Jupyter or Colab. Follow the instructions to record, label, and synthesize emotional speech.

## Reference Audio Files
- Example files in `Reference_Audio/` and `ValenceArousal_analysis/Reference_Audio-Updated/` are labeled by valence/arousal (e.g., `Reference_v1.000_a-1.000.wav`).
- `ValenceArousal_analysis/Reference_Audio/` also contains files named by emotion (e.g., `Angry.wav`, `Calm.wav`).

## Notes
- The project uses the valence-arousal model (see [Russell, 1980](https://doi.org/10.1037/h0077714)) for emotion representation.
- All code is for research and educational purposes.

## References
- Russell, J. A. (1980). "A circumplex model of affect." Journal of Personality and Social Psychology, 39(6), 1161–1178. https://doi.org/10.1037/h0077714
- Scherer, K. R. (2005). "What are emotions? And how can they be measured?" Social Science Information, 44(4), 695–729. https://doi.org/10.1177/0539018405058216
- Eyben, F., Wöllmer, M., & Schuller, B. (2010). "openSMILE – The Munich Versatile and Fast Open-Source Audio Feature Extractor." Proceedings of the ACM Multimedia (MM), 1459–1462. https://doi.org/10.1145/1873951.1874246
- Additional references can be found in the notebooks.

## Author
- Srishti Binwani
