# Multiclass Similarity-Based Modeling (SBM) for Rotating-Machine Fault Diagnosis

This repository contains an optimized Python implementation of the Multiclass Similarity-Based Modeling (SBM) architecture for detecting and classifying faults in rotating machines.

The theoretical foundation and pipeline strictly reproduce the methodology presented in the scientific paper:
**["Improved similarity-based modeling for the classification of rotating-machine failures"](https://doi.org/10.1016/j.jfranklin.2017.07.038)**
*Marins, M. A., Ribeiro, F. M. L., Netto, S. L., & da Silva, E. A. B. (Journal of the Franklin Institute, 2018).*
DOI: [10.1016/j.jfranklin.2017.07.038](https://doi.org/10.1016/j.jfranklin.2017.07.038)

Our fine-tuned implementation achieves an outstanding 98.47 percent accuracy on the test set, closely matching the paper's reported peak performance (~98.48 percent) for the Model B configuration. Notably, the model achieved 100 percent precision and recall for the severely underrepresented Normal operating class.

The project utilizes the [Machinery Fault Database (MaFaulDa)](https://www02.smt.ufrj.br/~offshore/mfs/index.html), an extensive multivariate time-series database acquired from a SpectraQuest alignment-balance-vibration trainer.
* **Sensors**: 6 accelerometers, 1 microphone, and 1 tachometer (sampled at 50 kHz).
* **Scenarios**: 1951 unique operational scenarios (5 seconds each).
* **Classes**: Normal Operation (49), Imbalance (333), Horizontal Misalignment (197), Vertical Misalignment (301), Overhang Bearing Fault (513), and Underhang Bearing Fault (558).

The solution is divided into three main conceptual modules:

### 1. Feature Extraction (Step 2)
Transforms raw multivariate time-series into a condensed 46-dimensional feature vector.
* **Spectral Features (22)**: Extracts the rotation frequency (f_r) from the tachometer's Discrete Fourier Transform (DFT). Then, it extracts the exact magnitudes of the remaining 7 sensors at f_r, 2 * f_r, and 3 * f_r.
* **Statistical Features (24)**: Computes the mean, entropy, and kurtosis for each of the 8 sensors.

### 2. SBM Class Dictionaries (Step 3)
Models the normal manifold of each fault class to generate highly discriminative similarity scores.
* **Initialization**: Employs a robust approximation of the Geometric Median of the class states.
* **Memory Matrix Construction (Threshold Method)**: Iteratively builds a compact dictionary of representative states for each class. A new state is memorized if its similarity to all existing states is strictly less than a threshold.
* **Similarity Estimation**: Uses the Wegerich Similarity Function (WSF) with L1 norm (Manhattan distance) to generate 6 similarity scores for any given input.

### 3. Classification Ensemble (Step 4)
An ensemble approach combining the engineered features with the SBM outputs.
* **Feature Extension**: Extends the original 46 features with the 6 SBM similarity scores (replicating the 3rd configuration of Experiment 3 from the paper).
* **Ensemble Model**: Employs a Random Forest Classifier with balanced class weights to mitigate the natural scarcity of Normal operation data.

---

## 🚀 How to Run the Project

### 1. Installation & Dependencies

To execute the rotating-machine diagnosis pipeline, you must install the core mathematical and machine learning libraries. You can install them via `pip`:
```bash
pip install numpy scipy scikit-learn
```
Alternatively, if you are using Conda, create and activate your environment with:
```bash
conda create -n mafaulda_env python=3.10 numpy scipy scikit-learn -y
conda activate mafaulda_env
```

### 2. MaFaulDa Database Folder Structure

The pipeline recursively traverses the raw dataset directory to map CSV operational scenarios to their respective labels based on directory structures. Ensure the raw MaFaulDa database folder is structured exactly as follows:
```text
mafaulda/
├── normal/
│   └── (normal operating state .csv files)
├── imbalance/
│   └── (imbalance fault .csv files)
├── horizontal-misalignment/
│   └── (horizontal misalignment fault .csv files)
├── vertical-misalignment/
│   └── (vertical misalignment fault .csv files)
├── overhang/
│   └── (overhang bearing fault .csv files)
└── underhang/
    └── (underhang bearing fault .csv files)
```

### 3. Execution Commands

To run the end-to-end fault diagnosis pipeline optimally using our advanced digital signal processing (DSP) parameters, run:
```bash
python main.py --use_hann --use_fixed_entropy
```

#### Exposing Command-Line Arguments

The pipeline exposes the following flexible arguments to control feature extraction, cross-validation, and caching:

* `--dataset_path <path>`: Specifies the directory path to the raw MaFaulDa database folder (defaults to `~/datasets/mafaulda`).
* `--use_hann`: Applies a Hanning window function (with exact coherent gain correction) to the Discrete Fourier Transform (DFT) signals. This mitigates spectral leakage and side-lobe noise, allowing peak frequency magnitude extraction to be highly precise.
* `--use_fixed_entropy`: Strictly locks the Shannon entropy histogram range to `(-10.0, 10.0)` across all files. This fixes a critical signal processing bug where dynamic-bin scaling (default behavior) distorts entropy features when signals experience sudden peak/shock impulses.
* `--tune`: Executes a full Stratified 10-fold Cross-Validation grid search to inspect SBM hyperparameter performance over gamma ($\gamma$) and threshold tau ($\tau$).
* `--skip_extraction`: Speeds up SBM iteration and model training loops by reusing pre-extracted features under the `./data` directory instead of parsing the 1,951 raw CSV files.
