# Multiclass Similarity-Based Modeling (SBM) for Rotating-Machine Fault Diagnosis

This repository contains an optimized, **zero-dependency pure Python** implementation of the Multiclass Similarity-Based Modeling (SBM) architecture for detecting and classifying faults in rotating machines.

> [!TIP]
> **🚀 GPU-Accelerated Version Available**: A high-performance GPU-accelerated implementation of this pipeline (utilizing PyTorch and CUDA for up to 30% faster execution) is available on the [`gpu` branch](https://github.com/titospadini/mafaulda-sbm/tree/gpu). Switch to that branch to check it out!

The theoretical foundation and pipeline strictly reproduce the methodology presented in the scientific paper:
**["Improved similarity-based modeling for the classification of rotating-machine failures"](https://doi.org/10.1016/j.jfranklin.2017.07.038)**
*Marins, M. A., Ribeiro, F. M. L., Netto, S. L., & da Silva, E. A. B. (Journal of the Franklin Institute, 2018).*
DOI: [10.1016/j.jfranklin.2017.07.038](https://doi.org/10.1016/j.jfranklin.2017.07.038)

Our fine-tuned implementation achieves an outstanding 98.47% accuracy on the test set, closely matching the paper's reported peak performance ($\approx 98.49$%) for the Model B configuration. Notably, the model achieved 100% precision and recall for the severely underrepresented Normal operating class.

The project utilizes the [Machinery Fault Database (MaFaulDa)](https://www02.smt.ufrj.br/~offshore/mfs/index.html), an extensive multivariate time-series database acquired from a SpectraQuest alignment-balance-vibration trainer.
* **Sensors**: 6 accelerometers, 1 microphone, and 1 tachometer (sampled at 50 kHz).
* **Scenarios**: 1951 unique operational scenarios (5 seconds each).
* **Classes**: Normal Operation (49), Imbalance (333), Horizontal Misalignment (197), Vertical Misalignment (301), Overhang Bearing Fault (513), and Underhang Bearing Fault (558).

The solution is structured into three main pipeline modules:

1. **Feature Extraction (Step 2)**: Transforms raw, high-frequency multivariate time-series signals into a 46-dimensional hand-crafted diagnostic feature vector (1 rotation frequency, 21 harmonic spectral amplitudes, and 24 statistical descriptors including Mean, Shannon Entropy, and Kurtosis).
2. **SBM Class Dictionaries (Step 3)**: Models the normal operational manifold of each of the 6 fault classes by building a compact representative dictionary matrix, seeded with Weiszfeld's Geometric Median and grown using the Threshold Method.
3. **Classification Ensemble (Step 4)**: Blends the engineered features with SBM-derived vectors—supporting either **SBM Model B** (concatenates 46-dimensional SBM error residuals, producing 92 features) or **Experiment 3 Configuration 3** (concatenates 6-dimensional direct similarity scores, producing 52 features)—classified using a balanced Random Forest ensemble.

> [!NOTE]
> For the complete mathematical formulations, exact equations, and exhaustive definitions of every symbol, variable, and constant, please refer to the [🔬 Detailed Mathematical Foundations](#-detailed-mathematical-foundations) section at the bottom of this page.

------

## 🚀 How to Run the Project

### 1. Installation & Dependencies

This is a **pure Python** implementation of the SBM architecture. It relies **exclusively** on the Python Standard Library (e.g., `math`, `random`, `pickle`, `multiprocessing`, `argparse`). 

**No third-party packages (such as NumPy, SciPy, or scikit-learn) are required to run the pipeline!** 

To execute the code, you only need:
* **Python 3.10 or higher** installed on your system.


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

### 3. Execution Options

You can reproduce the paper's experiments either by running the modular CLI Python scripts directly, or interactively using the provided Jupyter Notebook.

#### Option A: Direct CLI Scripts

##### Running SBM Model B (Default 92-Dimensional Error Residuals)
To execute the unified end-to-end Model B classification pipeline using our advanced digital signal processing (DSP) enhancements, run:
```bash
python main.py --use_hann --use_fixed_entropy
```

##### Running Experiment 3 Configuration 3 (52-Dimensional SBM Similarities)
To execute the replication pipeline utilizing direct SBM class similarity vectors as extended features, run:
```bash
python scripts/exp3_cfg3.py
```

##### Exposing Command-Line Arguments

The pipeline exposes the following flexible arguments to control feature extraction, cross-validation, and caching:

* `--dataset_path <path>`: Specifies the directory path to the raw MaFaulDa database folder (defaults to `~/datasets/mafaulda`).
* `--use_hann`: Applies a Hanning window function (with exact coherent gain correction) to the Discrete Fourier Transform (DFT) signals. This mitigates spectral leakage and side-lobe noise, allowing peak frequency magnitude extraction to be highly precise.
* `--use_fixed_entropy`: Strictly locks the Shannon entropy histogram range to `(-10.0, 10.0)` across all files. This fixes a critical signal processing bug where dynamic-bin scaling (default behavior) distorts entropy features when signals experience sudden peak/shock impulses.
* `--tune`: Executes a full Stratified 10-fold Cross-Validation grid search to inspect SBM hyperparameter performance over gamma ($\gamma$) and threshold tau ($\tau$).
* `--skip_extraction`: Speeds up SBM iteration and model training loops by reusing pre-extracted features under the `./data` directory instead of parsing the 1,951 raw CSV files.

#### Option B: Interactive Jupyter Notebook

For an interactive, step-by-step visual walk-through of the replication pipeline, you can use the provided Jupyter Notebook. This is an entirely optional alternative to using the CLI scripts:

1. Ensure Jupyter is installed in your environment:
   ```bash
   pip install jupyter
   # Or using Conda
   conda install jupyter -y
   ```
2. Start the notebook server and open the notebook:
   ```bash
   jupyter notebook notebooks/reproduce_experiments.ipynb
   ```
3. Execute the cells sequentially to inspect feature extraction, Weiszfeld centroid construction, memory dictionary growth, SBM residual generation, and Random Forest classification reports interactively.

---

## 🔬 Detailed Mathematical Foundations

This section provides the comprehensive mathematical equations, physical constants, variables, and index definitions for every conceptual module in the Similarity-Based Modeling (SBM) rotating-machine diagnosis pipeline.

### 1. Feature Extraction (Step 2)
Transforms raw, high-frequency multivariate time-series signals into a condensed 46-dimensional hand-crafted diagnostic feature vector $x(n)$ for each operational scenario.

* **Shaft Rotation Frequency Estimation via Tachometer DFT**:
  To find the mechanical rotation frequency $f(\text{rot})$, the pipeline computes the Discrete Fourier Transform (DFT) of the tachometer pulse train signal $x_{\text{tacho}}(n)$ and extracts the physical magnitude spectrum:

$$
\large X_{\text{tacho}}(k) = \sum_{n=0}^{N-1} x_{\text{tacho}}(n) \cdot e^{-i \dfrac{2\pi \cdot k \cdot n}{N}}
$$

$$
\large M_{\text{tacho}}(k) = \dfrac{2}{N} \cdot \left|X_{\text{tacho}}(k)\right| \quad \text{(or } \dfrac{4}{N} \cdot \left|X_{\text{tacho}}(k)\right| \text{ with Hanning Coherent Gain Correction)}
$$

$$
\large f(\text{rot}) = \arg\max_{f(k) \in [5.0, 120.0]} M_{\text{tacho}}(k)
$$

  * *Symbol Definitions*:
    * $x(n)$: The complete 46-dimensional hand-crafted feature vector extracted for sample scenario $n$.
    * $x_{\text{tacho}}(n)$: The normalized tachometer time-series signal amplitude at discrete sample step $n$.
    * $N$: Total number of discrete time-series samples in the signal segment ($N = 250,000$ for a 5-second recording at 50 kHz).
    * $n$: Discrete time-domain sample index ($n \in \{0, \dots, N-1\}$).
    * $k$: Discrete frequency-domain bin index ($k \in \{0, \dots, N/2\}$).
    * $X_{\text{tacho}}(k)$: Complex-valued Fourier coefficient representing the tachometer signal at frequency bin $k$.
    * $M_{\text{tacho}}(k)$: Physical amplitude spectrum value at frequency bin $k$.
    * $f(k)$: Discrete frequency in Hertz corresponding to bin $k$, calculated as $f(k) = \dfrac{k \cdot f_s}{N}$.
    * $f_s$: Signal sampling rate ($f_s = 50,000 \text{ Hz}$).
    * $f(\text{rot})$: Estimated mechanical shaft rotation frequency, constrained strictly to the physically plausible range $[5.0, 120.0] \text{ Hz}$ to bypass high-frequency electrical noise and low-frequency DC drift.

* **Spectral Harmonic Magnitudes Interpolation**:
  Extracts continuous peak amplitudes for the first 7 physical sensors at the fundamental rotation frequency and its first two harmonics:

$$
\large A(j, h) = \text{Interp}\left(h \cdot f(\text{rot}), f, M(j)\right), \quad \text{for } j \in \{0, \dots, 6\}, \; h \in \{1, 2, 3\}
$$

  * *Symbol Definitions*:
    * $j$: Physical sensor channel index. $j \in \{0, \dots, 6\}$ maps to the 6 accelerometers and 1 microphone.
    * $h$: Harmonic order. $h = 1$ is the fundamental rotation frequency $(f(\text{rot}))$, $h = 2$ is the second harmonic $(2 \cdot f(\text{rot}))$, and $h = 3$ is the third harmonic $(3 \cdot f(\text{rot}))$.
    * $A(j, h)$: Interpolated continuous spectral amplitude of the $j$-th physical sensor at harmonic frequency $h \cdot f(\text{rot})$.
    * $M(j)$: Discrete magnitude spectrum normalized to physical amplitude for sensor channel $j$.
    * $f$: Vector of discrete frequencies $f(k)$.
    * $\text{Interp}$: Linear interpolation function mapping continuous target frequencies to their corresponding amplitude spectrum values. Total features = 7 sensors $\times$ 3 harmonics = 21 features.

* **Statistical Signal Descriptors**:
  Computes three statistical indicators across all 8 sensor channels (6 accelerometers, 1 microphone, and 1 tachometer) to capture baseline drift, structural complexity, and impulsive mechanical shocks:

  * **Arithmetic Mean**:

$$
\large \mu(j) = \dfrac{1}{N} \sum_{n=0}^{N-1} x(j, n)
$$

  * **Shannon Entropy**:
    Constructs a $B$-bin histogram partition of the normalized signal amplitude. Let $c(b)$ be the sample count in the $b$-th bin. The empirical probability $p(b)$ and Shannon entropy $H(j)$ are:

$$
\large p(b) = \dfrac{c(b)}{\sum_{i=1}^{B} c(i)}, \quad \text{for } b \in \{1, \dots, B\}
$$

$$
\large H(j) = -\sum_{b=1}^{B} p(b) \cdot \log_2 p(b)
$$

  * **Fisher excess Kurtosis**:

$$
\large \kappa(j) = \dfrac{\dfrac{1}{N} \sum_{n=0}^{N-1} (x(j, n) - \mu(j))^4}{\left(\dfrac{1}{N} \sum_{n=0}^{N-1} (x(j, n) - \mu(j))^2\right)^2} - 3
$$

  * *Symbol Definitions*:
    * $j$: Sensor channel index across all 8 available channels ($j \in \{0, \dots, 7\}$).
    * $x(j, n)$: Normalized amplitude value of sensor channel $j$ at discrete time index $n$.
    * $\mu(j)$: Arithmetic mean of sensor channel $j$, representing static physical offsets.
    * $H(j)$: Shannon entropy (in bits, base 2) of sensor channel $j$, measuring signal uncertainty and mechanical complexity.
    * $B$: Number of histogram bins ($B = 100$).
    * $c(b)$: Count of samples falling into the $b$-th bin. Bins partition the locked range $[-10.0, 10.0]$ in the fixed-entropy configuration, and the dynamic range of each signal in the baseline configuration.
    * $p(b)$: Empirical probability of the signal values falling within the interval of the $b$-th bin.
    * $\kappa(j)$: Fisher excess kurtosis of sensor channel $j$, measuring distribution "peakedness" to highlight sudden mechanical impact shocks (e.g., bearing cracks).
    * $-3$: Kurtosis offset to normalize the excess kurtosis of a perfect normal Gaussian distribution to $0.0$. Total features = 8 sensors $\times$ 3 descriptors = 24 features.

### 2. SBM Class Dictionaries (Step 3)
Models the normal operational manifold of each fault class to generate highly discriminative similarity scores.

* **Weiszfeld's Geometric Median**:
  To seed each class dictionary with a highly robust normal state vector $y$ that is immune to impulse outliers, the pipeline computes the multi-dimensional geometric median of the training samples using Weiszfeld's iterative algorithm:

$$
\large y = \arg\min_{z} \sum_{i=1}^{K(c)} \Vert x(c, i) - z \Vert_2
$$

$$
\large y^{(m+1)} = \dfrac{\sum_{i=1}^{K(c)} \dfrac{x(c, i)}{\max\left(\Vert x(c, i) - y^{(m)}\Vert_2, \epsilon\right)}}{\sum_{i=1}^{K(c)} \dfrac{1}{\max\left(\Vert x(c, i) - y^{(m)}\Vert_2, \epsilon\right)}}
$$

  * *Symbol Definitions*:
    * $y$: The computed 46-dimensional geometric median vector, acting as the stable anchor seed (the very first state) of the class dictionary $D(c)$.
    * $z$: Candidate geometric median vector in the 46-dimensional feature space.
    * $x(c, i)$: The $i$-th training feature vector belonging to fault class $c$.
    * $K(c)$: Total number of training samples belonging to fault class $c$ in the training set.
    * $y^{(m)}$: Approximation of the geometric median vector at iteration $m$.
    * $\epsilon$: Regularization constant to prevent division by zero ($\epsilon = 10^{-12}$).
    * $\Vert\cdot\Vert_2$: Standard Euclidean $L_2$ norm.

* **Wegerich Similarity Function (WSF)**:
  Computes coordinate similarity using the $L_1$ norm (Manhattan distance) to yield values bounded strictly within the range $(0.0, 1.0]$:

$$
\large s(u, v) = \dfrac{1}{1 + \gamma \cdot \Vert u - v \Vert_1}
$$

$$
\large \Vert u - v \Vert_1 = \sum_{d=1}^{46} |u(d) - v(d)|
$$

  * *Symbol Definitions*:
    * $s(u, v)$: Similarity score between 46-dimensional vectors $u$ and $v$.
    * $u, v$: 46-dimensional feature vectors.
    * $u(d), v(d)$: Value of the $d$-th coordinate of vectors $u$ and $v$ respectively.
    * $\gamma$: WSF sensitivity scaling parameter, optimized to $\gamma = 0.0010$ (or $\gamma = 0.0100$ in the baseline).
    * $\Vert u - v\Vert_1$: Manhattan distance ($L_1$ norm) calculated as the sum of absolute coordinate differences.

* **Memory Matrix Construction via the Threshold Method**:
  Constructs a compact class dictionary $D(c)$ containing $M(c)$ representative states to prevent data redundancy and noise pollution:
  1. Initialize the dictionary with the robust geometric median: $D(c) = [y]$.
  2. For each candidate training vector $x(c, i) \in X(c)$, append it as a new row in $D(c)$ if and only if its similarity to all existing representative states is strictly below the threshold $\tau$:

$$
\large s(d(c, m), x(c, i)) < \tau, \quad \forall d(c, m) \in D(c)
$$

  * *Symbol Definitions*:
    * $D(c)$: Representative state dictionary matrix for class $c$, of shape $M(c) \times 46$.
    * $X(c)$: Full set of training feature vectors of class $c$.
    * $d(c, m)$: The $m$-th representative state (row vector of size 46) currently stored in $D(c)$ for $m \in \{1, \dots, M(c)\}$.
    * $\tau$: Memorization similarity threshold, optimized to $\tau = 0.85$ (or $\tau = 0.90$ in the baseline).

### 3. SBM State Estimation Module
For any given input feature vector $x(n)$ of sample $n$, SBM reconstructs a clean estimate vector $\hat{x}(n, c)$ on the manifold of class $c$:
1. **Pairwise Memory Similarity Matrix $G(c)$**:

$$
\large G(c)(i, k) = s(d(c, i), d(c, k))
$$

2. **Input-to-Memory Similarity Vector $A(c, n)$**:

$$
\large A(c, n)(k) = s(x(n), d(c, k))
$$

3. **Raw Interpolation Weights Vector $w(c, n)$**:

$$
\large w(c, n) = G(c)^{\dagger} \cdot A(c, n)
$$

4. **L1 Normalized Weights Vector $w'(c, n)$**:

$$
\large w'(c, n) = \dfrac{w(c, n)}{\Vert w(c, n)\Vert_1} = \dfrac{w(c, n)}{\sum_{k=1}^{M(c)} |w(c, n)(k)|}
$$

5. **Reconstructed Feature Vector $\hat{x}(n, c)$**:

$$
\large \hat{x}(n, c) = D(c)^T \cdot w'(c, n) = \sum_{k=1}^{M(c)} w'(c, n)(k) \cdot d(c, k)
$$

* *Symbol Definitions*:
  * $G(c)$: Symmetric pairwise similarity matrix between all representative states in $D(c)$, of shape $M(c) \times M(c)$.
  * $G(c)^{\dagger}$: Moore-Penrose pseudo-inverse of $G(c)$, ensuring stable matrix inversion even for highly collinear features.
  * $A(c, n)$: Input-to-memory similarity vector for input sample $n$ against class dictionary $D(c)$ (length $M(c)$).
  * $w(c, n)$: Raw SBM interpolation weight vector (length $M(c)$).
  * $w'(c, n)$: $L_1$ normalized interpolation weight vector, ensuring that the SBM reconstruction preserves physical signal bounds.
  * $d(c, k)$: The $k$-th representative row vector stored in the dictionary $D(c)$.
  * $\hat{x}(n, c)$: SBM reconstruction vector (length 46) of the input sample $x(n)$ on the manifold of class $c$.

### 4. Classification Ensemble (Step 4)
An ensemble approach combining the engineered features with SBM-derived vectors. The repository supports two distinct configurations from the paper:

* **SBM Model B (Default Orchestrator)**:
  Finds the best-matching fault class $c^*$ that maximizes reconstruction similarity, computes the residual error vector $e(n)$, and appends it to $x(n)$ to yield a 92-dimensional representation:

$$
\large c^* = \arg\max_{c \in \{1, \dots, 6\}} s(x(n), \hat{x}(n, c))
$$

$$
\large e(n) = x(n) - \hat{x}(n, c^*)
$$

$$
\large x_{\text{ext}}(n) = \begin{bmatrix} x(n) \\\\ e(n) \end{bmatrix}
$$

  * *Symbol Definitions*:
    * $c^*$: Index of the class dictionary that achieves the highest reconstruction similarity to the input vector $x(n)$.
    * $e(n)$: 46-dimensional SBM residual error vector, highlighting localized signal anomalies or structural deviations from the normal class manifold.
    * $x_{\text{ext}}(n)$: 92-dimensional extended feature representation fed into the Random Forest classifier.

* **Experiment 3 Configuration 3**:
  Directly appends the 6 SBM class similarity scores to the original 46 features, resulting in a compact 52-dimensional representation:

$$
\large x_{\text{ext}}(n) = \begin{bmatrix} x(n) \\\\ s(x(n), \hat{x}(n, c_1)) \\\\ \vdots \\\\ s(x(n), \hat{x}(n, c_6)) \end{bmatrix}
$$

  * *Symbol Definitions*:
    * $s(x(n), \hat{x}(n, c_r))$: Direct WSF similarity score between the input $x(n)$ and its reconstruction under class dictionary $D(c_r)$ for $c_r \in \{c_1, \dots, c_6\}$.
    * $c_1, \dots, c_6$: The 6 unique mechanical fault classes (Normal, Imbalance, Horizontal Misalignment, Vertical Misalignment, Overhang Fault, Underhang Fault).
    * $x_{\text{ext}}(n)$: 52-dimensional compact representation feeding class-similarity coordinates directly to the Random Forest ensemble.

* **Ensemble Model**:
  Employs a Random Forest Classifier with balanced class weights to mitigate the natural scarcity of Normal operation data.

