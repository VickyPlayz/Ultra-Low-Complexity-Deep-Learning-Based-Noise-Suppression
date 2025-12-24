# Ultra Low Complexity Deep Learning Based Noise Suppression

## Project Purpose
This project implements a **Deep Learning model designed to remove background noise from speech audio**. The goal is to achieve high-quality speech enhancement while keeping the model computationally "lightweight" (low complexity), making it potentially suitable for real-time applications or devices with limited power.

In simple terms: **Input Noisy Audio -> Model -> Output Clean Audio.**

Based on the research paper : Ultra Low Complexity Deep Learning Based Noise Suppression by Shrishti Saha Shetu, Soumitro Chakrabarty, Oliver Thiergart, Edwin Mabande
https://arxiv.org/pdf/2312.08132
## How It Works
The system uses a **two-stage hybrid approach**:
1.  **Stage 1 (CRN)**: A Convolutional Recurrent Network looks at the "magnitude" (loudness) of sound frequencies and estimates a mask to filter out noise.
2.  **Stage 2 (CNN)**: A Convolutional Neural Network refines the result by correcting the "phase" (timing/alignment) information, which is crucial for crisp audio quality.

## Project Structure
Here is how the project is organized to help you navigate:

-   **`src/`**: The main source code directory.
    -   **`models/`**: Contains the brain of the project.
        -   `crn.py`: Stage 1 model architecture.
        -   `cnn.py`: Stage 2 model architecture.
    -   **`dsp/`**: Digital Signal Processing helper functions (STFT, compression).
    -   **`features/`**: Handles preparing audio data for the model.
    -   **`utils/`**: Utility scripts (metrics like SI-SNR).
    -   `train.py`: The script used to teach (train) the model using the dataset.
    -   `inference.py`: The script to use the trained model on new audio files.
    -   `evaluate.py`: Calculates performance scores (Metrics).

-   **`data/`**: Scripts and storage for datasets.
    -   `download_voicebank.py`: Downloads the VoiceBank+DEMAND dataset used for training.

-   **`checkpoints/`**: Where the trained model weights (`.pth` files) are saved after every epoch.

## How to Use

### 1. Training the Model
To start teaching the model from scratch:
```bash
python -m src.train
```
This will run for 20 epochs and save the best models to the `checkpoints/` folder.

### 2. Testing (Inference)
To remove noise from your own audio file:
```bash
python -m src.inference -i input_noisy.wav -o output_clean.wav -m checkpoints/model_epoch_20.pth
```

### 3. Evaluation
To see how well the model performs on the test set:
```bash
python -m src.evaluate --model checkpoints/model_epoch_20.pth
```

## Metrics
We evaluate the model using **SI-SNR (Scale-Invariant Signal-to-Noise Ratio)**.
-   **Where to find them**: Run the `src/evaluate.py` script. It prints the average SI-SNR score.
-   **Current Result**: The model achieves approximately **17.55 dB** SI-SNR on the test set, indicating significant noise reduction.

## Requirements
-   Python 3.8+
-   PyTorch
-   SoundFile
-   NumPy
-   `pesq` and `pystoi` (Optional, for additional metrics)
