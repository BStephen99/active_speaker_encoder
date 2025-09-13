# PepperMint Role — Short-Term Encoder  

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)]() 
[![Conda](https://img.shields.io/badge/conda-env-green.svg)]()  
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()  

This repository contains the code used to train the **short-term encoder** for the baseline model of the **PepperMint Role dataset**.  

The implementation is adapted from:  
- [Active Speakers with Context](https://github.com/fuankarion/active-speakers-context)  
- [Temporal Shift Module](https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py)  
- Process inspired by [GraVi-T encoder](https://github.com/IntelLabs/GraVi-T)  

---

## 📑 Table of Contents  

- [Setup](#-setup)  
- [Data Preparation](#-data-preparation)  
  - [PepperMint Role dataset](#peppermint-role-dataset)  
  - [AVA dataset](#ava-dataset)  
  - [WASD dataset](#wasd-dataset)  
  - [Preprocessing scripts](#preprocessing-scripts)  
- [Training](#-training)  
- [Generating Embeddings](#-generating-embeddings)  

---

## ⚙️ Setup  

Clone the repository:  

```bash
git clone https://github.com/BStephen99/active_speaker_encoder.git
cd active-speakers-context
```

Create a Conda environment:  

```bash
conda env create -f environment.yml -n ASD
```

---

## 📂 Data Preparation  

We used **three datasets** in our experiments:  

- **PepperMint Role** (ours)  
- **AVA ActiveSpeaker Dataset**  
- **WASD**  

### PepperMint Role dataset  
- Preprocessed **face crops** and **audio clips** are available here:  
  [PepperMint Repository](https://repository.ortolang.fr/api/content/peppermint/head/)  
  - `AudioClips/`  
  - `FaceCrops/`  

### AVA dataset  
- Download from: [AVA Dataset](https://github.com/cvdfoundation/ava-dataset)  

### WASD dataset  
- Download from: [WASD](https://github.com/Tiago-Roxo/WASD)  

### Preprocessing scripts  

Extract audio tracks:  

```bash
./data/extract_audio_tracks.py
./data/slice_audio_tracks.py
```  

Extract face crops:  

```bash
./data/extract_face_crops_time.py
```

---

## 🏋️ Training  

1. Set paths for CSV files, face crops, and audio clips in:  

   ```text
   ./core/config.py
   ```

2. Define training/validation sets (`train_names`, `val_names`) and model name in:  

   ```text
   ./STE_train.py
   ```

3. Run training:  

   ```bash
   python3 STE_train.py 11 0
   ```

---

## 🎯 Generating Embeddings  

1. In the config file, set the **mode** (dataset to process):  
   - `pepper_back`  
   - `pepper_high`  
   - `ava_train`, `ava_test`  
   - `wasd_train`, `wasd_test`  

2. In `STE_forward.py`, specify:  
   - `model_weights` → path to trained model  
   - `target_directory` → output folder  

3. Run embedding extraction:  

   ```bash
   python3 STE_forward.py 11 0
   ```
