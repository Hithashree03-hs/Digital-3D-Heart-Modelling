# Digital 3D Heart Modelling

An AI-based patient-specific 3D heart reconstruction and cardiac abnormality analysis system designed to process cardiac medical imaging data, reconstruct 3D cardiac structures, extract quantitative features, and support analysis of septal defects.

> 🚧 This project is currently under active development.

---

## 📌 Project Overview

Digital 3D Heart Modelling aims to transform cardiac medical imaging data into an interactive 3D representation of the heart.

The system combines medical image preprocessing, cardiac segmentation, 3D reconstruction, feature extraction, visualization, and cardiac abnormality analysis into an integrated pipeline.

The current implementation includes cardiac reconstruction and septal defect analysis modules.

---

## 🎯 Objectives

- Process cardiac medical imaging data
- Perform cardiac structure segmentation
- Reconstruct patient-specific 3D heart models
- Extract quantitative geometric features
- Visualize reconstructed cardiac structures interactively
- Analyze cardiac abnormalities
- Support septal defect analysis
- Provide an integrated web-based interface

---

## 🧠 Key Features

### 1. Medical Image Processing

The system processes cardiac imaging data through preprocessing and preparation stages before reconstruction and analysis.

### 2. Cardiac Segmentation

Deep-learning-based segmentation is used to identify relevant cardiac structures from medical images.

The project includes segmentation components designed for extracting cardiac regions from imaging slices.

### 3. 3D Heart Reconstruction

Segmented cardiac structures are processed to generate a 3D representation of the heart.

The reconstruction pipeline includes processing of segmented data and generation of 3D anatomical structures.

### 4. Feature Extraction

The system extracts quantitative geometric features from reconstructed cardiac structures, including:

- Volume
- Surface area
- Dimensions
- Other geometric measurements

### 5. Cardiac Abnormality Analysis

The project includes modules for analyzing cardiac abnormalities based on extracted features and reconstructed structures.

### 6. Septal Defect Analysis

The current implementation includes septal defect analysis components for conditions such as:

- Atrial Septal Defect (ASD)
- Ventricular Septal Defect (VSD)

Machine-learning models are integrated into the analysis pipeline for septal defect classification/support.

### 7. Interactive 3D Visualization

The reconstructed heart model can be presented through a web-based visualization interface.

The website component provides an interactive environment for viewing the reconstructed cardiac model and associated analysis.

---

## 🏗️ System Architecture

```text
Medical Imaging Data
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Cardiac Segmentation│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 3D Reconstruction   │
└──────────┬──────────┘
           │
           ├───────────────┐
           ▼               ▼
┌─────────────────┐  ┌────────────────────┐
│ Feature         │  │ Abnormality /      │
│ Extraction      │  │ Septal Defect      │
└────────┬────────┘  │ Analysis           │
         │           └─────────┬──────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
          ┌───────────────────┐
          │ 3D Visualization  │
          │   Web Interface   │
          └───────────────────┘
