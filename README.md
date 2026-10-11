# 3D Spine Metastasis Segmentation (Work in Progress)

This repository contains the ongoing development of an end-to-end Deep Learning pipeline for segmenting spinal metastases from 3D CT scans using a 3D U-Net architecture in PyTorch. 

Currently, the project focuses on **Medical Image Preprocessing**—handling raw DICOM series, synchronizing spatial metadata with DICOM-SEG annotations, and preparing memory-efficient normalized 3D patches for training.


## Core Preprocessing Pipeline

Medical imaging requires precise spatial tracking. A neural network cannot simply consume raw 512x512x1000 NIfTI files. This project implements a custom preprocessing pipeline:

1. **DICOM-SEG Mapping:** Maps 2D segmentation frames to the correct Z-axis slice of the CT volume using physical `ImagePositionPatient` metadata.
2. **Otsu Body Cropping:** Temporarily clips HU values to reliably detect the patient's body contour using Otsu thresholding, extracts the largest connected component (ignoring the CT table/noise), and crops the 3D volume to a tight bounding box (saving significant RAM).
3. **Isotropic Resampling:** Resamples the cropped CT to `(1.0, 1.0, 1.0) mm` spacing. Crucially, uses **Linear Interpolation** for the CT scan and **Nearest-Neighbor** for the mask to prevent destroying integer class labels.
4. **HU Windowing:** Hard clips Hounsfield Units to `[-200, 1000]` to isolate bone and surrounding tissue, removing noise from air and metallic implants.
5. **Min-Max Normalization:** Scales the clipped window to `[0.0, 1.0]` (Float32) to prevent gradient explosion and align with neural network activation functions.
6. **Metadata Preservation:** Saves spatial parameters (Original Spacing, Origin, Direction, and Crop Coordinates) into a `.json` file. This is mandatory for the future inference stage to accurately map the U-Net's prediction back onto the original raw CT scan.

## Repository Structure

```text
├── src/
│   ├── dcm_to_nifti.py      # Handles spatial alignment of DICOM-SEG
│   ├── napari_view.py       # 3D volume and mask visualization
│   └── preprocessing.py     # Core Otsu, Resampling, and Normalization logic
├── README.md
└── requirements.txt