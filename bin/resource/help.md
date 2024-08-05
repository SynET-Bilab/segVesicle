# SegVesicle User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Overall Workflow](#overall-workflow)
3. [Component Overview](#component-overview)
   - [Data Preprocessing](#data-preprocessing)
   - [Inference and Manual Annotation](#inference-and-manual-annotation)
   - [Retrain Model](#retrain-model)
   - [Results Analysis](#results-analysis)
4. [Frequently Asked Questions](#frequently-asked-questions)
5. [References](#references)

## Introduction
`SegVesicle` is a tool designed for the neuroscience field, specifically for identifying and segmenting vesicles in microscopy images. This tool combines advanced image processing techniques with deep learning algorithms to provide an efficient and accurate automated solution.

## Overall Workflow
The workflow of `SegVesicle` can be divided into the following steps:

1. **Data Preprocessing**: Place all files to be processed in a single folder. Apply deconvolution and correction using IsoNet for image correction.
2. **Inference and Manual Annotation**: Use the trained model to predict on new images and manually correct annotations as needed.
3. **Retrain Model**: Based on the manual corrections, retrain the model to improve its accuracy.
4. **Results Analysis**: Analyze the results of the segmentation.

## Component Overview

### Data Preprocessing
Data preprocessing is the first step in the workflow and includes the following parts:
- **File Organization**: Place all files to be processed in their respective subfolders within a main folder. Each subfolder (e.g., `p545`, `p565`, `pp1134`) should contain the corresponding `.mrc` or `.rec` files. The main folder should also contain a `segVesicle.batch` file that stores information about the images to be segmented.
    If the `segVesicle.batch` file does not exist, you can create it using the command line:

    `bash
    ls */*-bin4-wbp.rec > segVesicle.batch
    `
    
    Alternatively, you can create it using the "Create segVesicle.batch" button in the top right corner of the SegVesicle software.

    The file organization should be as follows:
    ```
    any_folder/
    ├── p545/
    │   ├── p545*.mrc
    │   ├── p545*.rec
    │   └── ...
    ├── p565/
    │   ├── p565*.mrc
    │   ├── p565*.rec
    │   └── ...
    ├── pp1134/
    │   ├── p565*.mrc
    │   ├── p565*.rec
    │   └── ...
    ├── ...
    └── segVesicle.batch
    ```

- **Batch File Creation**: Use the command `ls */*-bin4-wbp.rec > segVesicle.batch` to create a file that stores information about the images to be segmented.
- **Image Correction**: Apply IsoNet's deconvolution and correction for image correction.

### Inference and Manual Annotation
Inference and manual annotation include:
- **Model Inference**: Use the trained model to predict on new images and generate vesicle segmentation results.
- **Manual Correction**: Manually correct annotations as needed to ensure accuracy.

### Retrain Model
Retraining the model includes:
- **Model Update**: Incorporate the manual corrections into the training dataset.
- **Retraining**: Retrain the model with the updated dataset to improve its accuracy and performance.

### Results Analysis
Results analysis includes:
- **Result Visualization**: Visualize the segmentation results for expert analysis and evaluation.
- **Performance Evaluation**: Evaluate the model performance using metrics such as accuracy, recall, F1 score, etc.

## Frequently Asked Questions
1. **Sometimes shortcut keys fail to register. How to resolve this?**
   - Select a few blank spots and rapidly click the small trash icon multiple times. This usually helps in successfully registering the shortcut keys, although the exact reason is unknown.


## References
- [1] Reference A
- [2] Reference B

---

If you have any questions or need further assistance, please contact the technical support team.
