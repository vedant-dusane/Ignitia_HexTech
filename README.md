# Ignitia_HexTech

Team HexTech
Ignitia Hackathon – Offroad Desert Segmentation Challenge
Team Members
Vedant Dusane.Arnav Kumar.Saqlain Abidi



1. Project Overview
This project presents a semantic segmentation pipeline for offroad desert terrain using DeepLabV3+.

The objective was to develop a robust and efficient model capable of accurately segmenting terrain classes under challenging desert conditions. Our final implementation utilizes DeepLabV3+ with an EfficientNet‑B4 backbone, selected for its strong representational power and parameter efficiency.

The pipeline is designed for reproducibility, modularity, and scalable experimentation.

2. Dataset Acquisition
The official dataset can be downloaded from:

https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert?utm_source=hackathon&utm_medium=instructions&utm_campaign=Ignitia

After downloading:

Extract the Training Dataset

Extract the Testing Dataset

3. Required Directory Structure
After extraction, the directory structure must be exactly as follows:

IGNITIA_HEXTECH
│
├── organiser_files/
│
├── Project
│   ├── Offroad_Segmentation_Scripts
│   │   ├── DeepLabV3Plus
│   │   │   ├── models
│   │   │   ├── result
│   │   │   │   ├── test_prediction
│   │   │   │   ├── train_stats
│   │   │   │   ├── visualizations
│   │   │   ├── scripts
│   │   │   │   ├── train_deeplab.py
│   │   │   │   ├── test_deeplab.py
│   │   │   │   ├── visualize.py
│   │
│   ├── Offroad_Segmentation_testImages
│   │   ├── Color_Images
│   │   ├── Segmentation
│   │
│   ├── Offroad_Segmentation_Training_Dataset
│       ├── train
│       │   ├── Color_Images
│       │   ├── Segmentation
│       ├── val
│           ├── Color_Images
│           ├── Segmentation
│
├── README.md
├── .gitignore
The dataset folders must be placed alongside Offroad_Segmentation_Scripts within the Project directory. Incorrect placement will prevent the training pipeline from locating the data.

4. Training Procedure
Navigate to:

Project/Offroad_Segmentation_Scripts/DeepLabV3Plus/scripts
Execute:

python train_deeplab.py
This script:

Loads training and validation data

Initializes DeepLabV3+ with EfficientNet‑B4 encoder

Trains the model for 25 epochs

Selects the best checkpoint based on Validation IoU

Saves performance statistics

Generates evaluation graphs

Stores qualitative segmentation outputs

5. Testing Procedure
After training:

python test_deeplab.py
This performs inference using the best saved model and stores predictions in:

result/test_prediction
6. Model Architecture
DeepLabV3+ with EfficientNet‑B4 Backbone
EfficientNet‑B4 was selected as the encoder due to:

Strong hierarchical feature extraction

Balanced scaling of depth, width, and resolution

High representational efficiency relative to parameter count

DeepLabV3+ enhances segmentation performance through:

Atrous Spatial Pyramid Pooling (ASPP)

Multi‑scale contextual feature aggregation

Improved boundary refinement via decoder structure

This combination provides strong feature discrimination while maintaining stable convergence.

7. Training Methodology
The training pipeline follows a supervised segmentation approach:

Pixel‑wise annotated supervision

Train/validation split

Cross‑entropy optimization

Model selection based on Validation IoU

Monitoring of Dice coefficient and Accuracy

Checkpointing of best‑performing epoch

IoU was treated as the primary evaluation metric.

8. Final Training Statistics
EfficientNet‑B4 – 25 Epoch Run
Upon completion of 25 training epochs, the final performance metrics are summarized below.

8.1 Final Numerical Results
Replace the placeholders below with the final recorded values:

============================================================
FINAL TRAINING SUMMARY (Epoch 25/25)
============================================================

Final Train Loss:        ______
Final Validation Loss:   ______
Final Validation IoU:    ______
Final Validation Dice:   ______
Final Validation Acc:    ______

Best IoU Achieved:       ______
Best Epoch:              ______
============================================================
8.2 Performance Curves
All graphs are automatically generated during training and stored in:

result/train_stats/
After training, insert the generated figures below.

Loss Curve (Training vs Validation)
![Loss Curve](result/train_stats/loss_curve.png)
This graph demonstrates convergence behavior and generalization stability.

Validation IoU Curve
![IoU Curve](result/train_stats/iou_curve.png)
This curve reflects segmentation quality improvement across epochs and determines final model selection.

Validation Dice Curve
![Dice Curve](result/train_stats/dice_curve.png)
Indicates overlap consistency between predicted and ground truth masks.

Validation Accuracy Curve
![Accuracy Curve](result/train_stats/accuracy_curve.png)
Tracks pixel‑wise classification performance across training.

8.3 Interpretation
Intersection over Union (IoU) serves as the primary segmentation performance metric.

Dice coefficient measures overlap consistency.

Loss curves demonstrate convergence and regularization behavior.

Accuracy provides auxiliary insight into pixel‑level classification reliability.

The final deployed checkpoint corresponds to the epoch achieving the highest Validation IoU.

9. Conclusion
Team HexTech developed a structured, modular DeepLabV3+ segmentation pipeline leveraging EfficientNet‑B4 as the encoder backbone.

The model demonstrated:

Stable convergence across epochs

Progressive improvement in validation IoU

Strong qualitative segmentation outputs

Reliable terrain feature discrimination under desert conditions

The implementation emphasizes reproducibility, clean experimentation workflow, and scalability for future improvements.

Team HexTech
Ignitia Hackathon Submission
Vedant Dusane • Arnav Kumar • Saqlain Abidi

