# Ignitia_HexTech

Team HexTech (Ignitia Hackathon – Offroad Desert Segmentation Challenge)
Team Members:
	Vedant Dusane
	Arnav Kumar
	Saqlain Abidi

1. Project Overview
	This project presents a semantic segmentation framework for offroad desert terrain, developed for the Ignitia Hackathon.
	Our solution is centered around:	
	DeepLabV3+ with EfficientNet‑B4 as the encoder backbone
		The primary architectural emphasis of this project is the integration of EfficientNet‑B4 to enhance feature extraction quality while maintaining parameter efficiency. The objective was to achieve stable convergence, strong generalization, and competitive Intersection over Union (IoU) performance within constrained training time.
		The pipeline is modular, reproducible, and structured for clean experimentation.

2. Dataset Acquisition
	The official dataset can be downloaded from: https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert?utm_source=hackathon&utm_medium=instructions&utm_campaign=Ignitia
	After downloading:
		Extract the Training Dataset
		Extract the Testing Dataset

3. Required Directory Structure
		After extraction, the directory must be structured as follows:

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

	The dataset directories must be placed alongside Offroad_Segmentation_Scripts inside the Project folder. Incorrect placement will prevent the EfficientNet‑B4 training pipeline from locating the data.

5. Training Procedure
		Navigate to: Project/Offroad_Segmentation_Scripts/DeepLabV3Plus/scripts
		Run: python train_deeplab.py
			This script:
					Initializes DeepLabV3+ with EfficientNet‑B4 encoder
					Loads training and validation datasets
					Trains for 25 epochs
					Monitors IoU, Dice, Accuracy, and Loss
					Saves the best EfficientNet‑B4 checkpoint (based on Validation IoU)
					Generates performance graphs
					Stores qualitative segmentation outputs

5. Testing Procedure
	After training: python test_deeplab.py
		This loads the best EfficientNet‑B4 checkpoint and performs inference on test images. Predictions are stored in: result/test_prediction
		
6. Architecture: DeepLabV3+ with EfficientNet‑B4
		6.1 EfficientNet‑B4 as Encoder Backbone
				EfficientNet‑B4 forms the core of our model architecture.
				Key properties:
   				Compound scaling of depth, width, and resolution
					Strong multi‑scale feature representation
					High representational efficiency
					Improved feature richness compared to lightweight encoders
					Its hierarchical structure enables robust extraction of desert terrain features such as texture variation, boundary transitions, and region consistency.
		6.2 DeepLabV3+ Decoder
				The DeepLabV3+ framework complements EfficientNet‑B4 through:
					Atrous Spatial Pyramid Pooling (ASPP)
					Multi‑scale context aggregation
					Boundary refinement via decoder upsampling
					This combination allows EfficientNet‑B4 features to be leveraged effectively for dense pixel‑wise prediction.

7. Training Methodology
		The training pipeline was designed specifically to maximize EfficientNet‑B4 performance:
				Supervised pixel‑wise segmentation
				Cross‑entropy based optimization
				Validation‑driven checkpointing
				IoU as the primary selection metric
				Dice coefficient as secondary overlap metric
				Monitoring convergence stability through loss curves
				Model selection strictly relied on the highest validation IoU achieved during the 25‑epoch run.

8. Final Training Statistics
		EfficientNet‑B4 – 25 Epoch Run. After full training completion, the final EfficientNet‑B4 model statistics are summarized below.
		8.1 Final Numerical Results
============================================================
FINAL TRAINING SUMMARY (EfficientNet‑B4 – Epoch 25/25)
============================================================

Final Train Loss:        ______
Final Validation Loss:   ______
Final Validation IoU:    ______
Final Validation Dice:   ______
Final Validation Acc:    ______

Best IoU Achieved:       ______
Best Epoch:              ______
============================================================

8.2 Generated Performance Curves
		All graphs are automatically generated during EfficientNet‑B4 training and stored in: result/train_stats/
		.
		Training vs Validation Loss Curve
		![Loss Curve](result/train_stats/loss_curve.png)
		Demonstrates convergence stability and generalization behavior of EfficientNet‑B4.
		.
		Validation IoU Curve
		![IoU Curve](result/train_stats/iou_curve.png)
		Primary performance curve used for selecting the best EfficientNet‑B4 checkpoint.
		.
		Validation Dice Curve
		![Dice Curve](result/train_stats/dice_curve.png)
		Reflects segmentation overlap consistency across epochs.
		.
		Validation Accuracy Curve
		![Accuracy Curve](result/train_stats/accuracy_curve.png)
		Tracks pixel‑wise prediction accuracy throughout training
		.
	8.3 Interpretation
		IoU is the principal segmentation performance metric.
		Dice reflects overlap quality between predicted and ground truth masks.
		Loss curves confirm stable optimization behavior.
		Accuracy provides complementary pixel‑level insight.
		The final deployed model corresponds to the highest validation IoU achieved by EfficientNet‑B4 during training.
		
9. Conclusion
		Team HexTech implemented a structured DeepLabV3+ segmentation pipeline centered on EfficientNet‑B4 as the encoder backbone.
		The model demonstrated:
			Stable and consistent convergence
			Progressive validation IoU improvement
			Strong qualitative segmentation outputs
			Reliable terrain feature discrimination
			EfficientNet‑B4 proved effective in capturing complex desert terrain patterns while maintaining training stability under hackathon constraints.

Team HexTech
Ignitia Hackathon Submission
Vedant Dusane • Arnav Kumar • Saqlain Abidi
