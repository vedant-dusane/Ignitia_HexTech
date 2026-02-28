# Ignitia_HexTech

Team HexTech (Ignitia Hackathon – Offroad Desert Segmentation Challenge)
Team Members:
	Vedant Dusane
	Arnav Kumar
	Saqlain Abidi

1. Project Overview
   
	This project presents a semantic segmentation framework for offroad desert terrain, developed for the Ignitia Hackathon.
	
	Our solution is centered around:DeepLabV3+ with EfficientNet‑B4 as the encoder backbone.
	
	The primary architectural emphasis of this project is the integration of EfficientNet‑B4 to enhance feature extraction quality while maintaining parameter efficiency. The objective was to achieve stable convergence, strong generalization, and competitive Intersection over Union (IoU) performance within constrained training time.The pipeline is modular, reproducible, and structured for clean experimentation.

3. Dataset Acquisition
	
	The official dataset can be downloaded from: https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert?utm_source=hackathon&utm_medium=instructions&utm_campaign=Ignitia
	
	After downloading:

		1.Extract the Training Dataset
		2.Extract the Testing Dataset


5. Required Directory Structure
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

6. Training Procedure

Navigate to: Project/Offroad_Segmentation_Scripts/DeepLabV3Plus/scripts
		
Run: python train_deeplab.py
			
This script:
			
			1.Initializes DeepLabV3+ with EfficientNet‑B4 encoder
			
			2.Loads training and validation datasets
			
			3.Trains for said number of epochs
			
			4.Monitors IoU, Dice, Accuracy, and Loss
			
			5.Saves the best EfficientNet‑B4 checkpoint (based on Validation IoU)
			
			6.Generates performance graphs
			
			7.Stores qualitative segmentation outputs

5. Testing Procedure
	
	After training: python test_deeplab.py
	
	This loads the best EfficientNet‑B4 checkpoint and performs inference on test images. Predictions are stored in: result/test_prediction
		
6. Architecture: DeepLabV3+ with EfficientNet‑B4
   
	6.1 EfficientNet‑B4 as Encoder Backbone

   EfficientNet‑B4 forms the core of our model architecture.

   Key properties:

	   1. Compound scaling of depth, width, and resolution
	
	   2.Strong multi‑scale feature representation
	
	   3.High representational efficiency
	
	   4.Improved feature richness compared to lightweight encoders
	
	   5.Its hierarchical structure enables robust extraction of desert terrain features such as texture variation, boundary transitions, and region consistency.

   6.2 DeepLabV3+ Decoder
	
	The DeepLabV3+ framework complements EfficientNet‑B4 through:

  		1.Atrous Spatial Pyramid Pooling (ASPP)

		2.Multi‑scale context aggregation

		3.Boundary refinement via decoder upsampling

	This combination allows EfficientNet‑B4 features to be leveraged effectively for dense pixel‑wise prediction.

8. Training Methodology
   
	The EfficientNet‑B4 pipeline incorporates several performance‑oriented engineering decisions to maximize segmentation accuracy and convergence stability:
	

		1.Hybrid Dice + Focal Loss: Combines overlap optimization (Dice) with hard‑example emphasis (Focal) to handle class imbalance and improve boundary precision.

		2.Mixed Precision Training (AMP): Reduces memory usage and increases training throughput without compromising numerical stability.

		3.AdamW Optimizer: Decouples weight decay from gradient updates, improving generalization and convergence behavior.

		4.Cosine Annealing Warm Restarts Scheduler: Periodically resets the learning rate to escape shallow minima and encourage smoother convergence.

		5.Channels‑Last Memory Format: Optimizes tensor layout for improved GPU throughput and memory efficiency.

		6.Gradient Clipping: Prevents exploding gradients and stabilizes training during early epochs.

		7.Validation‑Driven Checkpointing: Saves the model strictly based on highest validation IoU to ensure optimal generalization performance.

		8.Test‑Time Augmentation (Flip Averaging): Improves inference robustness by averaging predictions across transformed inputs.

		9.Automated Metric Logging & Visualization: Generates loss and metric curves for transparent monitoring of convergence behavior.

10. Final Training Statistics

    EfficientNet‑B4 – 25 Epoch Run. After full training completion, the final EfficientNet‑B4 model statistics are summarized below.
    
	8.1 Final Numerical Results
   
		TRAINING RESULTS (Epoch 25/25)
		==================================================
		
		Final Metrics:
		  Final Train Loss:     0.2793
		  Final Val Loss:       0.2637
		  Final Train IoU:      0.5953
		  Final Val IoU:        0.5297
		  Final Train Dice:     0.7196
		  Final Val Dice:       0.7053
		  Final Train Accuracy: 0.8604
		  Final Val Accuracy:   0.8621
		==================================================
		
		Best Results:
		  Best Val IoU:      0.5300 (Epoch 24)
		  Best Val Dice:     0.7062 (Epoch 24)
		  Best Val Accuracy: 0.8621 (Epoch 25)
		  Lowest Val Loss:   0.2637 (Epoch 25)
		==================================================

	8.2 Generated Performance Curves
    
	All graphs are automatically generated during EfficientNet‑B4 training and stored in: result/train_stats/
		
		1.All Metrics Curves
    
	<img width="1200" height="1000" alt="all_metrics_curves" src="https://github.com/user-attachments/assets/af4ec89f-f0a3-4f7b-8655-34a5e97f3f1b" />

		The model demonstrates stable convergence, gradual performance improvement, and strong generalization with minimal overfitting.
		
		2.Dice Curves
	<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/27953f6b-b9ca-4c6d-8b1f-a320dd94c68b" />

    	This shows the steady upward trend in validation Dice indicates improved overlap quality between predicted masks and ground truth. The small gap between train and validation curves suggests controlled overfitting and good generalization.
		
		3.IoU Curve
	<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/a4a94813-97cf-436b-9e2a-5d922b46fc38" />
		
		The consistent improvement in validation IoU confirms effective feature learning. The moderate gap between train and validation IoU is expected in segmentation tasks and does not indicate instability.
		
		4.Training Curve
	<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/9192f72a-061a-4f43-b922-6f7d412c4f2f" />
		
		Shows the model exhibits stable convergence behavior, smooth optimization dynamics, and no signs of training collapse or erratic learning rate behavior.

		
	8.3 Interpretation
    
		1.IoU is the principal segmentation performance metric.
		
		2.Dice reflects overlap quality between predicted and ground truth masks.
		
		3.Loss curves confirm stable optimization behavior.
		
		4.Accuracy provides complementary pixel‑level insight.
		
		5.The final deployed model corresponds to the highest validation IoU achieved by EfficientNet‑B4 during training.

12. Conclusion
    
	Team HexTech implemented a structured DeepLabV3+ segmentation pipeline centered on EfficientNet‑B4 as the encoder backbone.
	
	The model demonstrated:

		1.Stable and consistent convergence
		
		2.Progressive validation IoU improvement
		
		3.Strong qualitative segmentation outputs
		
		4.Reliable terrain feature discrimination

	EfficientNet‑B4 proved effective in capturing complex desert terrain patterns while maintaining training stability under hackathon constraints.

Team HexTech
Ignitia Hackathon Submission
Vedant Dusane • Arnav Kumar • Saqlain Abidi
