Multi-Stage Image Deblurring

PyTorch implementation of a 3-stage U-Net-based image deblurring network that combines Supervised Attention Modules (SAM) with Sobel edge guidance and Cross-Stage Feature Fusion (CSFF).  The script is designed for Kaggle/Colab-style execution and includes dataset loading, training, checkpointing, and simple qualitative visualization on held-out samples. 
Features
	•	GoPro blur/sharp paired dataset loader (expects Kaggle dataset layout under  DBlur/Gopro/... ). 
	•	3-stage deblurring architecture:
	•	 UNetStage : per-stage encoder–decoder U-Net producing an RGB output + intermediate features. 
	•	 SAM : supervised attention using Sobel edges computed from the previous stage output. 
	•	 CSFF : 1×1 fusion of intermediate features across stages. 
	•	Multi-stage L1 training loss with stage weights  0.5, 0.5, 1.0 . 
	•	CosineAnnealing learning-rate scheduler and best-model saving. 
	•	Inference helper  deblur_image()  that resizes input to 256×256 for prediction and restores original dimensions. 
Project structure
This repo currently consists of a single training script:
	•	 v2_sam_-_csff.py : end-to-end pipeline (install deps, dataset, model, training loop, and visualization). 
Requirements
The script installs dependencies inline (Kaggle/Colab style):  torch ,  torchvision ,  pillow ,  tqdm ,  matplotlib .  A CUDA GPU is recommended; the script selects  cuda  if available. 
Dataset
By default, the code uses the Kaggle dataset “A curated list of image deblurring datasets” and expects the GoPro subset at: 
	•	 /kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/train/blur 
	•	 /kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/train/sharp 
	•	 /kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/test/blur 
	•	 /kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/test/sharp   
If running locally, update the  path  variable in the script accordingly. 
How to run
Kaggle / Colab
	1.	Add the Kaggle dataset input (or ensure the path matches your environment). 
	2.	Run the script top-to-bottom (it includes  pip install ...  and Kaggle dataset download helper code). 
Local (recommended cleanup)
Because the script contains notebook-style cells and Kaggle-specific lines (e.g.,  !pip install ... ), a clean local run typically requires: 
	•	Removing shell-magics ( !pip ... ). 
	•	Installing requirements via  pip install -r requirements.txt  (create one from the imports). 
	•	Replacing Kaggle paths with local filesystem paths. 
Training details
Default settings in the script: 
	•	Input training size: 256×256 (images are resized in the dataset class). 
	•	Batch size: 8, epochs: 30. 
	•	Optimizer: Adam ( lr=2e-4 ). 
	•	Scheduler: CosineAnnealingLR with  T_max=50 . 
	•	Loss: multi-stage weighted L1. 
Checkpoints/outputs: 
	•	Best model saved to  multistage_deblur_best.pth . 
	•	Additional checkpoints saved every 10 epochs as  checkpoint_epoch_10.pth , etc. 
Visualization / inference
After training, the script runs a small qualitative test on fixed indices and saves a figure: 
	•	 deblur_results.png  (3 columns: blurry / ground truth / deblurred). 
For single-image inference, use the helper: 
	•	 deblur_image(model, image_path, device)  returns a PIL image resized back to the original input resolution. 
Notes / known issues
	•	This file is an auto-export from a notebook and may need formatting fixes (indentation/markdown remnants like the standalone  model  line) before production use. 
	•	The current stage execution calls  stage2  and  stage3  more than once per forward pass (once to get features and again to get outputs), which is functional but inefficient; refactoring could compute each stage once and reuse both outputs and features. 
License
Add a license file if you plan to publish this repo (e.g., MIT/Apache-2.0). 
If you want, the intended audience can be clarified (course project vs research replication), and the README can be adjusted to include exact command lines and a cleaned “local run” version of the script.
