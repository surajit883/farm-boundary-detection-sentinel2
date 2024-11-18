# Sentinel-2 Farm Boundary Detection and NDVI Downloader
This project aims to detect farm boundaries from Sentinel-2 satellite imagery and analyze vegetation using NDVI (Normalized Difference Vegetation Index). The process involves downloading Sentinel-2 data, performing NDVI calculations, and utilizing a U-Net architecture for farm boundary detection.

## Project Structure

```bash
project/
│
├── data/                       # Contains input data and images
│   ├── AOI/                    # Area of Interest (AOI) data
│   └── chunk/                  # Chunks of data
│       ├── train_imgs/         # Training images
│       ├── mask_imgs/          # Mask images for training
│       └── test_imgs/          # Test images
│
├── output/                     # Output results
│   ├── predict/                # Model predictions
│   ├── model/                  # Saved model files
│   └── ndvi/                   # NDVI data or results
│
├── src/                        # Source code
│   ├── model_test_v2.py        # Script for testing model
│   ├── ndvi_downloader.py      # NDVI downloader from Sentinel-2
│   ├── train_v2.ipynb          # Jupyter notebook for training the model
│   └── unet_v2.py              # U-Net model for farm boundary detection
│
└── env/                        # Environment configuration

Installation
Prerequisites
Python 3.x
TensorFlow / Keras for deep learning model
Other dependencies as listed in the requirements.txt file
To install the required dependencies, create a virtual environment and install the necessary packages:

# Create a virtual environment

```bash
python3 -m venv env

# Activate the environment (Linux/macOS)
```bash
source env/bin/activate

# Activate the environment (Windows)
```bash
env\Scripts\activate

# Install the required dependencies
```bash
pip install -r requirements.txt


### Required Libraries
TensorFlow / Keras for deep learning model training
Numpy, Pandas for data handling
Rasterio, Geopandas for geospatial data processing
SentinelHub-Py or other libraries for Sentinel-2 data download
Matplotlib, Seaborn for visualization
Setup and Usage
## 1. Download Sentinel-2 NDVI Data


Run the ndvi_downloader.py
 script to download Sentinel-2 data for your Area of Interest (AOI).
bash

```bash
python src/ndvi_downloader.py --aoi <path_to_aoi> --output <path_to_save_data>

## 2. Prepare Training Data
Make sure that your training data (images and masks) are placed in the appropriate directories under data/chunk/train_imgs/ and data/chunk/mask_imgs/.
## 3. Train U-Net Model
The train_v2.ipynb Jupyter notebook contains the steps for training the U-Net model for farm boundary detection. You can open the notebook and run the cells to start training the model.
## 4. Testing and Evaluation
Once the model is trained, you can test its performance using the model_test_v2.py script.
bash

```bash
python src/model_test_v2.py --model <path_to_model> --test_imgs <path_to_test_images>

This will output predictions in the output/predict/ folder.
## 5. NDVI Analysis
NDVI results are stored in the output/ndvi/ folder. You can use this data for further vegetation analysis.
Model Architecture
This project uses the U-Net architecture for semantic segmentation of farm boundaries. U-Net is a convolutional network designed for fast and precise segmentation, especially in cases where the dataset is small.
Input: Sentinel-2 imagery with vegetation information.
Output: Farm boundary mask.
## U-Net Model
The U-Net model is defined in unet_v2.py. It consists of an encoder-decoder architecture with skip connections, making it suitable for image segmentation tasks.
