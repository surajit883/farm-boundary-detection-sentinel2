# Sentinel-2 Farm Boundary Detection and NDVI Downloader
This project aims to detect farm boundaries from Sentinel-2 satellite imagery and analyze vegetation using NDVI (Normalized Difference Vegetation Index). The process involves downloading Sentinel-2 data, performing NDVI calculations, and utilizing a U-Net architecture for farm boundary detection.

## Project Structure

```bash
📂 farm-boundary-detection-sentinel2
├── 📂 data
│   ├── 📂 AOI                  # Area of Interest for NDVI downloader (shapefiles, geojson, etc.)
│   └── 📂 chunk                # Chunk-related data for processing
│       ├── 📂 mask_imgs        # Mask images for chunking
│       ├── 📂 train_imgs       # Training images for the model
│       └── 📂 test_imgs        # Test images for model evaluation
├── 📂 output                   # Output data and results
│   ├── 📂 ndvi                 # NDVI pipeline results (e.g., .tif files)
│   ├── 📂 model                # Trained model files (e.g., U-Net weights)
│   └── 📂 predict              # Prediction outputs
├── 📂 src                      # Source code for the project
│   ├── 📂 __init__.py
│   ├── 📂 model_test_v2.py      # Model testing script (Evaluate U-Net model)
│   ├── 📂 ndvi_downloader.py    # Script for downloading NDVI data
│   ├── 📂 train_v2.ipynb        # Jupyter notebook for model training
│   ├── 📂 unet_v2.py            # U-Net model implementation in PyTorch
│   ├── 📂 unet-v2.ipynb          # Full script wtih model create ,train and test and output
├── 📂 requirements.txt         # List of Python dependencies
└── README.md                # Environment configuration
```

Installation
Prerequisites
Python 3.x
TensorFlow / Keras for deep learning model
Other dependencies as listed in the requirements.txt file
To install the required dependencies, create a virtual environment and install the necessary packages:

### Create a virtual environment

```bash
python3 -m venv env
```
### Activate the environment (Linux/macOS)
```bash
source env/bin/activate
```
### Activate the environment (Windows)
```bash
env\Scripts\activate
```
# Install the required dependencies
```bash
pip install -r requirements.txt
```

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
```
## 2. Farm boundary segmetation 
Make sure that your training data (images and masks) are placed in the appropriate directories under data/chunk/train_imgs/ and data/chunk/mask_imgs/.
## 3. Train U-Net Model
The train_v2.ipynb Jupyter notebook contains the steps for training the U-Net model for farm boundary detection. You can open the notebook and run the cells to start training the model.
## 4. Testing and Evaluation
Once the model is trained, you can test its performance using the model_test_v2.py script.
bash

```bash
python src/model_test_v2.py --model <path_to_model> --test_imgs <path_to_test_images>
```
This will output predictions in the output/predict/ folder.
## 5. NDVI Analysis
NDVI results are stored in the output/ndvi/ folder. You can use this data for further vegetation analysis.
Model Architecture
This project uses the U-Net architecture for semantic segmentation of farm boundaries. U-Net is a convolutional network designed for fast and precise segmentation, especially in cases where the dataset is small.
Input: Sentinel-2 imagery with vegetation information.
Output: Farm boundary mask.

After running ndvi downloader scritp output will store in output/ndvi/

## U-Net Model
The U-Net model is defined in unet_v2.py. It consists of an encoder-decoder architecture with skip connections, making it suitable for image segmentation tasks.

#### Model Architecture
This project uses the U-Net architecture for semantic segmentation of farm boundaries. U-Net is a convolutional network designed for fast and precise segmentation, especially in cases where the dataset is small.

#### Input: Sentinel-2 imagery with vegetation information(blue,green,red,nir).
Output: Farm boundary mask.

## File Descriptions
data/: Stores raw data, including training, testing images, and masks.
output/: Stores results, including model predictions, NDVI results, and saved models.
src/:
ndvi_downloader.py: Script to download Sentinel-2 NDVI data.
unet_v2.py: U-Net model architecture for farm boundary detection.
train_v2.ipynb: Jupyter notebook for model training.
model_test_v2.py: Script to test and evaluate the trained model.
## Example Output
After running the model, the predicted farm boundaries are stored in output/predict/.

[Download the PDF with predicted images](./Detection%20Results_%20UNet%20Farm%20Boundary%20Segmentation.pdf)


For more details and visualizations of all tested images, please refer to the [unet-v2.ipynb](src/unet-v2.ipynb) script.



