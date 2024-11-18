import torch
import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt  



model = torch.jit.load('/home/surajit/d/PROJECTS/farmdata/model/model_scripted.pt', map_location='cpu')
model.eval()

with open("/home/surajit/d/PROJECTS/escortscubota_assigment/chunk/test_imgs/test_images.txt", "r") as file:
    file_content = file.readlines()

unique_names = {os.path.basename(line.split(',')[0].strip(" '()")) for line in file_content}

def read_raster(file):
    with rio.open(file) as src:
        red = src.read(4)  
        green = src.read(3)
        blue = src.read(2)
        nir = src.read(8)
        profile = src.profile

        rgb_image = np.stack((blue, green, red, nir), axis=-1)
        imgdata = (rgb_image / rgb_image.max() * 255).astype(np.uint8)

    image = imgdata[:256, :256, :] / 255  
    image = np.transpose(image, (2, 0, 1))  

    # Convert to PyTorch tensor
    image_tensor = torch.tensor(image, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)  
    return image_tensor, profile

for name in unique_names:
    file = f'/home/surajit/d/PROJECTS/escortscubota_assigment/chunk/train_imgs/{name}'

    image_data , profile= read_raster(file)

    with torch.no_grad():
        output = model(image_data)

    output_image = output.squeeze().cpu().numpy()  
    output_image_filter = np.where(output_image >= 0.5, 1, 0)

    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes[0].imshow(image_data.squeeze(0).permute(1, 2, 0).cpu().numpy())  
    # axes[0].set_title('Original Image')
    # axes[0].axis('off')

    # axes[1].imshow(output_image, cmap='gray') 
    # axes[1].set_title('Model Prediction')
    # axes[1].axis('off')
    # plt.show()

    output_filename = f'/home/surajit/d/PROJECTS/escortscubota_assigment/output/predict/_prediction{name}'
    profile.update(count = 1)
    with rio.open(output_filename, 'w', **profile) as dst:
        dst.write(output_image, 1)  
    
    print(f"Saved prediction for {name} at {output_filename}")




