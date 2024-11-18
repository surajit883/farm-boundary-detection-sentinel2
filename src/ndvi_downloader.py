
import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime,timedelta
import geemap
from typing import List, Optional


import ee
ee.Authenticate()
ee.Initialize()



def ndvi_downloder(cords: List[List[List[float]]], PATH: str, sdate: Optional[str] = None, edate: Optional[str] = None) -> None:
    """
    Downloads NDVI data for a given area of interest (AOI).

    Args:
        cords (List[List[List[float]]]): Coordinates defining the AOI as a nested list of [longitude, latitude] pairs.
        PATH (str): Directory path where the NDVI data will be saved.
        sdate (Optional[str]): Start date for the data collection in 'YYYY-MM-DD' format. Defaults to 60 days before the current date if not provided.
        edate (Optional[str]): End date for the data collection in 'YYYY-MM-DD' format. Defaults to the current date if not provided.
    
    Returns:
        None
    """
    
    def ndviS2image(image):
        nir = image.select('B8')
        red = image.select('B4')
        ndvi =  nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        return image.addBands(ndvi)
    
    # SCL band cloud mask
    def sclCloudMask(image):
        ndvi = image.select('NDVI')
        crs = (ndvi.projection()).crs()
        scl = image.select('SCL')
        reScl = scl.resample('bilinear').reproject(crs = crs, scale = 10)
        mask = reScl.gt(3) and reScl.lte(7)
        # mask = mask.lte(7) # values which are not cloud
        maskedNdvi = ndvi.updateMask(mask)
        return ee.Image(maskedNdvi)
    
    def reduce_resolution(image) :
        crs = (image.select('NDVI').projection()).crs()
        return image.reproject(crs = crs, scale = 30)
    
    # create sample rectangles
    def mapRectangle(image) :
        return ee.Image(image).sampleRectangle(region = AOI, defaultValue = float(-1))
    
    def clipImage(image) :
        return image.clip(AOI)
    
    
    ## Main computation function...
    def copmute_ndvi(cords, PATH, sdate, edate):
        
        global scale_res, AOI
        
        AOI = ee.Geometry.Polygon(cords)

        if sdate is None:
            sdate = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        if edate is None:
            edate = datetime.now().strftime('%Y-%m-%d')
        
        dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(sdate, edate).filterBounds(AOI)
        ndvi_data=dataset.map(ndviS2image).map(sclCloudMask).select('NDVI').map(clipImage)
        
        geemap.ee_export_image_collection(ndvi_data, out_dir = PATH, scale = 10)

    
    ###########################################################################\

    try:
        copmute_ndvi(cords, PATH, sdate, edate)
    except Exception as e:
        print(e)

    ###########################################################################




if __name__ == '__main__':

    import geopandas as gpd
    
    kr_shape_map = gpd.read_file('/home/surajit/d/PROJECTS/escortscubota_assigment/data/AOI/AOI.gpkg')
    
    x, y = kr_shape_map.geometry[0].exterior.coords.xy    
    x = list(x)
    y = list(y)
    cords = [[[x[i], y[i]] for i in range(len(x))]]
    
    
    dest_dir = "/home/surajit/d/PROJECTS/escortscubota_assigment/output/NDVI/"
    
    ndvi_downloder(cords, dest_dir, sdate = '2022-02-01', edate = '2022-03-01')


