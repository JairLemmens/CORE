import shapely
import os
import json
from matplotlib import image
import numpy as np
from shapely.wkt import loads


def split_tile_dataset(img_dir,label_dir,buildings_dir,no_buildings_dir,divisions = 10):
    """
    Generate a dataset from the xbd dataset splitting tiles which contain and do not contain buildings.
    Img_dir refers to the satelite image directory.
    Label_dir contains accompanying json file directory.
    Buildings_dir is where tiles containing buildings will be stored.
    No_buildings_dir is where tiles NOT containing buildings will be stored.
    Divisions indicates the amount of fragments both horizontal and vertical will be extracted from the original images.
    """
    for _filename in os.listdir(img_dir):
        _filename = os.path.splitext(_filename)[0]
        _label_file = open(f'{label_dir}/{_filename}.json')
        _label_data =json.load(_label_file)
        _img = image.imread(f'{img_dir}/{_filename}.png')
        _polygons = []
        for _obj in _label_data['features']['xy']:
            _polygon = loads(_obj['wkt'])
            _polygons.append(_polygon)
            _polygon = np.array(list(_polygon.exterior.coords),dtype=np.int32)

        _grid_size = np.floor(np.divide(_img.shape[:2],divisions)).astype(np.int32)    
        for i in range(divisions):
            for j in range(divisions):
                _imgsection = _img[i*_grid_size[0]:(i+1)*_grid_size[0],j*_grid_size[0]:(j+1)*_grid_size[0]]
                _poly = shapely.linearrings(np.multiply([[i,j],[i+1,j],[i+1,j+1],[i,j+1]],_grid_size))
                temp = shapely.intersects(_polygons,_poly)
                if np.sum(temp) > 0:
                    image.imsave(f"{buildings_dir}/{_filename}_{i}_{j}.jpeg", _imgsection)
                else:
                    image.imsave(f"{no_buildings_dir}/{_filename}_{i}_{j}.jpeg", _imgsection)
