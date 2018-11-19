import os
import netCDF4
import re

def get_netcdf_collection_global_var_min(root_dir, global_var_name):
    collection_min = None
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('nc'):
                file_path = os.path.join(root_dir, file)
                dataset = netCDF4.Dataset(file_path)
                global_var = getattr(dataset, global_var_name)
                if collection_min is None:
                    collection_min = global_var

                print(dataset.variables['crs'])
                collection_min = min(collection_min, global_var)


    return collection_min


def get_netcdf_collection_global_var_max(root_dir, global_var):
    collection_max = None
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('nc'):
                file_path = os.path.join(root_dir, file)
                dataset = netCDF4.Dataset(file_path)
                collection_max = max(dataset.global_var)

    return collection_max


root_dir = ""

# collection_lat_min = get_netcdf_collection_global_var_min(root_dir, "geospatial_lat_min")
# collection_lat_max = get_netcdf_collection_global_var_min(root_dir, "geospatial_lat_max")
# collection_lon_min = get_netcdf_collection_global_var_min(root_dir, "geospatial_lon_min")
# collection_lon_max = get_netcdf_collection_global_var_min(root_dir, "geospatial_lon_max")