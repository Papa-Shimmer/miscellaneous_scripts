import netCDF4
import os
from collections import Counter

netcdf_path_local = os.path.abspath("C:\\Users\\u62231\\Desktop\\airbourne_rad_mag")

#netcdf_local_test_file_name = "\\GSSA_P1255MAG_Marree.nc"
netcdf_local_test_file_name = "GSWA_P1233_Hickman_Crater_MAG.nc"

print(netcdf_path_local)
print(netcdf_local_test_file_name)
netcdf_input_path = netcdf_path_local + '\\' + netcdf_local_test_file_name
print(netcdf_input_path)

netcdf_input_dataset = netCDF4.Dataset(netcdf_input_path,
                                                 mode="r",
                                                 clobber=True,
                                                 format='NETCDF4')
# print(nc_input_dataset)
# print(nc_input_dataset.variables['index_line'][:])


nc_output_dataset_path = "C:\\Users\\u62231\\Desktop\\airbourne_rad_mag_edited\\EDITED_{}".format(netcdf_local_test_file_name)

def get_list_of_line_indexes_for_points(nc_input_dataset):

    point_dim_len = len(nc_input_dataset.dimensions['point'])

    i = 0
    index_line_index = 0
    point_line_index_list = [None] * point_dim_len #  list to populate with line indexe for each point

    while i < point_dim_len:
    #for point in nc_output_dataset.variables['point']:
        if index_line_index < len(nc_input_dataset.dimensions['line']):
            if i >= nc_input_dataset.variables['index_line'][index_line_index]:  #  and i < nc_output_dataset.variables['index_line'][index_line_index + 1]:
                #point_line_index_list[i] = nc_output_dataset.variables['line'][index_line_index]
                point_line_index_list[i] = index_line_index
                i = i + 1
                #print("point: {}".format(i))
                #print("line index: {}".format(nc_input_dataset.variables['line'][index_line_index]))

            if i == nc_input_dataset.variables['index_line'][index_line_index] + nc_input_dataset.variables['index_count'][index_line_index]:
                index_line_index = index_line_index + 1
    return point_line_index_list

def np_get_list_of_line_indexes_for_points(nc_input_dataset):
    pass

def test_list():
    count_dict = Counter(point_line_index_list)
    print(count_dict)
    index = 0
    for i in netcdf_input_dataset.variables['index_count']:
        print("index: {}".format(index))
        print("index count: {}".format(count_dict.get(index)))
        assert i == count_dict.get(index)
        index = index + 1

vars_to_change_attribute = ['bearing', 'line', 'flag_linetype', 'longitude_first', 'flight', 'latitude_last', 'point', 'mag_awags',
                            'bounding_polygon', 'height', 'longitude_last', 'mag_mlev',
                            'survey', 'fiducial', 'date', 'line', 'mag_lev',
                            'latitude_first']

#vars_to_exclude = ['index_line', 'index_count']
vars_to_exclude = []
with netcdf_input_dataset as src, netCDF4.Dataset(nc_output_dataset_path, "w") as dst:
    # copy global attributes all at once via dictionary
    dst.setncatts(src.__dict__)
    # copy dimensions
    for name, dimension in src.dimensions.items():
        dst.createDimension(
            name, (len(dimension) if not dimension.isunlimited() else None))
    # copy all file data except for the excluded
    for name, variable in src.variables.items():
        print(name)
        print(variable)
        if name == 'point': #make point_id?
            var_attributes_dict = {"long_name": "zero-based index of value in line",
                                   "lookup": "line"}
            point_line_index_list = get_list_of_line_indexes_for_points(src)
            print(point_line_index_list)
            test_list()
            point_line_index_var = dst.createVariable(varname="line_index", datatype="i1", dimensions="point")
            point_line_index_var[:] = point_line_index_list
            point_line_index_var.setncatts(var_attributes_dict)

        if name not in vars_to_exclude:
            edited_dict = {}
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            # copy variable attributes all at once via dictionary
            if name in vars_to_change_attribute:
                print("source_dict")
                print(src[name].__dict__)
                edited_dict = src[name].__dict__
                edited_dict['long_name'] = edited_dict.pop('standard_name')
                dst[name].setncatts(edited_dict)
                dst[name][:] = src[name][:]
            elif name == "crs":
                edited_dict = src['crs'].__dict__
                edited_dict['long_name'] = 'coordinate_reference_system'
                dst[name].setncatts(edited_dict)
                dst['crs'][:] = src['crs'][:]
            else:

                dst[name].setncatts(src[name].__dict__)
                dst[name][:] = src[name][:]

    print(dst.variables['line'][:])


    # print("SOURCE DICT FORMAT {}".format(src[name].__dict__))

    #var_attributes_dict.units = "k"



# byte line_index(point) ;
#         line_index:_FillValue = -1b ;
#         line_index:long_name = "zero-based index of value in line" ;
#         line_index:lookup = "line" ;
