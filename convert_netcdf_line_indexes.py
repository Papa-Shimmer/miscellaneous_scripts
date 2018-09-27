import netCDF4
import os
from collections import Counter
import sys
import numpy as np


crs_gda94_string = '''GEOGCS["GDA94", DATUM["Geocentric_Datum_of_Australia_1994", 
SPHEROID["GRS 1980",6378137,298.257222101, AUTHORITY["EPSG","7019"]], 
TOWGS84[0,0,0,0,0,0,0], AUTHORITY["EPSG","6283"]], PRIMEM["Greenwich",0, AUTHORITY["EPSG","8901"]], 
UNIT["degree",0.0174532925199433, AUTHORITY["EPSG","9122"]], AUTHORITY["EPSG","4283"]]'''



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
    point_dim_len = len(nc_input_dataset.dimensions['point'])
    line_dim_len = len(nc_input_dataset.dimensions['line'])
    line_start_indexes = np.array(nc_input_dataset.variables['index_line'])
    print("Here")
    print(line_start_indexes)
    line_end_indexes = np.array(nc_input_dataset.variables['index_line'] + np.array(nc_input_dataset.variables['index_count']))
    print("line_start_indexes")
    print(line_start_indexes)
    print("line_end_indexes")
    print(line_end_indexes)
    new_line_values = np.arange(line_dim_len)
    print(new_line_values)

    line_index = np.zeros(point_dim_len, dtype='int8')

    for i in np.arange(line_dim_len):
        print("I: {}".format(i))
        start_i = line_start_indexes[i]
        print(start_i)
        end_i = line_end_indexes[i]
        print(end_i)
        line_index[start_i:end_i] = i

    return line_index


def test_list(point_line_index_list, nc_input_dataset):
    print("TESTING")
    count_dict = Counter(point_line_index_list)
    print(count_dict)
    index = 0
    for i in nc_input_dataset.variables['index_count']:
        print("index: {}".format(index))
        print("index count: {}".format(count_dict.get(index)))
        assert i == count_dict.get(index)
        index = index + 1




    # print("SOURCE DICT FORMAT {}".format(src[name].__dict__))

    #var_attributes_dict.units = "k"



# byte line_index(point) ;
#         line_index:_FillValue = -1b ;
#         line_index:long_name = "zero-based index of value in line" ;
#         line_index:lookup = "line" ;

def main():
    netcdf_input_path = sys.argv[1]
    nc_output_dataset_path = sys.argv[2]

    netcdf_input_dataset = netCDF4.Dataset(netcdf_input_path,
                                           mode="r",
                                           clobber=True,
                                           format='NETCDF4')
    # print(nc_input_dataset)
    # print(nc_input_dataset.variables['index_line'][:])






    vars_to_change_attribute = ['bearing', 'line', 'flag_linetype', 'longitude_first', 'flight', 'latitude_last',
                                'point', 'mag_awags',
                                'bounding_polygon', 'height', 'longitude_last', 'mag_mlev',
                                'survey', 'fiducial', 'date', 'line', 'mag_lev',
                                'latitude_first']

    vars_to_exclude = ['point', 'index_line', 'index_count', 'survey']
    #vars_to_exclude = []
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
            if name == 'point':  # make point_id?
                var_attributes_dict = {"long_name": "zero-based index of value in line",
                                       "lookup": "line"}
                point_line_index_list = np_get_list_of_line_indexes_for_points(netcdf_input_dataset)
                print(point_line_index_list)
                test_list(point_line_index_list, netcdf_input_dataset)
                point_line_index_var = dst.createVariable(varname="line_index", datatype="i1", dimensions="point")
                point_line_index_var[:] = point_line_index_list
                point_line_index_var.setncatts(var_attributes_dict)

            if name not in vars_to_exclude:
                if name =="crs":
                    dst.createVariable(name, 'b')  # type byte and no dimension (scalar)
                    edited_dict = src['crs'].__dict__
                    print(edited_dict)
                    edited_dict['long_name'] = 'coordinate_reference_system'
                    edited_dict['spatial_ref'] = crs_gda94_string
                    dst[name].setncatts(edited_dict)
                else:
                #edited_dict = {}
                    x = dst.createVariable(name, variable.datatype, variable.dimensions)
                    # copy variable attributes all at once via dictionary
                    if name in vars_to_change_attribute:
                        print("source_dict")
                        print(src[name].__dict__)
                        edited_dict = src[name].__dict__
                        edited_dict['long_name'] = edited_dict.pop('standard_name')
                        dst[name].setncatts(edited_dict)
                        dst[name][:] = src[name][:]
                    else:
                        dst[name].setncatts(src[name].__dict__)
                        dst[name][:] = src[name][:]

        print(dst.variables['line'][:])

if __name__ == "__main__":

    main()
