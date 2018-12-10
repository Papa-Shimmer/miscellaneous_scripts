'''
Created on 29 Sep. 2018

@author: Andrew Turner
'''
import sys
import netCDF4
import numpy as np
from geophys_utils import NetCDFLineUtils
from scipy.interpolate import InterpolatedUnivariateSpline
import logging
import csv
import os
import re

logging.basicConfig(level=logging.INFO)

np.set_printoptions(threshold=30000, suppress=True,
                    formatter={'float_kind': '{:0.15f}'.format})


# snipped from https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        linear interpolation of NaNs
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def get_lat_and_long_of_line(line_utils, line_number):
    line_number, line_dict = next(line_utils.get_lines(line_numbers=line_number, variables=["latitude", "longitude"], get_contiguous_lines=True))
    return line_dict['latitude'], line_dict['longitude']


def fill_in_masked_values_with_interpolated_values(narray):
    is_nan_bool_array, x = nan_helper(narray)
    interpolate_indexes = (is_nan_bool_array.nonzero()[0])
    narray[is_nan_bool_array] = np.interp(x(is_nan_bool_array), x(~is_nan_bool_array), narray[~is_nan_bool_array])
    return narray, interpolate_indexes

def get_array_to_extrapolate_left(array_with_nans):
    logging.debug("Extrapolating left")
    # find the number of nans until a real value is hit. This will determine how many values need to be extrapolated.
    nan_count = 0
    for nan in array_with_nans:
        if np.isnan(nan):
            nan_count = nan_count + 1
        else:
            break

    array_to_extrapoltate = np.arange(nan_count, 0, -1) * -1

    index_of_extrap_values = np.arange(0, nan_count, 1)

    return nan_count, array_to_extrapoltate, index_of_extrap_values


def extrapolate_left(array_with_nans, line, interpolation_func):
    nan_count, array_to_extrapoltate, index_of_extrap_values = get_array_to_extrapolate_left(array_with_nans)

    # Extrapolate with linear interpolation function.
    y = interpolation_func(array_to_extrapoltate)

    # attach extrapolated array onto existing array
    values_to_change = array_with_nans[index_of_extrap_values]
    array_with_nans[index_of_extrap_values] = y
    log_changes_in_csv(nan_count, line, index_of_extrap_values, values_to_change, y)

    return array_with_nans, index_of_extrap_values


def get_interpolation_function(array_with_nans):
    # get interpolation function
    array_without_nans = array_with_nans[~np.isnan(array_with_nans)]
    index_no_nans = np.arange(0, len(array_without_nans), 1)
    order = 1
    interpolation_func = InterpolatedUnivariateSpline(x=index_no_nans, y=array_without_nans, k=order)

    return interpolation_func

def extrapolate_right(array_w_nans, line, interpolation_func):
    logging.debug("Extrapolating right")
    # find the number of nans until a real value is hit. This will determine how many values need to be extrapolated.
    nan_count = 0
    length_of_array_with_nans = len(array_w_nans)
    array_index = len(array_w_nans) - 1

    while array_index >= 0:
        if np.isnan(array_w_nans[array_index]):
            nan_count = nan_count + 1
        else:
            break
        array_index = array_index - 1

    array_to_extrapoltate = np.arange(length_of_array_with_nans - nan_count, length_of_array_with_nans, 1)
    print('array_to_extrapoltate_right')

    index_of_extrap_values = np.arange(0, nan_count, 1)
    values_to_change = array_w_nans[array_to_extrapoltate]
    # Extrapolate with linear interpolation function.
    y = interpolation_func(array_to_extrapoltate)
    array_w_nans[array_to_extrapoltate] = y
    logging.debug('array_to_extrapoltate_right')
    logging.debug(array_to_extrapoltate)
    #  np.append(y, array_with_nans)
    # attach extrapolated array onto existing array


    array_w_nans[index_of_extrap_values] = y[index_of_extrap_values]
    log_changes_in_csv(nan_count, line, index_of_extrap_values, values_to_change, y)

    return array_w_nans, array_to_extrapoltate


def get_array_for_prediction_type_variable(var_array_including_nan, interpolated_index, extrapolate_left_indexes, extrapolate_right_indexes):
    a = np.zeros(len(var_array_including_nan), dtype=int)
    a[interpolated_index] = 1
    logging.debug('interpolated_index')
    logging.debug(interpolated_index)
    logging.debug('extrapolate_left_indexes')
    logging.debug(extrapolate_left_indexes)
    if extrapolate_left_indexes is not None:
        a[extrapolate_left_indexes] = 2
    if extrapolate_right_indexes is not None:
        a[extrapolate_right_indexes] = 2
    return a

def log_changes_in_csv(nan_count, line, index_of_extrap_values, values_to_change, changed_values):

    print('nan_count: {}'.format(nan_count))
    print('line: {}'.format(line))
    print('index_of_extrap_values: {}'.format(index_of_extrap_values))
    print('values_to_change: {}'.format(values_to_change))
    print('changed_values: {}'.format(changed_values))

    index = 0
    while index < nan_count:
        writer.writerow([line, index_of_extrap_values[index], values_to_change[index], changed_values[index]])
        index = index + 1

def check_lat_and_long_are_masked_consistently(lat_w_nans, long_w_nans):
    index = 0
    while index < len(lat_w_nans):
        if np.isnan(lat_w_nans[index]) or np.isnan(long_w_nans[index]):
            if np.isnan(lat_w_nans[index]) and np.isnan(long_w_nans[index]):
                pass
            else:
                return False
        index = index + 1
    return True


def check_for_lines_with_no_coords(lat_w_nans):
    # check line has coord values.
    nan_bool = np.isnan(lat_w_nans)
    if np.all(nan_bool):
        logging.debug(nan_bool)
        return False
    else:
        logging.debug(nan_bool)
        pass
    return True


def replace_nan_in_variable_with_predicted_value(variable_narray, line):

    var_array_including_nan = np.ma.filled(variable_narray.astype(float), np.nan)

    # var_array_including_nan[10:12] = np.nan
    # var_array_including_nan[0] = np.nan
    # var_array_including_nan[-1] = np.nan
    # logging.debug('var_array_including_nan')
    # logging.debug(var_array_including_nan)

    # print("var_array_including_nan")
    # print(var_array_including_nan)
    # get the number of values to extrapolate at beginning.
    extrapolate_left_indexes = None
    extrapolate_right_indexes = None

    interp_func = get_interpolation_function(var_array_including_nan)

    if np.isnan(var_array_including_nan[0]):
        # then the first value is nan and extrapolation is required
        var_array_including_nan, extrapolate_left_indexes = extrapolate_left(var_array_including_nan, line, interp_func)

    if np.isnan(var_array_including_nan[-1]):
        logging.debug(var_array_including_nan[-1])
        var_array_including_nan, extrapolate_right_indexes = extrapolate_right(var_array_including_nan, line, interp_func)


    var_array_including_nan, interpolated_index = fill_in_masked_values_with_interpolated_values(var_array_including_nan)

    lookup_index_array = get_array_for_prediction_type_variable(var_array_including_nan,
                                           interpolated_index,
                                           extrapolate_left_indexes,
                                           extrapolate_right_indexes)

    return var_array_including_nan, lookup_index_array


def get_interpolated_values_and_index(nc_dataset):
    # nc_path = sys.argv[1]
    # nc_dataset = netCDF4.Dataset(nc_path)
    line_list = nc_dataset.variables['line'][:]
    netcdf_line_utils = NetCDFLineUtils(nc_dataset)

    dict_of_arrays = {
        'lats_w_predictions': [],
        'longs_w_predictions': [],
        'lookup_index_array': []
        }

    for line in line_list:
        logging.debug("LINE: {}".format(line))

        lat, long = get_lat_and_long_of_line(netcdf_line_utils, line)

        lat_w_nans = np.ma.filled(lat.astype(float), np.nan)
        long_w_nans = np.ma.filled(long.astype(float), np.nan)

        if check_lat_and_long_are_masked_consistently(lat_w_nans, long_w_nans):
            pass
        else:
            logging.error(("Line {} contains latitude and longitude coordinates with in consistent Nan values.".format(line)))
            return False

        if check_for_lines_with_no_coords(lat_w_nans):
            pass
        else:
            logging.error(("Line {} has no coordinates. All values are Nan".format(line)))
            return False


        complete_array_long, lookup_index_array = replace_nan_in_variable_with_predicted_value(long, line)
        complete_array_lat, lookup_index_array = replace_nan_in_variable_with_predicted_value(lat, line)

        dict_of_arrays['lats_w_predictions'] = np.append(dict_of_arrays['lats_w_predictions'], complete_array_lat)

        dict_of_arrays['longs_w_predictions'] = np.append(dict_of_arrays['longs_w_predictions'], complete_array_long)
        dict_of_arrays['lookup_index_array'] = np.append(dict_of_arrays['lookup_index_array'], lookup_index_array)

    return dict_of_arrays

def test_indexes_match_nan_locations(lat_w_nans, longs_w_nans):
    pass

def main():

    input_netcdf_folder_path = sys.argv[1]
    output_netcdf_folder_path = sys.argv[2]
    csv_file_name = sys.argv[3]
    # netcdf_input_path = sys.argv[1]
    # nc_output_dataset_path = sys.argv[2]

    csv_file = open(csv_file_name, 'w')
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["LINE", "LINE_ARRAY_INDEX", "VALUE_TO_CHANGE", "CHANGED_VALUE"])

    for root, dirs, files in os.walk(input_netcdf_folder_path):
        print(root)
        for filename in files:
            if re.search(".nc", filename):
                netcdf_input_dataset_path = "{}\\{}".format(root, filename)
                print(netcdf_input_dataset_path)

                netcdf_input_dataset = netCDF4.Dataset(netcdf_input_dataset_path,
                                               mode="r+",
                                               clobber=True,
                                               )
                # add interpolated_values_to_lines.
                dict_of_arrays = get_interpolated_values_and_index(netcdf_input_dataset)
                if dict_of_arrays is False:
                    logging.debug("Skipping file {}".format(netcdf_input_dataset))
                    continue

                logging.debug('dict_of_arrays')
                logging.debug(dict_of_arrays)
                nc_output_dataset_path = "{}\\{}".format(output_netcdf_folder_path, filename)

                with netcdf_input_dataset as src, netCDF4.Dataset(nc_output_dataset_path, "w") as dst:

                    # copy global attributes all at once via dictionary
                    logging.debug('copying global attributes from input netcdf to output netcdf...')
                    dst.setncatts(src.__dict__)
                    logging.debug(dst.__dict__)

                    # copy dimensions
                    logging.debug('copying dimensions from input netcdf to output netcdf...')
                    for name, dimension in netcdf_input_dataset.dimensions.items():
                        logging.debug("Dimension: {}".format(name))
                        dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))


                    for name, variable in src.variables.items():
                        # edit lat and long variables
                        if name is 'latitude':
                            dst.createVariable(name, variable.datatype, variable.dimensions)
                            dst[name][:] = dict_of_arrays['lats_w_predictions']
                            edited_dict = src[name].__dict__
                            dst[name].setncatts(edited_dict)
                        if name is 'longitude':
                            dst[name][:] = dict_of_arrays['longs_w_predictions']
                            edited_dict = src[name].__dict__
                            dst[name].setncatts(edited_dict)

                        # copy over all the existing variables
                        logging.debug("Adding variable: {}...".format(name))
                        dst.createVariable(name, variable.datatype, variable.dimensions)
                        edited_dict = src[name].__dict__
                        dst[name].setncatts(edited_dict)
                        dst[name][:] = src[name][:]

                    # create variable for the lookup_index_arrays
                    dst.createVariable('coord_predicted', dict_of_arrays['lookup_index_array'].dtype, ('point',))
                    logging.info("dict_of_arrays['lookup_index_array']")
                    logging.info(dict_of_arrays['lookup_index_array'])
                    dst['coord_predicted'][:] = dict_of_arrays['lookup_index_array']
                    # set attributes
                    edited_dict = {"long_name": "coordinate_predicted_flag"}
                    dst['coord_predicted'].setncatts(edited_dict)

                    # create variable for the lookup_table
                    coord_predicted_lookup_table = np.array(["Coordinate is a GPS recording.", "Coordinate is a linear interpolation prediction value calculated from the existing gps recordings of points within the line.", "Coordinate is a linear extrapolation prediction value calculated from the existing gps recordings of points within the line."])

                    # make a new dimension
                    dst.createDimension('coord_predicted_lookup_table', (len(coord_predicted_lookup_table)))
                    dst.createVariable('coord_predicted_lookup_table', 'S3', ('coord_predicted_lookup_table',))
                    dst['coord_predicted_lookup_table'][:] = coord_predicted_lookup_table

if __name__ == '__main__':
    main()
