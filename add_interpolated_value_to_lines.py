'''
Created on 29 Sep. 2018

@author: Andrew Turner
'''
import sys
import netCDF4
import numpy as np
from geophys_utils import NetCDFLineUtils
import logging
import csv
import os
import re
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


logging.basicConfig(level=logging.INFO)

np.set_printoptions(threshold=30000, suppress=True,
                    formatter={'float_kind': '{:0.15f}'.format})

input_netcdf_folder_path = sys.argv[1]
output_netcdf_folder_path = sys.argv[2]
csv_file_name = sys.argv[3]
csv_file = open(csv_file_name, 'w')
writer = csv.writer(csv_file, delimiter=',')
writer.writerow(["LINE", "LINE_ARRAY_INDEX", "VALUE_TO_CHANGE", "CHANGED_VALUE"])

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


# def get_interpolation_function(array_with_nans):
#     # get interpolation function
#     array_without_nans = array_with_nans[~np.isnan(array_with_nans)]
#     index_no_nans = np.arange(0, len(array_without_nans), 1)
#     order = 1
#     interpolation_func = InterpolatedUnivariateSpline(x=index_no_nans, y=array_without_nans, k=order)
#     return interpolation_func


# def get_interpolation_function_distance(array_with_nans, coords):
#     utms_coords = utm_coords()
#     _transect_utils.coords2distance()
#     # get interpolation function
#     array_without_nans = array_with_nans[~np.isnan(array_with_nans)]
#     index_no_nans = np.arange(0, len(array_without_nans), 1)
#     order = 1
#
#     interpolation_func = InterpolatedUnivariateSpline(x=index_no_nans, y=array_without_nans, k=order)
#
#     return interpolation_func


# def extrap1d(interpolator):
#     xs = interpolator.x
#     ys = interpolator.y
#
#     def pointwise(x):
#         if x < xs[0]:
#             return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
#         elif x > xs[-1]:
#             return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
#         else:
#             return interpolator(x)
#
#     def ufunclike(xs):
#         return array(map(pointwise, array(xs)))
#
#     return ufunclike


def extrapolate_left(array_w_nans, line, num_points_for_extrap = 3):
    logging.debug("Extrapolating left")
    # find the number of nans until a real value is hit. This will determine how many values need to be extrapolated.
    nan_count = 0
    for nan in array_w_nans:
        if np.isnan(nan):
            nan_count = nan_count + 1
        else:
            break

    #  Get the index and values of the nan values to extrapolate.
    index_of_extrap_values = np.arange(0, nan_count, 1)
    values_to_change = array_w_nans[index_of_extrap_values]

    #  Find the array to base the interpolation off. This will be the points immediately before the nan values to
    #  extrapolate.  The size of this array is based on the input num_existing_points_use_for_extrap.
    index_array_to_use_for_interp = np.arange(nan_count, nan_count + num_points_for_extrap, 1)

    #  Get the interpolation function to use for extrapolating
    interp_function = interp1d(index_array_to_use_for_interp, array_w_nans[index_array_to_use_for_interp],
                               fill_value='extrapolate')
    #  Extrapolate the points into the array
    array_w_nans[index_of_extrap_values] = interp_function(index_of_extrap_values)

    #  Test the extrapolation
    assert_extrapolation_correct(array_w_nans, index_of_extrap_values, index_array_to_use_for_interp,
                                 extrapolate_left=True)
    #  Log the changes in a csv
    log_changes_in_csv(nan_count, line, index_of_extrap_values, values_to_change, array_w_nans[index_of_extrap_values],
                       "extrapolate_left")

    return array_w_nans, index_of_extrap_values


def extrapolate_right(array_w_nans, line, num_existing_points_use_for_extrap=3):
    # find the number of nans until a real value is hit. This will determine how many values need to be extrapolated.
    nan_count = 0
    array_index = len(array_w_nans) - 1
    while array_index >= 0:
        if np.isnan(array_w_nans[array_index]):
            nan_count = nan_count + 1
        else:
            break
        array_index = array_index - 1

    #  Get the index and values of the nan values to extrapolate.
    last_index = len(array_w_nans) - 1
    index_of_extrap_values = np.arange(len(array_w_nans) - nan_count, len(array_w_nans), 1)
    values_to_change = array_w_nans[index_of_extrap_values]

    #  Find the array to base the interpolation off. This will be the points immediately before the nan values to
    #  extrapolate.  The size of this array is based on the input num_existing_points_use_for_extrap.
    start_index = last_index - nan_count - num_existing_points_use_for_extrap
    index_array_to_use_for_interp = np.arange(start_index, last_index - nan_count, 1)

    #  Get the interpolation function to use for extrapolating
    interp_function = interp1d(index_array_to_use_for_interp, array_w_nans[index_array_to_use_for_interp],
                               fill_value='extrapolate')
    #  Extrapolate the points into the array
    array_w_nans[index_of_extrap_values] = interp_function(index_of_extrap_values)

    #  Test the extrapolation
    assert_extrapolation_correct(array_w_nans, index_of_extrap_values, index_array_to_use_for_interp,
                                 extrapolate_left=False)
    #  Log the changes in a csv
    log_changes_in_csv(nan_count, line, index_of_extrap_values, values_to_change, array_w_nans[index_of_extrap_values],
                       "extrapolate_right")

    return array_w_nans, index_of_extrap_values


def assert_extrapolation_correct(complete_array, index_of_exptrapolate_values, array_to_use_for_interp, extrapolate_left):
    print("Asserting interpolation is correct...")
    print("Index array to use for interp {}".format(array_to_use_for_interp))
    print("Array Values to use for interp {}".format(complete_array[array_to_use_for_interp]))

    if extrapolate_left:
        test_array_indexes = np.append(index_of_exptrapolate_values, array_to_use_for_interp)
    else:
        test_array_indexes = np.insert(index_of_exptrapolate_values, 0, array_to_use_for_interp)
    print("index of values to extrapolate: {}".format(index_of_exptrapolate_values))
    print("extrapolated values: {}".format(complete_array[index_of_exptrapolate_values]))

    test_array = complete_array[test_array_indexes]
    print("array values to use for interpolation function with extralopalated values added: {}".format(test_array))

    value_index = 0
    difference_array = np.array([])
    while value_index < len(test_array) - 2:
        difference = test_array[value_index] - test_array[value_index + 1]
        print("{} - {} = {}".format(test_array[value_index],  test_array[value_index + 1], difference))
        assert difference < 0.1
        difference_array = np.append(difference_array, difference)

        value_index = value_index + 1
    print("variance")
    print(difference_array)
    print(np.var(difference_array))



# def fill_in_masked_values_with_interpolated_values(narray, interp_func, line):
#     is_nan_bool_array, x = nan_helper(narray)
#
#     ndarray2 = narray.copy()
#     interpolate_indexes = (is_nan_bool_array.nonzero()[0])
#
#     if interpolate_indexes is not None: # if are points to interpolate then interpolate them
#         narray[is_nan_bool_array] = interp_func(x(is_nan_bool_array))
#         log_changes_in_csv(len(narray[is_nan_bool_array]), line, interpolate_indexes, ndarray2[interpolate_indexes], narray[interpolate_indexes], "interpolate")
#         #interpolate_indexes2 = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]        return narray, interpolate_indexes
#     return narray, []

# def interpolate():
#     pass


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def fill_in_the_blanks(narray, num_points_each_way_to_use_for_interp_func):

    is_nan_bool_array, x = nan_helper(narray)
    interpolate_indexes = (is_nan_bool_array.nonzero()[0])
    print("interpolate_indexes: {}".format(interpolate_indexes))
    if len(interpolate_indexes) > 0:  # if are points to interpolate then interpolate them

        groups_of_values_to_interp = (group_consecutives(interpolate_indexes))
        logging.debug("Groups of Nans found: {}".format(groups_of_values_to_interp))

        for group_to_interp in groups_of_values_to_interp:
            # check they are not nans at the beginning or end as these will require extrapolation instead.
            if group_to_interp[0] == 0 or group_to_interp[-1] == len(narray) - 1:
                pass
            else:
                # interpolate the group_to_interp
                min_index = np.min(group_to_interp)
                max_index = np.max(group_to_interp)
                i = 1
                indxes = np.array([], dtype=int)
                while i <= num_points_each_way_to_use_for_interp_func:
                    indxes = np.append(indxes, min_index - i)
                    indxes = np.append(indxes, max_index + i)
                    i = i + 1

                sorted_index = np.sort(indxes)
                x = sorted_index
                y = narray[sorted_index]
                print("X: {}".format(x))
                print("Y: {}".format(y))

                interpolation_func = interp1d(x, y)

                xnew = group_to_interp
                ynew = interpolation_func(xnew)
                print("xnew: {}".format(xnew))
                print("ynew: {}".format(ynew))

                # put the interp points in the array
                print("old")
                print(narray[xnew])
                narray[xnew] = ynew
                print("new")
                print(narray[xnew])
        # else:
        #     return narray, interpolate_indexes
    print("interpolate_indexes2: {}".format(interpolate_indexes))
    return narray, interpolate_indexes


def get_array_for_prediction_type_variable(var_array_including_nan, interpolated_index, extrapolate_left_indexes,
                                           extrapolate_right_indexes):

    a = np.zeros(len(var_array_including_nan), dtype=int)
    print(a)
    print(interpolated_index)
    a[interpolated_index] = 1
    print("interpolated ones: {}".format(a[interpolated_index]))
    if extrapolate_left_indexes is not None:
        a[extrapolate_left_indexes] = 2
    if extrapolate_right_indexes is not None:
        a[extrapolate_right_indexes] = 2
    return a


def log_changes_in_csv(nan_count, line, index_of_extrap_values, values_to_change, changed_values, extrap_direction):
    """Record the points that changed to a csv."""
    print('nan_count: {}'.format(nan_count))
    print('line: {}'.format(line))
    print('index_of_extrap_values: {}'.format(index_of_extrap_values))
    print('values_to_change: {}'.format(values_to_change))
    print('changed_values: {}'.format(changed_values))
    print('extrap_direction: {}'.format(extrap_direction))

    index = 0
    while index < nan_count:
        writer.writerow([line, index_of_extrap_values[index], values_to_change[index], changed_values[index], extrap_direction])
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

    #var_array_including_nan[10:12] = np.nan
    #var_array_including_nan[250:350] = np.nan
    #var_array_including_nan[0:150] = np.nan
    #var_array_including_nan[-1] = np.nan
    #var_array_including_nan[-2] = np.nan

    extrapolate_left_indexes = None
    extrapolate_right_indexes = None

    # interpolate all the nans not on the edges of the line arrays.
    var_array_including_nan, interpolated_index = fill_in_the_blanks(var_array_including_nan, 1)

    # if the first value is nan and extrapolation is required
    if np.isnan(var_array_including_nan[0]):
        var_array_including_nan, extrapolate_left_indexes = extrapolate_left(var_array_including_nan, line, 3)

    # if the las value is nan and extrapolation is required
    if np.isnan(var_array_including_nan[-1]):
        logging.debug(var_array_including_nan[-1])
        var_array_including_nan, extrapolate_right_indexes = extrapolate_right(var_array_including_nan, line, 3)

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

    line_number = 0
    for line in line_list:
        logging.debug("LINE: {}".format(line))

        lat, long = get_lat_and_long_of_line(netcdf_line_utils, line)
        #coords = lat, long = get_lat_and_long_of_line(netcdf_line_utils, line)

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

            # print(line)
            #
            # print(nc_dataset.variables['latitude_first'][line_number])
            # print(next(netcdf_line_utils.get_lines(line_numbers=line, variables=["line_index", "longitude"],
            #                          get_contiguous_lines=True)))
            # index_of_null_points_line_list = []
            # index_point = 0
            # while index_point < len(nc_dataset.dimensions['point']):
            #     #print(index_point)
            #     #print(nc_dataset.variables['line_index'][index_point])
            #     if nc_dataset.variables['line_index'][index_point] == 97:
            #         #print("HERE")
            #         index_of_null_points_line_list.append(index_point)
            #
            #     index_point = index_point + 1
            # print("index_of_null_points_line_list")
            # print(index_of_null_points_line_list)

            #nc_dataset.variables['line_index'][index_of_null_points_line_list] = None


            #######
            # remove line from dataset?
            #######
            continue
        line_number = line_number + 1

        complete_array_long, lookup_index_array = replace_nan_in_variable_with_predicted_value(long, line)
        complete_array_lat, lookup_index_array = replace_nan_in_variable_with_predicted_value(lat, line)

        dict_of_arrays['lats_w_predictions'] = np.append(dict_of_arrays['lats_w_predictions'], complete_array_lat)

        dict_of_arrays['longs_w_predictions'] = np.append(dict_of_arrays['longs_w_predictions'], complete_array_long)
        dict_of_arrays['lookup_index_array'] = np.append(dict_of_arrays['lookup_index_array'], lookup_index_array)

    return dict_of_arrays


def main():
    # netcdf_input_path = sys.argv[1]
    # nc_output_dataset_path = sys.argv[2]
    for root, dirs, files in os.walk(input_netcdf_folder_path):
        for filename in files:
            if re.search(".nc", filename):
                netcdf_input_dataset_path = "{}\{}".format(root, filename)
                #print(netcdf_input_dataset_path)

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
                nc_output_dataset_path = "{}\{}".format(output_netcdf_folder_path, filename)
                print('nc_output_dataset_path')
               # print(nc_output_dataset_path)

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
                        print("NAME")
                        print(name)
                        # edit lat and long variables
                        if name == 'latitude':
                            edited_dict = src[name].__dict__
                            # fill value must be given here rather than copied over with the other attributes.
                            dst.createVariable(name, variable.datatype, variable.dimensions, fill_value=edited_dict['_FillValue'])
                            print("dict_of_arrays['lats_w_predictions']")
                            print(dict_of_arrays['lats_w_predictions'])
                            dst[name][:] = dict_of_arrays['lats_w_predictions']
                            del edited_dict['_FillValue']
                            dst[name].setncatts(edited_dict)
                        elif name == 'longitude':
                            edited_dict = src[name].__dict__
                            dst.createVariable(name, variable.datatype, variable.dimensions, fill_value=edited_dict['_FillValue'])
                            dst[name][:] = dict_of_arrays['longs_w_predictions']
                            del edited_dict['_FillValue']
                            dst[name].setncatts(edited_dict)

                        else:
                            # copy over all the existing variables
                            logging.debug("Adding variable: {}...".format(name))
                            dst.createVariable(name, variable.datatype, variable.dimensions)
                            edited_dict = src[name].__dict__
                            dst[name].setncatts(edited_dict)
                            dst[name][:] = src[name][:]

                    # create variable for the lookup_index_arrays
                    dst.createVariable('coord_predicted', dict_of_arrays['lookup_index_array'].dtype, ('point',))
                    logging.info("dict_of_arrays['lookup_index_array']")
                    logging.info(len(dict_of_arrays['lookup_index_array']))
                    logging.info(len(dst['coord_predicted'][:]))
                    dst['coord_predicted'][:] = dict_of_arrays['lookup_index_array']
                    # set attributes
                    edited_dict = {"long_name": "coordinate_predicted_flag"}
                    dst['coord_predicted'].setncatts(edited_dict)

                    # create variable for the lookup_table
                    coord_predicted_lookup_table = np.array(["Coordinate is a GPS recording.",
                    "Coordinate is a linear interpolation prediction value calculated from the existing gps recordings "
                    "of points within the line.",
                    "Coordinate is a linear extrapolation prediction value calculated from the existing gps recordings "
                    "of points within the line.",])

                    # make a new dimension
                    dst.createDimension('coord_predicted_lookup_table', (len(coord_predicted_lookup_table)))
                    dst.createVariable('coord_predicted_lookup_table', 'S3', ('coord_predicted_lookup_table',))
                    dst['coord_predicted_lookup_table'][:] = coord_predicted_lookup_table

                    print(dst.variables['latitude'][:])
                    print(dst.variables['longitude'][:])
if __name__ == '__main__':
    main()
