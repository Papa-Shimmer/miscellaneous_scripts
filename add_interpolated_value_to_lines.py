'''
Created on 29 Sep. 2018

@author: Andrew Turner
'''
import sys
import netCDF4
import numpy as np
from geophys_utils import NetCDFLineUtils, _transect_utils
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
import logging
import csv
import os
import re
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy import arange, array, exp

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
    #print("line_DICT")
    #print(line_dict)
    #return line_dict['coordinates']
    return line_dict['latitude'], line_dict['longitude']


def get_interpolation_function(array_with_nans):
    # get interpolation function
    array_without_nans = array_with_nans[~np.isnan(array_with_nans)]
    index_no_nans = np.arange(0, len(array_without_nans), 1)
    order = 1
    interpolation_func = InterpolatedUnivariateSpline(x=index_no_nans, y=array_without_nans, k=order)
    return interpolation_func


def get_interpolation_function_distance(array_with_nans, coords):
    utms_coords = utm_coords()
    _transect_utils.coords2distance()
    # get interpolation function
    array_without_nans = array_with_nans[~np.isnan(array_with_nans)]
    index_no_nans = np.arange(0, len(array_without_nans), 1)
    order = 1

    interpolation_func = InterpolatedUnivariateSpline(x=index_no_nans, y=array_without_nans, k=order)

    return interpolation_func




def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike


def extrapolate_left(array_with_nans, line, interpolation_func, num_points_for_extrap):
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

    array_without_nans = array_with_nans[~np.isnan(array_with_nans)]
    x = np.arange(0, 5, 1)
    y= array_with_nans[x]

    interpolation_func = InterpolatedUnivariateSpline(x=x, y=y, k=1)

    # # make interp function
    # x = np.arange(0, num_points_for_extrap, 1)
    # y = array_with_nans[x]
    #
    # #interpolation_func = interp1d(x=x, y=y)
    # interpolation_func = InterpolatedUnivariateSpline(x, y, k=1)  # k 1 for linear
    # xnew = np.arange(nan_count, 0, -1)
    # print('xnew')
    # print(xnew)
    # ynew = interpolation_func(xnew)
    # print(ynew)
    # plt.plot(x, y, 'o', xnew, ynew, '-')
    # plt.show()
    print('extrap left before')
    print(array_to_extrapoltate)

    # Extrapolate with linear interpolation function.
    y = interpolation_func(array_to_extrapoltate)


    print('extrap left after')
    print(y)
    # attach extrapolated array onto existing array
    values_to_change = array_with_nans[index_of_extrap_values]
    print("VALUES TO CHANGE")
    print(values_to_change)
    array_with_nans[index_of_extrap_values] = y
    log_changes_in_csv(nan_count, line, index_of_extrap_values, values_to_change, y,  "extrapolate_left")



    # assert that the difference between the new values are fairly similar

    return array_with_nans, index_of_extrap_values


def extrapolate_right(array_w_nans, line, num_existing_points_use_for_extrap=3):
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

    last_most_index = length_of_array_with_nans - 1
    array_to_extrapoltate = np.arange(length_of_array_with_nans - nan_count, length_of_array_with_nans, 1)

    index_of_extrap_values = np.arange(length_of_array_with_nans - nan_count - 1, length_of_array_with_nans - 1, 1)
    values_to_change = array_w_nans[array_to_extrapoltate]
    print("index_of_extrap_values")
    print(index_of_extrap_values)

    # get extrapolation function
    print(len(array_w_nans))
    print(array_w_nans[len(array_w_nans) -1])

    # how many points back to use for extrap method?
    start_index = len(array_w_nans) - 1 - num_existing_points_use_for_extrap - 1
    array_to_use_for_interp = np.arange(start_index, len(array_w_nans) - 1, 1)
    print('array_to_use_for_interp')
    print(array_to_use_for_interp)

    #last_measured_value = array_w_nans[index_of_extrap_values[0] -1]
    t = np.arange(4,8,1)
    #f = InterpolatedUnivariateSpline(array_to_use_for_interp, array_w_nans[array_to_use_for_interp], k=1)
    f = interp1d(array_to_use_for_interp, array_w_nans[array_to_use_for_interp], fill_value='extrapolate')


    print("HERERER")
    print(f([7]))
    print(f([9]))
    print(f([-1]))
    print(f([2098]))
    print(f([2100]))

    array_w_nans[array_to_extrapoltate] = f(index_of_extrap_values)




    #interpolation_func = interp1d(array_to_use_for_interp, array_w_nans[array_to_use_for_interp], fill_value="extrapolate")

    #  Extrapolate with linear interpolation function.
    #y = interpolation_func(array_to_extrapoltate)
    #array_w_nans[array_to_extrapoltate] = y
    logging.debug('array_to_extrapoltate_right')
    logging.debug(array_to_extrapoltate)

    #  np.append(y, array_with_nans)
    #  attach extrapolated array onto existing array

    #array_w_nans[index_of_extrap_values] = y[index_of_extrap_values]
    log_changes_in_csv(nan_count, line, index_of_extrap_values, values_to_change, array_w_nans[array_to_extrapoltate], "extrapolate_right")

    assert_extrapolation_correct(array_w_nans, index_of_extrap_values, array_to_use_for_interp)

    return array_w_nans, array_to_extrapoltate



def assert_extrapolation_correct(complete_array, index_of_exptrapolate_values, array_to_use_for_interp):
    print("HERE DOG")
   # print((index_of_extrap_values[0]) - 1)

   # last_measured_value_index = index_of_extrap_values[0] - 1

   # print(last_measured_value_index)
    print("index array_to use for interp {}".format(array_to_use_for_interp))
    print("array_to use for interp {}".format(complete_array[array_to_use_for_interp]))
    test_array_indexes = np.insert(index_of_exptrapolate_values, 0, array_to_use_for_interp)
    print("index_extrap values: {}".format(index_of_exptrapolate_values))
    print("extrap values: {}".format(complete_array[index_of_exptrapolate_values]))

    test_array = complete_array[test_array_indexes]
    print(test_array)

    value_index = 0
    while value_index < len(test_array) - 2:
        difference = test_array[value_index] - test_array[value_index + 1]
        print("{} - {} = {}".format(test_array[value_index],  test_array[value_index + 1], difference))
        assert difference < 0.1

        value_index = value_index + 1


def fill_in_masked_values_with_interpolated_values(narray, interp_func, line):
    is_nan_bool_array, x = nan_helper(narray)

    ndarray2 = narray.copy()
    interpolate_indexes = (is_nan_bool_array.nonzero()[0])

    if interpolate_indexes is not None: # if are points to interpolate then interpolate them
        narray[is_nan_bool_array] = interp_func(x(is_nan_bool_array))
        log_changes_in_csv(len(narray[is_nan_bool_array]), line, interpolate_indexes, ndarray2[interpolate_indexes], narray[interpolate_indexes], "interpolate")
        #interpolate_indexes2 = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        return narray, interpolate_indexes
    return narray, []

def interpolate():
    pass


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    #print(vals)
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
    print(result)
    return result



def fill_in_the_blanks(narray, num_points_each_way_to_use_for_interp_func):
    print("HERE")

    is_nan_bool_array, x = nan_helper(narray)
    ndarray2 = narray.copy()
    interpolate_indexes = (is_nan_bool_array.nonzero()[0])
    if len(interpolate_indexes) > 0:  # if are points to interpolate then interpolate them


        groups_of_values_to_interp = (group_consecutives(interpolate_indexes))
        print("Group consecutives")
        print(groups_of_values_to_interp)

        for group_to_interp in groups_of_values_to_interp:
            # interpolate the group_to_interp
            min_index = np.min(group_to_interp)
            max_index = np.max(group_to_interp)




            i = 1
            indxes_ = np.array([], dtype= int)
            while i <= num_points_each_way_to_use_for_interp_func:
                indxes_ = np.append(indxes_, min_index - i)
                indxes_ = np.append(indxes_, max_index + i)
                i = i + 1
            # i = 0
            # while i > num_points_each_way_to_use_for_interp_func:
            #     indxes_[i] = min_index + i
            #     i = i + 1
            print('indxes_')
            print(np.sort(indxes_))
            sorted_index = np.sort(indxes_)

            x = sorted_index
            y = narray[sorted_index]
            print("X: {}".format(x))
            print("Y: {}".format(y))

            interpolation_func = interp1d(x, y)

            xnew = group_to_interp
            ynew = interpolation_func(xnew)

            print("xnew: {}".format(xnew))
            print("ynew: {}".format(ynew))

            # plt.plot(x, y, 'o', xnew, ynew, '-')
            # plt.show()

            # put the interp points in
            print("old")
            print(narray[xnew])
            narray[xnew] = ynew
            print("new")
            print(narray[xnew])

    else:
        return narray, []

    return narray, interpolate_indexes
# def fill_in_the_blanks2(narray):


def get_array_for_prediction_type_variable(var_array_including_nan, interpolated_index, extrapolate_left_indexes, extrapolate_right_indexes):

    a = np.zeros(len(var_array_including_nan), dtype=int)
    a[interpolated_index] = 1
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

    var_array_including_nan[10:12] = np.nan
    var_array_including_nan[15:17] = np.nan
    var_array_including_nan[0:4] = np.nan
    var_array_including_nan[-1] = np.nan
    # logging.debug('var_array_including_nan')
    # logging.debug(var_array_including_nan)

    extrapolate_left_indexes = None
    extrapolate_right_indexes = None

    interp_func = get_interpolation_function(var_array_including_nan)
    #interp_func = get_interpolation_function_distance(var_array_including_nan, coords)

    if np.isnan(var_array_including_nan[0]):
        # then the first value is nan and extrapolation is required
        var_array_including_nan, extrapolate_left_indexes = extrapolate_left(var_array_including_nan, line, interp_func, 5)

    if np.isnan(var_array_including_nan[-1]):
        logging.debug(var_array_including_nan[-1])
        var_array_including_nan, extrapolate_right_indexes = extrapolate_right(var_array_including_nan, line, 3)

    var_array_including_nan, interpolated_index = fill_in_the_blanks(var_array_including_nan, 1)
   # var_array_including_nan, interpolated_index = fill_in_masked_values_with_interpolated_values(var_array_including_nan, interp_func, line)

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
            #######
            # remove line from dataset?
            #######
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
    # netcdf_input_path = sys.argv[1]
    # nc_output_dataset_path = sys.argv[2]
    for root, dirs, files in os.walk(input_netcdf_folder_path):
      #  print('root')
       # print(root)
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
                        print("HERE")
                        print(src[name][:])

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
