import netCDF4
import os
from collections import Counter
import sys
import numpy as np
import logging
import numpy.testing

logging.basicConfig(level=logging.DEBUG)

np.set_printoptions(threshold=np.nan)

crs_gda94_string = '''GEOGCS["GDA94", DATUM["Geocentric_Datum_of_Australia_1994", 
SPHEROID["GRS 1980",6378137,298.257222101, AUTHORITY["EPSG","7019"]], 
TOWGS84[0,0,0,0,0,0,0], AUTHORITY["EPSG","6283"]], PRIMEM["Greenwich",0, AUTHORITY["EPSG","8901"]], 
UNIT["degree",0.0174532925199433, AUTHORITY["EPSG","9122"]], AUTHORITY["EPSG","4283"]]'''

vars_to_change_standard_name_to_long_name = ['bearing', 'line', 'flag_linetype', 'longitude_first', 'flight',
                                             'latitude_last','point', 'mag_awags', 'bounding_polygon', 'height',
                                             'longitude_last', 'mag_mlev', 'survey', 'fiducial', 'date', 'line',
                                             'mag_lev', 'latitude_first', 'rad_air_dose_rate',
                                             'rad_air_dose_rate_unsmoothed']
var_to_have_long_name = ['longitude', 'latitude']

remove_unit_list = ['line', 'date', 'flight', 'flag_levelling', 'flag_linetype', 'fiducial']

vars_to_exclude = ['point', 'index_line', 'index_count', 'survey']


def get_list_of_line_indexes_for_points(nc_input_dataset):
    """python loop version of writing the line_index variable - much slower than numpy version below."""
    point_dim_len = len(nc_input_dataset.dimensions['point'])

    i = 0
    index_line_index = 0
    point_line_index_list = [None] * point_dim_len  # list to populate with line indexes for each point

    while i < point_dim_len:
        if index_line_index < len(nc_input_dataset.dimensions['line']):
            if i >= nc_input_dataset.variables['index_line'][index_line_index]:  #  and i < nc_output_dataset.variables['index_line'][index_line_index + 1]:
                point_line_index_list[i] = nc_output_dataset.variables['line'][index_line_index]
                point_line_index_list[i] = index_line_index
                i = i + 1
                logging.debug("point: {}".format(i))
                logging.debug("line index: {}".format(nc_input_dataset.variables['line'][index_line_index]))

            if i == nc_input_dataset.variables['index_line'][index_line_index] + nc_input_dataset.variables['index_count'][index_line_index]:
                index_line_index = index_line_index + 1
    return point_line_index_list


def np_get_list_of_line_indexes_for_points(nc_input_dataset):
    """
    Use existing variables index_line and index_count to generate a point dimension variable line_index, which acts
    as a foreign key to the index of values in the line dimension.
    """

    line_start_indexes_array = np.array(nc_input_dataset.variables['index_line'])
    logging.debug("line start indexes: {}".format(line_start_indexes_array))

    line_end_indexes_array = np.array(nc_input_dataset.variables['index_line'] + np.array(nc_input_dataset.variables['index_count']))
    logging.debug("line end indexes: {}".format(line_end_indexes_array))

    new_line_values = np.arange(len(nc_input_dataset.dimensions['line']))
    logging.debug(new_line_values)

    line_index_array = np.zeros(len(nc_input_dataset.dimensions['point']),
                                dtype='int8' if nc_input_dataset.dimensions['line'].size < 128 else 'int32')

    # loop through an array [0, 1, 2....n) length of line dimension.
    for i in np.arange(len(nc_input_dataset.dimensions['line'])):
        start_i = line_start_indexes_array[i]
        end_i = line_end_indexes_array[i]
        line_index_array[start_i:end_i] = i # from the start index to the end index, the value equals i
        logging.debug("Point values from {0} to {1} assigned line_index value of {2}.".format(start_i, end_i, i))

    return line_index_array


def test_list(point_line_index_list, nc_input_dataset):
    logging.debug("Check line_index contains the correct count of points for each line...")
    count_dict = Counter(point_line_index_list)
    logging.debug(count_dict)
    index = 0
    for i in nc_input_dataset.variables['index_count']:
        assert i == count_dict.get(index)
        index = index + 1
    logging.debug("PASSED")

def test_new_line_index(nc_input_dataset, nc_output_dataset):

    logging.debug('Assert numpy array generated from new output netcdf variable "line_index" is equal to '
                  'input netcdf removed variable "index_count".')
    unique, counts = np.unique(nc_output_dataset.variables['line_index'][:], return_counts=True)
    index_count = dict(zip(unique, counts))
    assert np.alltrue(counts == nc_input_dataset.variables['index_count'][:])
    logging.debug("PASSED")

    logging.debug('Assert numpy array generated from new output netcdf variable "line_index" is equal to '
                  'input netcdf removed variable "index_line".')
    line_index = np.arange(len(index_count))
    lines_last_index = np.arange(len(index_count))
    count_sum = 0
    for i in np.arange(len(index_count)):
        line_index[i] = count_sum
        count_sum = count_sum + counts[i]
        lines_last_index[i] = count_sum - 1
    assert np.alltrue(line_index == nc_input_dataset.variables['index_line'][:])
    logging.debug("PASSED")

    longitude_first = nc_output_dataset.variables['longitude'][line_index]
    assert np.alltrue(longitude_first == nc_input_dataset.variables['longitude_first'][:])
    #
    # print('Equivalent of longitude_last')
    # longitude_last = netcdf_input_dataset.variables['longitude'][lines_last_index]
    # print(longitude_last)

def main():

    netcdf_input_path = sys.argv[1]
    nc_output_dataset_path = sys.argv[2]

    netcdf_input_dataset = netCDF4.Dataset(netcdf_input_path,
                                           mode="r+",
                                           clobber=True,
                                           )

    with netcdf_input_dataset as src, netCDF4.Dataset(nc_output_dataset_path, "w") as dst:

        # copy global attributes all at once via dictionary
        logging.info('copying global attributes from input netcdf to output netcdf...')
        dst.setncatts(src.__dict__)
        logging.info(dst.__dict__)

        # copy dimensions
        logging.info('copying dimensions from input netcdf to output netcdf...')
        for name, dimension in netcdf_input_dataset.dimensions.items():
            logging.info("Dimension: {}".format(name))
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None))

        # copy all file data except for the excluded
        for name, variable in src.variables.items():
            array = netcdf_input_dataset.variables[name][:]
            if type(array) == np.ma.core.MaskedArray:
                print('yes')

                print(np.array_equal(array.mask, array.data))
                # print(array.data)
                #print(array.mask)
                i = 0
                mask_index = []
                while i < len(array.data):
                    #print(array.mask[i])
                    if array.mask[i] == True:
                        #print('haHA')
                        print(array.data[i])
                        mask_index.append(i)
                    i = i + 1
                print(mask_index)
                num_masked = len(array.mask)
                print("variable {} has {} masked values".format(name,
                                                                num_masked))

                array.data[mask_index[1]] = 3

                num_masked = len(array.mask)
                print("variable {} has {} masked values".format(name,
                                                            num_masked))

            # take care of the special cases 'point', 'crs'. This is pretty messy but they require once off unique fixes...
            if name == 'point':  # make point_id?
                logging.info('Creating new line_index variable...')
                var_attributes_dict = {"long_name": "zero-based index of value in line",
                                       "lookup": "line"}
                point_line_index_list = np_get_list_of_line_indexes_for_points(netcdf_input_dataset)
                logging.debug(point_line_index_list)
                #test_list(point_line_index_list, netcdf_input_dataset)
                point_line_index_var = dst.createVariable(varname="line_index",
                                                          datatype=point_line_index_list.dtype,
                                                          dimensions="point")
                point_line_index_var[:] = point_line_index_list
                point_line_index_var.setncatts(var_attributes_dict)

            elif name == "crs":
                logging.info('Creating modified crs variable...')
                dst.createVariable(name, 'b')  # type byte and no dimension (scalar)
                edited_dict = src['crs'].__dict__
                logging.debug(edited_dict)
                edited_dict['long_name'] = 'coordinate_reference_system'
                edited_dict['spatial_ref'] = crs_gda94_string
                dst[name].setncatts(edited_dict)

            # add the remaining variables not in the exclusion list.
            elif name not in vars_to_exclude:
                logging.info("Adding variable: {}...".format(name))
                dst.createVariable(name, variable.datatype, variable.dimensions)
                # Change standard names to long names
                if name not in var_to_have_long_name:
                    logging.debug("Changing attribute 'standard_name' to 'long_name' for variable {}".format(name))
                    edited_dict = src[name].__dict__
                    edited_dict['long_name'] = edited_dict.pop('standard_name')

                    # remove units for variables in remove_unit_list
                    if name in remove_unit_list:
                        edited_dict.pop('units')
                else:
                    edited_dict = src[name].__dict__

                    if name in remove_unit_list:
                        logging.debug("removing attribute 'units' from variable: {}".format(name))
                        edited_dict.pop('units')

                dst[name].setncatts(edited_dict)
                dst[name][:] = src[name][:]

            else:
                logging.info("Excluding variable: {}".format(name))

        # survey was excluded previously. Now recreate it as a scalar
        logging.info("Recreating variable survey as a scalar of type byte.")
        survey_scalar_dict = {'long_name': 'survey_number',
                              'original_database_name': 'survey',
                              'survey_number': src.survey_id}
        logging.debug("New 'survey' attributes: {}".format(survey_scalar_dict))
        dst.createVariable('survey', 'b')
        dst['survey'].setncatts(survey_scalar_dict)

        test_new_line_index(src, dst)

if __name__ == "__main__":
    main()
