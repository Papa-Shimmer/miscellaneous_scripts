import netCDF4
import numpy as np
import collections


path = "C:\\Users\\u62231\\Desktop\\airbourne_rad_mag_edited\\EDITED_P411MAG.nc"
netcdf_input_dataset = netCDF4.Dataset(path,
                                       mode="r",
                                       clobber=True,
                                       format='NETCDF4')
np.set_printoptions(precision=12)


print("Equivalent of index_count")
unique, counts = np.unique(netcdf_input_dataset.variables['line_index'][:], return_counts=True)
index_count = dict(zip(unique, counts))
print(index_count)

print("Equivalent of index_line")
index_line = np.arange(len(index_count))
lines_last_index = np.arange(len(index_count))
count_sum = 0
for i in np.arange(len(index_count)):
    index_line[i] = count_sum
    count_sum = count_sum + counts[i]
    lines_last_index[i] = count_sum - 1
print(index_line)

#print("line index: {}".format(netcdf_input_dataset.variables['line_index'][:]))
#index_line = np.where(netcdf_input_dataset.variables['line_index'][:][:-1] != netcdf_input_dataset.variables['line_index'][:][1:])[0]

print('Equivalent of longitude_first')
longitude_first = netcdf_input_dataset.variables['longitude'][index_line]
print(longitude_first)

print('Equivalent of longitude_last')
longitude_last = netcdf_input_dataset.variables['longitude'][lines_last_index]
print(longitude_last)

