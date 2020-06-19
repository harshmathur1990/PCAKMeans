import sys
import numpy as np
import netCDF4
from multiprocessing import Pool
from itertools import product


def cluster_with_mean_std(filename_dataset, zscore_filename):

    sys.stdout.write('Started Processing {}\n'.format(filename_dataset))
    f = netCDF4.Dataset(zscore_filename, 'r')

    dataset = netCDF4.Dataset(filename_dataset, 'r+')

    cluster_number = dataset.dimensions['num_clusters'].size

    inertia = 0
    for this_cluster in np.arange(cluster_number):
        indices = np.where(dataset['membership'][()] == this_cluster)
        subtract_mean = np.subtract(
            f['zscaled'][()][indices],
            dataset['clusters'][()][this_cluster]
        )
        squarred_subtract = np.power(
            2,
            subtract_mean
        )
        inertia += np.sqrt(np.sum(squarred_subtract))

    #dataset.createDimension('std', 1)

    #dataset.createVariable('inertia', 'float', dimensions=('std',))

    dataset['inertia'][:] = inertia

    sys.stdout.write('Result {}: {}'.format(filename_dataset, inertia))

    f.close()

    dataset.close()
    sys.stdout.write('Finished Processing {}\n'.format(filename_dataset))


if __name__ == '__main__':
    filename_format = 'out_{}.nc'
    zscore_filename = '/home/harsh/zscaled_nc3.nc'

    indexes = list(range(2, 101))# + list(range(11, 101))i

    filename_list = list()
    for i in indexes:
        filename_list.append(filename_format.format(i))

    p = Pool(32)

    res_list = list()
    for filename in filename_list:
        res_list.append(p.apply_async(cluster_with_mean_std, args = (filename, zscore_filename, )))

    p.close()
    p.join()
