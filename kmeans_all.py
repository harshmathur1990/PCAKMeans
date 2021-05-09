import sys
import numpy as np
import sunpy.io
import h5py
import time
from helita.io.lp import *
import daal4py as d4p
import tables as tb
from mpi4py import MPI

y = 0.011995
x = 0.022555
weights = np.ones(30 + 20 + 14) * y
weights[10:20] = x
weights[30 + 4:30 + 16] = x


def log(logString):
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
    sys.stdout.write(
        '[{}] {}\n'.format(
            current_time,
            logString
        )
    )


def do_kmeans(method='plusPlusDense'):

    old_kmeans_file = '/data/harsh/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
    input_file_3950 = '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
    input_file_8542 = '/data/harsh/nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
    input_file_6173 = '/data/harsh/nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
    mask_file_crisp = '/data/harsh/crisp_chromis_mask_2019-06-06.fits'
    if d4p.my_procid() == 0:
        selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])

        mask = np.transpose(
            sunpy.io.fits.read(mask_file_crisp, memmap=True)[0][0],
            axes=(2, 1, 0)
        )

        a, b, c = np.where(mask[selected_frames] == 1)

        data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]

        whole_data = np.zeros((a.size, 30 + 20 + 14))

        whole_data[:, 0:30] = data[selected_frames][a, 0, :, b, c]

        sh, dt, header = getheader(input_file_8542)
        data = np.memmap(
            input_file_8542,
            mode='r',
            shape=sh,
            dtype=dt,
            order='F',
            offset=512
        )

        data = np.transpose(
            data.reshape(1848, 1236, 100, 4, 20),
            axes=(2, 3, 4, 1, 0)
        )

        whole_data[:, 30:30 + 20] = data[selected_frames][a, 0, :, b, c]

        sh, dt, header = getheader(input_file_6173)
        data = np.memmap(
            input_file_6173,
            mode='r',
            shape=sh,
            dtype=dt,
            order='F',
            offset=512
        )

        data = np.transpose(
            data.reshape(1848, 1236, 100, 4, 14),
            axes=(2, 3, 4, 1, 0)
        )

        whole_data[:, 30 + 20:30 + 20 + 14] = data[selected_frames][a, 0, :, b, c]

        mn = np.mean(whole_data, 0)

        sd = np.std(whole_data, 0)

        f = h5py.File(old_kmeans_file, 'r')

        labels = f['new_final_labels'][selected_frames][a, b, c]

        f.close()

        whole_data = (whole_data - mn) / sd
        whole_data *= weights

        initial_centroids= np.zeros((100, 30 + 20 + 14))

        for i in range(100):
            ind = np.where(labels == i)[0]

            initial_centroids[i] = np.mean(whole_data[ind], 0)

        whole_data /= weights
        whole_data = whole_data * sd + mn

        data_part = a.size // d4p.num_procs()

        extradata = a.size % d4p.num_procs()

        for i in range(1, d4p.num_procs()):
            comm.send(
                {
                    'whole_data': whole_data[data_part * i + extradata: data_part * i + data_part + extradata, :],
                    'mn': mn,
                    'sd': sd,
                    'data_part': data_part,
                    'extradata': extradata,
                    'initial_centroids': initial_centroids
                },
                dest=i,
                tag=3
            )
        log(
            'Process 0: Distributed actual data'
        )
        whole_data = whole_data[0:data_part + extradata]
    else:
        data_dict = comm.recv(
            source=0,
            tag=3,
            status=status
        )
        whole_data = data_dict['whole_data']
        mn = data_dict['mn']
        sd = data_dict['sd']
        data_part = data_dict['data_part']
        extradata = data_dict['extradata']
        initial_centroids = data_dict['initial_centroids']
        log(
            'Process {}: Received actual data'.format(
                d4p.my_procid()
            )
        )

    nClusters = 100

    maxIter = 10000

    accuracyThreshold = 1e-6

    data_local = (whole_data - mn) / sd

    data_local *= weights

    log(
        'Process {}: Finished Loading data, proceeding for kmeans'.format(
            d4p.my_procid()
        )
    )

    algo = d4p.kmeans(
        nClusters,
        maxIter,
        accuracyThreshold=accuracyThreshold,
        distributed=True
    )

    result = algo.compute(data_local, initial_centroids)

    log(
        'Process {}: Finished KMeans'.format(
            d4p.my_procid()
        )
    )

    algo = d4p.kmeans(nClusters, 0, assignFlag=True)

    assignments = algo.compute(data_local, result.centroids).assignments

    log(
        'Process {}: Asigned Labels to training data'.format(
            d4p.my_procid()
        )
    )

    whole_data /= weights
    whole_data = whole_data * sd + mn

    local_clusters = np.zeros((100, 30 + 20 + 14))
    total_numbers = np.zeros(100)
    for i in range(100):
        ind = np.where(assignments == i)[0]

        local_clusters[i] = np.sum(whole_data[ind], 0)
        total_numbers[i] = ind.size

    log(
        'Process {}: Calculated the sum of profiles and frequencies'.format(
            d4p.my_procid()
        )
    )

    data_part_all = 100 * 1236 * 1848 // d4p.num_procs()
    extradata_all = 100 * 1236 * 1848 % d4p.num_procs()

    if d4p.my_procid() == 0:
        order_array = np.arange(100 * 1236 * 1848, dtype=np.int64).reshape(100, 1236, 1848)
        for i in range(1, d4p.num_procs()):
            a, b, c = np.where(
                (order_array >= (i * data_part_all + extradata_all)) &
                (order_array < (i * data_part_all + data_part_all + extradata_all))
            )
            comm.send(
                {
                    'a': a,
                    'b': b,
                    'c': c
                },
                dest=i,
                tag=5
            )

        whole_data = np.zeros((data_part_all + extradata_all, 64))
        a, b, c = np.where(order_array < data_part_all + extradata_all)
        del order_array
    else:
        data_dict = comm.recv(
            source=0,
            tag=5,
            status=status
        )
        whole_data = np.zeros((data_part_all, 64))
        a = data_dict['a']
        b = data_dict['b']
        c = data_dict['c']

    data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]

    whole_data[:, 0:30] = data[a, 0, :, b, c]

    sh, dt, header = getheader(input_file_8542)
    data = np.memmap(
        input_file_8542,
        mode='r',
        shape=sh,
        dtype=dt,
        order='F',
        offset=512
    )

    data = np.transpose(
        data.reshape(1848, 1236, 100, 4, 20),
        axes=(2, 3, 4, 1, 0)
    )

    whole_data[:, 30:30 + 20] = data[a, 0, :, b, c]

    sh, dt, header = getheader(input_file_6173)
    data = np.memmap(
        input_file_6173,
        mode='r',
        shape=sh,
        dtype=dt,
        order='F',
        offset=512
    )

    data = np.transpose(
        data.reshape(1848, 1236, 100, 4, 14),
        axes=(2, 3, 4, 1, 0)
    )

    whole_data[:, 30 + 20:30 + 20 + 14] = data[a, 0, :, b, c]

    whole_data = (whole_data - mn) / sd
    whole_data *= weights

    log(
        'Process {}: Loaded Partitioned data from whole full data'.format(
            d4p.my_procid()
        )
    )

    algo = d4p.kmeans(nClusters, 0, assignFlag=True)

    final_labels = algo.compute(whole_data, result.centroids).assignments

    log(
        'Process {}: Assigned labels to whole partitioned data'.format(
            d4p.my_procid()
        )
    )

    return (a, b, c, final_labels, mn, sd, data_part, extradata, total_numbers, local_clusters, assignments, result)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    status = MPI.Status()

    d4p.daalinit()

    (a, b, c, final_labels, mn, sd, data_part, extradata, total_numbers, local_clusters, assignments, result) = do_kmeans()

    if d4p.my_procid() == 0:

        f = tb.open_file('/data/harsh/result_kmeans_whole_data.h5', mode='w', title='KMeans Data')

        gcolumns = f.create_group('/', "columns", "KMeans arrays")

        f.create_array(gcolumns, 'assignments', np.zeros((d4p.num_procs() * data_part + extradata)), "Labels")

        labels = f.root.columns.assignments

        labels[0:assignments.shape[0]] = assignments[:, 0]

        f.create_array(gcolumns, 'final_labels', np.zeros((100, 1236, 1848)), "Labels")

        flabels = f.root.columns.final_labels

        flabels[a, b, c] = final_labels[:, 0]

        f.create_array(gcolumns, 'centroids', result.centroids, "Centroids")

        f.create_array(gcolumns, 'objectiveFunction', result.objectiveFunction, "Objective Function")

        f.create_array(gcolumns, 'nIterations', result.nIterations, "Number of Iterations")

        f.create_array(gcolumns, 'mn', mn, "Mean of samples")

        f.create_array(gcolumns, 'sd', sd, "Std of Samples")

        f.create_array(gcolumns, 'weights', weights, "Wavelength weights")

        f.close()

        log(
            'Process 0: Created File {}'.format(
                '/data/harsh/result_kmeans_whole_data.h5'
            )
        )

        for i in range(1, d4p.num_procs()):
            comm.send({}, dest=i, tag=4)

        k = d4p.num_procs() - 2

        while (k >= 0):
            comm.recv(
                source=MPI.ANY_SOURCE,
                tag=0,
                status=status
            )
            sender = status.Get_source()
            data_dict = comm.recv(
                source=sender,
                tag=1,
                status=status
            )

            total_numbers = np.add(total_numbers, data_dict['total_numbers'])
            local_clusters = np.add(local_clusters, data_dict['local_clusters'])

            k -= 1

        rps = local_clusters / total_numbers[:, np.newaxis]

        f = tb.open_file('/data/harsh/result_kmeans_whole_data.h5', mode='a')

        f.create_array(f.root.columns, 'rps', rps, "Representative Profiles")

        f.close()

        log(
            'Process {}: Updated File {}'.format(
                d4p.my_procid(),
                '/data/harsh/result_kmeans_whole_data.h5'
            )
        )

    else:
        comm.recv(
            source=0,
            tag=4,
            status=status
        )
        comm.send({}, dest=0, tag=0)
        f = tb.open_file('/data/harsh/result_kmeans_whole_data.h5', mode='a')
        labels = f.root.columns.assignments
        labels[data_part * d4p.my_procid() + extradata: data_part * d4p.my_procid() + data_part + extradata] = assignments[:, 0]
        flabels = f.root.columns.final_labels
        flabels[a, b, c] = final_labels[:, 0]
        f.close()
        log(
            'Process {}: Updated File {}'.format(
                d4p.my_procid(),
                '/data/harsh/result_kmeans_whole_data.h5'
            )
        )
        comm.send({'total_numbers':total_numbers, 'local_clusters': local_clusters}, dest=0, tag=1)
