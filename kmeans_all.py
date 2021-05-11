import sys
import numpy as np
import sunpy.io
import h5py
import time
from helita.io.lp import *
import daal4py as d4p
import tables as tb
from mpi4py import MPI


# weights = np.array(
#     [
#         0.01163262, 0.01156011, 0.01134999, 0.0112064 , 0.01130734,
#         0.01225208, 0.0117292 , 0.01231974, 0.01335993, 0.01600201,
#         0.01743874, 0.02022419, 0.02158955, 0.02184081, 0.02165721,
#         0.02116544, 0.02008175, 0.01908124, 0.01721239, 0.01488855,
#         0.01319056, 0.01222358, 0.01169559, 0.01138442, 0.01133857,
#         0.01153457, 0.01175034, 0.01205306, 0.01241954, 0.01478657,
#         0.01760499, 0.0154747 , 0.01449769, 0.01396424, 0.01432799,
#         0.01721551, 0.01987806, 0.02105709, 0.02198457, 0.02294375,
#         0.02242386, 0.02127857, 0.02134655, 0.02094614, 0.0168878 ,
#         0.01406624, 0.01384265, 0.01496679, 0.0166017 , 0.01611505,
#         0.01246966, 0.01246273, 0.01268374, 0.01410584, 0.01827967,
#         0.01983389, 0.01919988, 0.01529413, 0.01419569, 0.01350287,
#         0.01268345, 0.01242869, 0.01251868, 0.01264109
#     ]
# )

weights = np.array(
    [
        0.01991948, 0.02004443, 0.02041552, 0.02067709, 0.02049252,
        0.01891237, 0.01975547, 0.01880849, 0.01734409, 0.01448042,
        0.01328742, 0.01145736, 0.01073278, 0.01060931, 0.01069924,
        0.01094784, 0.01153863, 0.01214365, 0.01346215, 0.01556335,
        0.01756679, 0.01895647, 0.01981224, 0.02035377, 0.02043608,
        0.02008881, 0.01971992, 0.01922464, 0.01865736, 0.01567069,
        0.01316194, 0.01497385, 0.01598295, 0.01659352, 0.01617225,
        0.01345971, 0.01165686, 0.01100417, 0.01053993, 0.0100993 ,
        0.01033345, 0.01088963, 0.01085495, 0.01106246, 0.0137209 ,
        0.01647319, 0.01673927, 0.015482  , 0.01395735, 0.01437885,
        0.01858237, 0.0185927 , 0.01826873, 0.01642694, 0.01267615,
        0.01168282, 0.01206861, 0.01515063, 0.01632296, 0.01716048,
        0.01826915, 0.01864362, 0.0185096 , 0.01833037
    ]
)

outfilename = '/data/harsh/sub_fov_result_kmeans_whole_data_inertia_inverted_weights.h5'

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

    whole_data = (whole_data - mn) / sd

    whole_data *= weights

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

    result = algo.compute(whole_data, initial_centroids)

    log(
        'Process {}: Finished KMeans'.format(
            d4p.my_procid()
        )
    )

    algo = d4p.kmeans(nClusters, 0, assignFlag=True)

    assignments = algo.compute(whole_data, result.centroids).assignments

    log(
        'Process {}: Asigned Labels to training data'.format(
            d4p.my_procid()
        )
    )

    whole_data /= weights
    whole_data = whole_data * sd + mn

    local_clusters = np.zeros((100, 30 + 20 + 14))
    squarred_differences = np.zeros((100, 30 + 20 + 14))
    total_numbers = np.zeros(100)
    for i in range(100):
        ind = np.where(assignments == i)[0]

        local_clusters[i] = np.sum(whole_data[ind], 0)
        squarred_differences[i] = np.sum(
            np.square(
                (
                    (
                        (
                            whole_data[ind] - mn
                        ) / sd
                    ) * weights
                ) - result.centroids[i]
            ),
            0
        )
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

    return (a, b, c, final_labels, mn, sd, data_part, extradata, total_numbers, local_clusters, squarred_differences, assignments, result)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    status = MPI.Status()

    d4p.daalinit()

    (a, b, c, final_labels, mn, sd, data_part, extradata, total_numbers, local_clusters, squarred_differences, assignments, result) = do_kmeans()

    comm.Barrier()

    if d4p.my_procid() == 0:

        f = tb.open_file(outfilename, mode='w', title='KMeans Data')

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
                outfilename
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
            squarred_differences = np.add(squarred_differences, data_dict['squarred_differences'])

            k -= 1

            log(
                'K: {}'.format(
                    k
                )
            )

        log(
            'Out of for loop'
        )
        rps = local_clusters / total_numbers[:, np.newaxis]

        inertia = np.sqrt(squarred_differences) / total_numbers[:, np.newaxis]

        f = tb.open_file(outfilename, mode='a')

        f.create_array(f.root.columns, 'rps', rps, "Representative Profiles")

        f.create_array(f.root.columns, 'inertia', inertia, "Inertia")

        f.close()

        log(
            'Process {}: Updated File {}'.format(
                d4p.my_procid(),
                outfilename
            )
        )

    else:
        comm.recv(
            source=0,
            tag=4,
            status=status
        )
        comm.send({}, dest=0, tag=0)
        f = tb.open_file(outfilename, mode='a')
        labels = f.root.columns.assignments
        labels[data_part * d4p.my_procid() + extradata: data_part * d4p.my_procid() + data_part + extradata] = assignments[:, 0]
        flabels = f.root.columns.final_labels
        flabels[a, b, c] = final_labels[:, 0]
        f.close()
        log(
            'Process {}: Updated File {}'.format(
                d4p.my_procid(),
                outfilename
            )
        )
        comm.send({'total_numbers':total_numbers, 'local_clusters': local_clusters, 'squarred_differences': squarred_differences}, dest=0, tag=1)
        log(
            'Process {}: Send message with tag 1'.format(
                d4p.my_procid()
            )
        )
