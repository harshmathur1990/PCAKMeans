import numpy as np
import h5py
import tables as tb
from mpi4py import MPI


file = '/data/harsh/sub_fov_result_kmeans_whole_data_inertia_inverted_weights.h5'

selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])
old_kmeans_file = '/data/harsh/out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'
mask_file_crisp = '/data/harsh/crisp_chromis_mask_2019-06-06.fits'
input_file_3950 = '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
input_file_8542 = '/data/harsh/nb_8542_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'
input_file_6173 = '/data/harsh/nb_6173_aligned_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fcube'


def mainfunc(label):
    f = h5py.File(file, 'r')

    inertia = np.mean(f['columns']['inertia'][()], 1)

    f.close()
    return inertia[label]


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    for k in range(100):
        if rank == 0:
            f = h5py.File(file, 'r')

            inertia = np.mean(f['columns']['inertia'][()], 1)
            
            data_part = 100 * 1236 * 1848 // size

            extradata = 100 * 1236 * 1848 % size

            order_array = np.arange(100 * 1236 * 1848, dtype=np.int64).reshape(100, 1236, 1848)

            for i in range(1, size):
                a, b, c = np.where(
                    (order_array >= (i * data_part_all + extradata_all)) &
                    (order_array < (i * data_part_all + data_part_all + extradata_all))
                )
                comm.send(
                    {
                        'inertia': inertia
                        'a': a,
                        'b': b,
                        'c': c
                    },
                    dest=i,
                    tag=0
                )

            labels = f['columns']['final_labels'][()][a, b, c]
            a, b, c = np.where(order_array < data_part_all + extradata_all)
            del order_array
            f.close()
        else:
            data_dict = comm.recv(
                source=0,
                tag=0,
                status=status
            )
            labels = data_dict['labels']
            inertia = data_dict['inertia']

        inertia_array = inertia[labels]

