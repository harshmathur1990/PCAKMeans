import sys
import time
import numpy as np
import sunpy.io
import h5py
from mpi4py import MPI
from pathlib import Path

base_path = Path('/home/harsh/OsloAnalysis')
new_kmeans = base_path / 'new_kmeans'
input_file_3950 = base_path / 'nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
old_kmeans_file = new_kmeans / 'out_100_0.5_0.5_n_iter_10000_tol_1en5.h5'

weak_shocks_profiles = np.array(
    [
        6, 57, 10, 80, 49, 56, 98, 96, 87, 9, 91, 23,  5, 12, 65, 67,
        92
    ]
)

medium_shocks_profiles = np.array(
    [
        1, 55, 39, 22, 94, 30, 54, 93, 17, 77, 26, 72, 52, 19, 79, 37, 4
    ]
)

strong_shocks_profiles = np.array(
    [
        85, 36, 18, 78
    ]
)

quiet_profiles = np.array(
    [
        0, 8, 11, 14, 15, 20, 21, 24, 28, 29, 31, 34, 42, 43, 44, 47, 48, 51, 58, 60, 61, 62, 63, 66, 69, 70, 73, 74, 75, 82, 84, 86, 89, 90, 99
    ]
)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        sys.stdout.write(
            '{} : {} ms\n'.format(
                method.__name__, (te - ts) * 1000
            )
        )
        return result
    return timed


# @timeit
def do_work(index):
    i, j = np.unravel_index(index, (1236, 1848))
    profile_array = data[:, 0, 0:29, i, j]
    classify_array = f['new_final_labels'][:, i, j]

    data_t = list()

    data_intensity_enhancement = list()

    start = False
    start_indice = -1
    for i in range(100):
        if classify_array[i] in quiet_profiles:
            start = True
            start_indice = i
        elif classify_array[i] in all_shock_profiles:

            if start_indice == -1 or start == False:
                continue

            minima_points = np.r_[True, profile_array[i][1:] < profile_array[i][:-1]] & np.r_[profile_array[i][:-1] < profile_array[i][1:], True]

            maxima_points = np.r_[False, profile_array[i][1:] > profile_array[i][:-1]] & np.r_[profile_array[i][:-1] > profile_array[i][1:], False]

            minima_indices = np.where(minima_points == True)[0]

            maxima_indices = np.where(maxima_points == True)[0]

            if maxima_indices.size <= 0:
                continue

            maxima_indice = maxima_indices[
                np.argmax(
                    profile_array[i][
                        maxima_indices
                    ]
                )
            ]

            mn = np.where(
                minima_indices <= maxima_indice
            )[0]

            if mn.size <= 0:
                continue

            minima_indice = minima_indices[mn[-1]]

            if minima_indice < maxima_indice and maxima_indice <= 15:

                shock_intensity = (profile_array[i][maxima_indice] - profile_array[i][minima_indice]) / profile_array[i][minima_indice]

                data_t.append(i - start_indice)

                data_intensity_enhancement.append(shock_intensity)

                # print (
                #         '{}-{}'.format(
                #             i, start_indice
                #         )
                #     )

        else:
            start = False

    return data_t, data_intensity_enhancement


# @timeit
def write_to_file(f1, f2, dt, de):
    np.savetxt(f1, dt)
    np.savetxt(f2, de)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data, header = sunpy.io.fits.read(input_file_3950, memmap=True)[0]
    f = h5py.File(old_kmeans_file, 'r')
    all_shock_profiles = list(weak_shocks_profiles) + list(medium_shocks_profiles) + list(strong_shocks_profiles)

    if rank == 0:
        status = MPI.Status()
        waiting_queue = set()
        running_queue = set()
        data_t = list()
        data_enhance = list()
        f1 = open('data_t.txt', 'a')
        f2 = open('data_enhance.txt', 'a')

        for i in range(1236 * 1848):
            waiting_queue.add(i)

        for worker in range(1, size):
            if len(waiting_queue) == 0:
                break
            item = waiting_queue.pop()
            work_type = {
                'job': 'work',
                'item': item
            }
            comm.send(work_type, dest=worker, tag=1)
            running_queue.add(item)

        sys.stdout.write('Finished First Phase\n')

        while len(running_queue) != 0 or len(waiting_queue) != 0:
            try:
                status_dict = comm.recv(
                    source=MPI.ANY_SOURCE,
                    tag=2,
                    status=status
                )
            except Exception:
                sys.stdout.write('Failed to get\n')
                sys.exit(1)

            sender = status.Get_source()
            item = status_dict['item']
            sys.stdout.write(
                'Sender: {} item: {}\n'.format(
                    sender, item
                )
            )
            running_queue.discard(item)
            dt = status_dict['data_t']
            de = status_dict['data_enhance']

            write_to_file(f1, f2, dt, de)

            if len(waiting_queue) != 0:
                new_item = waiting_queue.pop()
                work_type = {
                    'job': 'work',
                    'item': new_item
                }
                comm.send(work_type, dest=sender, tag=1)
                running_queue.add(new_item)

        for worker in range(1, size):
            work_type = {
                'job': 'stopwork'
            }
            comm.send(work_type, dest=worker, tag=1)

        f1.close()
        f2.close()
    else:
        while 1:
            work_type = comm.recv(source=0, tag=1)

            if work_type['job'] != 'work':
                break

            item = work_type['item']

            data_t, data_enhance = do_work(item)

            comm.send(
                {
                    'item': item,
                    'data_t': data_t,
                    'data_enhance': data_enhance
                },
                dest=0,
                tag=2
            )

