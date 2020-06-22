# import os
import sys
import enum
import traceback
import numpy as np
# import joblib
import os
import os.path
import sunpy.io
import h5py
# from dask.distributed import LocalCluster, Client
from mpi4py import MPI
from sklearn.cluster import KMeans
from delay_retry import retry


kmeans_output_dir = '/data/harsh1/kmeans_output'
input_file = '/data/harsh1/colabd/nb_3950_2019-' + \
    '06-06T10:26:20_scans=0-99_corrected_im.fits'


# kmeans_output_dir = '/Users/harshmathur/CourseworkRepo/tst'
# input_file = '/Users/harshmathur/CourseworkRepo/tst/sample_input.h5'
# input_key = 'selected_data'


class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


def save_model(fout, model):
    fout['cluster_centers_'] = model.cluster_centers_
    fout['labels_'] = model.labels_
    fout['inertia_'] = model.inertia_
    fout['n_iter_'] = model.n_iter_


@retry((Exception,))
def get_value(input_file):
    # f = h5py.File(input_file, 'r', driver='mpio', comm=MPI.COMM_WORLD)
    indices = [0, 11, 25, 36, 60, 78, 87]
    total_indices = list()

    for i in indices:
        total_indices += list(
            np.arange(i * 2284128, (i + 1) * 2284128)
        )

    total_indices = np.array(total_indices)

    data, header = sunpy.io.read_file(input_file)[0]

    new_arr = np.lib.stride_tricks.as_strided(
        data, shape=(100 * 1236 * 1848, 30),
        strides=(4, 4 * 1848 * 1236)
    )

    selected_data = new_arr[total_indices]

    selected_data = selected_data.astype(np.float64)

    mean = np.mean(selected_data, 0)

    std = np.std(selected_data, 0)

    mean = mean[np.newaxis, :]

    std = std[np.newaxis, :]

    mean_repeat = np.repeat(
        mean,
        axis=0,
        repeats=selected_data.shape[0]
    )

    std_repeat = np.repeat(
        std,
        axis=0,
        repeats=selected_data.shape[0]
    )

    selected_data = np.divide(
        np.subtract(
            selected_data,
            mean_repeat
        ),
        std_repeat
    )

    return selected_data, mean_repeat[0], std_repeat[0]


def do_work(num_clusters):
    sys.stdout.write('Processing for Num Clusters: {}\n'.format(num_clusters))
    # try:
    #     n_workers = os.environ['NWORKER']
    # except KeyError:
    #     n_workers = None

    # try:
    #     threads_per_worker = os.environ['NTHREADS']
    # except KeyError:
    #     threads_per_worker = None

    # cluster = LocalCluster(
    #     # n_workers=n_workers, threads_per_worker=threads_per_worker
    # )

    # client = Client(cluster)

    try:
        data, mean, std = get_value(input_file)

        sys.stdout.write('Process: {} Read from File\n'.format(num_clusters))
        model = KMeans(n_clusters=num_clusters)
        # with joblib.parallel_backend('dask'):
        sys.stdout.write('Process: {} Before KMeans\n'.format(num_clusters))
        model.fit(data)

        sys.stdout.write('Process: {} Fitted KMeans\n'.format(num_clusters))

        fout = h5py.File(
            '{}/out_{}.h5'.format(kmeans_output_dir, item), 'w'
        )
        sys.stdout.write(
            'Process: {} Open file for writing\n'.format(num_clusters)
        )
        save_model(fout, model)
        fout['mean'] = mean
        fout['std'] = std
        sys.stdout.write('Process: {} Wrote to file\n'.format(num_clusters))
        fout.close()
        sys.stdout.write('Success for Num Clusters: {}\n'.format(num_clusters))
        return Status.Work_done
    except Exception:
        sys.stdout.write('Failed for {}\n'.format(item))
        exc = traceback.format_exc()
        sys.stdout.write(exc)
        return Status.Work_failure


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        status = MPI.Status()
        waiting_queue = set()
        running_queue = set()
        finished_queue = set()
        failure_queue = set()

        for i in range(2, 100):
            waiting_queue.add(i)

        filepath = '{}/status_job.h5'.format(kmeans_output_dir)
        if os.path.exists(filepath):
            mode = 'r+'
        else:
            mode = 'w'

        f = h5py.File(filepath, mode)

        if 'finished' in list(f.keys()):
            finished = f['finished'][()]
        else:
            finished = list()

        for index in finished:
            waiting_queue.discard(index)

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

        while len(running_queue) != 0 and len(waiting_queue) != 0:
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
            jobstatus = status_dict['status']
            item = status_dict['item']
            sys.stdout.write(
                'Sender: {} item: {} Status: {}\n'.format(
                    sender, item, jobstatus.value
                )
            )
            running_queue.discard(item)
            if jobstatus == Status.Work_done:
                finished_queue.add(item)
                del f['finished']
                finished.append(item)
                f['finished'] = finished
            else:
                failure_queue.add(item)

            if len(waiting_queue) != 0:
                new_item = waiting_queue.pop()
                work_type = {
                    'job': 'work',
                    'item': new_item
                }
                comm.send(work_type, dest=sender, tag=1)

        f.close()

        for worker in range(1, size):
            work_type = {
                'job': 'stopwork'
            }
            comm.send(work_type, dest=worker, tag=1)

    if rank > 0:
        while 1:
            work_type = comm.recv(source=0, tag=1)

            if work_type['job'] != 'work':
                break

            item = work_type['item']

            status = do_work(item)

            comm.send({'status': status, 'item': item}, dest=0, tag=2)
