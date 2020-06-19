# import os
import sys
import enum
import traceback
import numpy as np
# import joblib
import h5py
# from dask.distributed import LocalCluster, Client
from mpi4py import MPI
from sklearn.cluster import KMeans
from delay_retry import retry


kmeans_output_dir = '/home/harsh/kmeans_output'
input_file = '/home/harsh/selected_samples.h5'
input_key = 'selected_data'


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
def get_value(input_file, input_key):
    f = h5py.File(input_file, 'r')
    value = f[input_key][()]
    f.close()
    return value


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
        value = get_value(input_file, input_key)
        model = KMeans(n_clusters=num_clusters)
        # with joblib.parallel_backend('dask'):
        model.fit(value)

        fout = h5py.File(
            '{}/out_{}.h5'.format(kmeans_output_dir, item), 'w'
        )
        save_model(fout, model)
        fout.close()
        sys.stdout.write('Success for Num Clusters: {}\n'.format(num_clusters))
        return Status.Work_done
    except Exception:
        sys.stdout.write('Failed for {}\n'.format(item))
        exc = traceback.format_exc()
        sys.stdout.write(exc)
        return Status.Work_failure


if __name__ == '__main__':
    np.float64 = np.float32

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

        try:
            f = h5py.File('{}/status_job.h5'.format(kmeans_output_dir), 'r')

            if 'finished' in list(f.keys()):
                finished = f['finished'][()]
            else:
                finished = list()

            for index in finished:
                waiting_queue.discard(index)

            f.close()
        except Exception:
            pass

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

        while len(waiting_queue) != 0 or len(running_queue) != 0:
            status_dict = comm.recv(
                source=MPI.ANY_SOURCE,
                tag=2,
                status=status
            )
            sender = status.Get_source()
            jobstatus = status_dict['status']
            item = status_dict['item']
            running_queue.discard(item)
            if jobstatus == Status.Work_done:
                finished_queue.add(item)
            else:
                failure_queue.add(item)

            if len(waiting_queue) != 0:
                new_item = waiting_queue.pop()
                work_type = {
                    'job': 'work',
                    'item': new_item
                }
                comm.send(work_type, dest=sender, tag=1)

        try:
            f = h5py.File('{}/status_job.h5'.format(kmeans_output_dir), 'a')

            if 'finished' in list(f.keys()):
                finished = f['finished'][()]
                del f['finished']
            else:
                finished = list()

            f['finished'] = finished + list(finished_queue)

            f.close()

        except Exception:
            pass

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
