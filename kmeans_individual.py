import sys
import enum
import traceback
import h5py
import joblib
# from sklearn.cluster import KMeans
from delay_retry import retry
from dask.distributed import Client, LocalCluster
from dask_ml.cluster import KMeans


kmeans_output_dir = '/home/harsh/kmeans_alternate'
input_file = '/home/harsh/selected_samples.h5'
input_key = 'selected_data'
data_arr = None


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
def get_data(input_file, input_key):
    f = h5py.File(input_file, 'r')
    data = f[input_key][()]
    f.close()
    return data


def do_work(num_clusters):

    sys.stdout.write('Processing for Num Clusters: {}\n'.format(num_clusters))

    try:
        global data_arr
        if data_arr is None:
            data_arr = get_data(input_file, input_key)
        sys.stdout.write('Process: {} Read from File\n'.format(num_clusters))
        model = KMeans(n_clusters=num_clusters, init='k-means++', n_jobs=32)
        # with joblib.parallel_backend('dask'):
        sys.stdout.write(
            'Process: {} Before KMeans\n'.format(num_clusters)
        )
        model.fit(data_arr[:, :29])

        sys.stdout.write('Process: {} Fitted KMeans\n'.format(num_clusters))

        fout = h5py.File(
            '{}/out_{}.h5'.format(kmeans_output_dir, num_clusters), 'w'
        )
        sys.stdout.write(
            'Process: {} Open file for writing\n'.format(num_clusters)
        )
        save_model(fout, model)

        sys.stdout.write('Process: {} Wrote to file\n'.format(num_clusters))
        fout.close()
        sys.stdout.write('Success for Num Clusters: {}\n'.format(num_clusters))
        return Status.Work_done
    except Exception:
        sys.stdout.write('Failed for {}\n'.format(num_clusters))
        exc = traceback.format_exc()
        sys.stdout.write(exc)
        return Status.Work_failure


if __name__ == '__main__':

    cluster = LocalCluster(n_workers=1, threads_per_worker=32)
    client = Client(cluster)
    do_work(40)
