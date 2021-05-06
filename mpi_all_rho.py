import sys
import enum
import traceback
import numpy as np
import os
import os.path
import sunpy.io
import h5py
# from dask.distributed import LocalCluster, Client
from mpi4py import MPI
from sklearn.cluster import KMeans
from delay_retry import retry
from pathlib import Path
from witt import witt

output_atmos_quiet_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/quiet_profiles_frame_56_77_x_770_820_y_338_388_cycle_1_t_5_vl_1_vt_4_atmos.nc')

output_atmos_shock_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/shock_spicule_profiles_frame_56_77_x_770_820_y_338_388_cycle_1_t_4_vl_5_vt_4_atmos.nc')

output_atmos_shock_78_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/shock_78_frame_56_77_x_770_820_y_338_388_cycle_1_t_4_vl_5_vt_4_atmos.nc')

output_atmos_reverse_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/reverse_shock_profiles_frame_56_77_x_770_820_y_338_388_cycle_1_t_4_vl_5_vt_4_atmos.nc')

output_atmos_retry_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/retry_shock_spicule_frame_56_77_x_770_820_y_338_388_cycle_1_t_4_vl_5_vt_4_atmos.nc')

output_atmos_other_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/other_emission_profiles_frame_56_77_x_770_820_y_338_388_cycle_1_t_5_vl_5_vt_4_atmos.nc')


kmeans_output_dir = '/data/harsh/kmeans_output'
input_file = '/data/harsh/nb_3950_2019-06-06T10:26:20_scans=0-99_corrected_im.fits'
selected_frames = np.array([0, 11, 25, 36, 60, 78, 87])
data, header = sunpy.io.fits.read(input_file, memmap=True)[0]
framerows = np.transpose(
    data[selected_frames][:, 0],
    axes=(0, 2, 3, 1)
).reshape(
    7 * 1236 * 1848,
    30
)
mn = np.mean(framerows, axis=0)
sd = np.std(framerows, axis=0)
weights = np.ones(30) * 0.025
weights[10:20] = 0.05


class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


def do_work(num_clusters):
    global framerows
    global mn
    global sd
    global weights

    sys.stdout.write('Processing for Num Clusters: {}\n'.format(num_clusters))

    try:

        sys.stdout.write('Process: {} Read from File\n'.format(num_clusters))

        framerows = (framerows - mn) / sd
        framerows *= weights

        model = KMeans(
            n_clusters=num_clusters,
            max_iter=10000,
            tol=1e-6
        )

        sys.stdout.write('Process: {} Before KMeans\n'.format(num_clusters))

        model.fit(framerows)

        sys.stdout.write('Process: {} Fitted KMeans\n'.format(num_clusters))

        fout = h5py.File(
            '{}/out_{}.h5'.format(kmeans_output_dir, num_clusters), 'w'
        )
        sys.stdout.write(
            'Process: {} Open file for writing\n'.format(num_clusters)
        )
        fout['cluster_centers_'] = model.cluster_centers_
        fout['labels_'] = model.labels_
        fout['inertia_'] = model.inertia_
        fout['n_iter_'] = model.n_iter_

        rps = np.zeros_like(model.cluster_centers_)

        framerows /= weights
        framerows = (framerows * sd) - mn

        for i in range(num_clusters):
            a = np.where(model.labels_ == i)
            rps[i] = np.mean(framerows[a], axis=0)

        fout['rps'] = rps

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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        fout_atmos_quiet = h5py.File(output_atmos_quiet_filepath, 'r')

        fout_atmos_shock = h5py.File(output_atmos_shock_filepath, 'r')

        fout_atmos_shock_78 = h5py.File(output_atmos_shock_78_filepath, 'r')

        fout_atmos_reverse = h5py.File(output_atmos_reverse_filepath, 'r')

        fout_atmos_retry = h5py.File(
            output_atmos_retry_filepath,
            'r'
        )

        fout_atmos_other = h5py.File(
            output_atmos_other_filepath,
            'r'
        )

        all_temp = np.zeros(
            (
                frames[1] - frames[0],
                x[1] - x[0],
                y[1] - y[0],
                150
            )
        )

        all_pgas = np.zeros(
            (
                frames[1] - frames[0],
                x[1] - x[0],
                y[1] - y[0],
                150
            )
        )

        all_temp[a1, b1, c1] = fout_atmos_quiet['temp'][0, 0]
        all_temp[a2, b2, c2] = fout_atmos_shock['temp'][0, 0]
        all_temp[a3, b3, c3] = fout_atmos_shock_78['temp'][0, 0]
        all_temp[a4, b4, c4] = fout_atmos_reverse['temp'][0, 0]
        all_temp[a5, b5, c5] = fout_atmos_retry['temp'][0, 0]
        all_temp[a6, b6, c6] = fout_atmos_other['temp'][0, 0]

        all_pgas[a1, b1, c1] = fout_atmos_quiet['pgas'][0, 0]
        all_pgas[a2, b2, c2] = fout_atmos_shock['pgas'][0, 0]
        all_pgas[a3, b3, c3] = fout_atmos_shock_78['pgas'][0, 0]
        all_pgas[a4, b4, c4] = fout_atmos_reverse['pgas'][0, 0]
        all_pgas[a5, b5, c5] = fout_atmos_retry['pgas'][0, 0]
        all_pgas[a6, b6, c6] = fout_atmos_other['pgas'][0, 0]

        status = MPI.Status()
        waiting_queue = set()
        running_queue = set()
        finished_queue = set()
        failure_queue = set()

        for zz in range(21):
            for xx in range(50):
                for yy in range(50):
                    waiting_queue.add((zz, xx, yy, all_temp[zz, xx, yy], all_pgas[zz, xx, yy]))

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
                if 'finished' in list(f.keys()):
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
                running_queue.add(new_item)

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
