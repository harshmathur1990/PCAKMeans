import sys
import enum
import traceback
import numpy as np
import os
import os.path
import sunpy.io
import h5py
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpi4py import MPI
from delay_retry import retry
from pathlib import Path


base_input_path = Path(
    '/home/harsh/OsloAnalysis/new_kmeans/inversions/'
)

second_path = base_input_path / 'plots_quiet_v1'
profile_files = [
    second_path / 'quiet_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_1_vt_4_profs.nc',
]

atmos_files = [
    second_path / 'quiet_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_1_vt_4_atmos.nc',
]
observed_file = Path(
    base_input_path / 'quiet_frame_0_21_x_662_712_y_708_758.nc'
)


profiles = [h5py.File(filename, 'r') for filename in profile_files]
atmos = [h5py.File(filename, 'r') for filename in atmos_files]
observed = h5py.File(observed_file, 'r')

indices = np.where(observed['profiles'][0, 0, 0, :-1, 0] != 0)[0]

write_path = second_path / 'profile_fits'

red = '#f6416c'
brown = '#ffde7d'
green = '#00b8a9'

size = plt.rcParams['lines.markersize']

fontP = FontProperties()
fontP.set_size('xx-small')

calib_velocity = 333390.00079943583

tuple_map = dict()

indfile = h5py.File(base_input_path / 'pixel_indices.h5', 'r')

a, b, c = indfile['pixel_indices'][()]

index = 0
for k, i, j in zip(a, b, c):
    tuple_map[index] = tuple([k, i, j])
    index += 1

total_elements = index


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


class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


@timeit
def do_work(index):

    sys.stdout.write('Processing for k: {}, i:{}, j:{}\n'.format(k, i, j))

    try:
        #do something
        plt.close('all')
        plt.clf()
        plt.cla()
        fig, axs = plt.subplots(2, 2)
        obp,  = axs[0][0].plot(
            observed['wav'][indices],
            observed['profiles'][0, 0, index, :, 0][indices],
            color=red,
            linewidth=0.5,
            label='Shock'
        )

        obs = axs[0][0].scatter(
            observed['wav'][indices],
            observed['profiles'][0, 0, index, :, 0][indices],
            color=red,
            s=size / 4
            # linewidth=0.5
            # label='Shock'
        )

        # plotting the inverted profile
        inp, = axs[0][0].plot(
            observed['wav'][:-1],
            profiles[0]['profiles'][0, 0, index, :-1, 0],
            color=green,
            linewidth=0.5,
            label='Fit'
        )

        ins = axs[0][0].scatter(
            observed['wav'][:-1],
            profiles[0]['profiles'][0, 0, index, :-1, 0],
            color=green,
            # linewidth=0.5,
            s=size / 4
            # label='Fit'
        )

        axs[0][0].set_ylim(0, 0.5)

        # plot inverted temperature profile
        temp, = axs[0][1].plot(
            atmos[0]['ltau500'][0][0][index],
            atmos[0]['temp'][0][0][index],
            color=green,
            linewidth=0.5
        )

        axs[0][1].set_ylim(4000, 11000)

        # plot inverted Vlos profile
        vlos, = axs[1][0].plot(
            atmos[0]['ltau500'][0][0][index],
            (atmos[0]['vlos'][0][0][index] - calib_velocity)/ 1e5,
            color=green,
            linewidth=0.5
        )

        axs[1][0].set_ylim(-20, 20)

        # plot inverted Vturb profile
        vturb, = axs[1][1].plot(
            atmos[0]['ltau500'][0][0][index],
            atmos[0]['vturb'][0][0][index] / 1e5,
            color=green,
            linewidth=0.5
        )

        axs[1][1].set_ylim(-10, 10)

        fig.tight_layout()

        axs[0][0].legend(loc='upper right', prop=fontP)

        plt.savefig(write_path / 'plot_{}_{}_{}.png'.format(*tuple_map[index]), format='png', dpi=200)
        return Status.Work_done
    except Exception:
        sys.stdout.write('Failed for index: {}, k: {}, i:{}, j:{}\n'.format(index, *tuple_map[index]))
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

        sys.stdout.write('Total Elements: {}\n'.format(total_elements))
        for m in range(total_elements):
            waiting_queue.add(m)

        filepath = write_path / 'status_job.h5'
        if os.path.exists(str(filepath)):
            mode = 'r+'
        else:
            mode = 'w'

        f = h5py.File(filepath, mode)

        if 'finished' in list(f.keys()):
            finished = list(f['finished'][()])
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
            sys.stdout.write('Item: {}, worker: {}\n'.format(item, worker))
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

            sys.stdout.write('Rank: {}, Received: {}\n'.format(rank, item))

            status = do_work(item)

            comm.send({'status': status, 'item': item}, dest=0, tag=2)
