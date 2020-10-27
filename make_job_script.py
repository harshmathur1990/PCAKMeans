if __name__ == '__main__':
    formatted = 'mpirun -np 32 /home/harsh/parallel-kmeans-int64/mpi_main -t 0.0001 -i /home/harsh/zscaled_nc3.nc -v zscaled -n {} -o /home/harsh/kmeans_output/out_{}.nc 2>&1 | tee /home/harsh/kmeans_output/output_{}.txt'

    lines = list()
    for i in range(1, 101):
        lines.append(
            formatted.format(i, i, i)
        )

    f = open('run.sh', 'w')
    f.write('\n'.join(lines))
    f.close()
