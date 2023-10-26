#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#define NUMSTEPS 1000000

int main(int argc, char** argv) {
    int i, rank, size;
    double x, pi, sum = 0.0;
    struct timespec start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    clock_gettime(CLOCK_MONOTONIC, &start);
    double step = 1.0 / (double) NUMSTEPS;
    x = (0.5 + rank) * step;

    for (i = rank; i < NUMSTEPS; i += size) {
        sum += 4.0 / (1.0 + x * x);
        x += step * size; // Skip to the next work item for this process
    }

    double localPi = step * sum;
    MPI_Reduce(&localPi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC, &end);
    u_int64_t diff = 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

    if (rank == 0) {
        printf("PI is %.20f\n", pi);
        printf("Elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
    }

    MPI_Finalize();

    return 0;
}
