#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

#define N1 1024
#define N2 2048
#define N3 4096
#define FactorIntToDouble 1.1

double firstMatrix[N3][N3] = {0.0};
double secondMatrix[N3][N3] = {0.0};
double matrixMultiResult[N3][N3] = {0.0};

void matrixInit(int n)
{
    #pragma omp parallel for
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            srand(row + col);
            firstMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
            secondMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
        }
    }
}

void smallMatrixMult(int upperOfRow, int bottomOfRow, int leftOfCol, int rightOfCol, int transLeft, int transRight, int n)
{
    for (int i = upperOfRow; i <= bottomOfRow; i++) {
        for (int j = leftOfCol; j <= rightOfCol; j++) {
            matrixMultiResult[i][j] = 0.0;
            for (int k = transLeft; k <= transRight; k++) {
                matrixMultiResult[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
            }
        }
    }
}

void matrixMulti(int upperOfRow, int bottomOfRow, int leftOfCol, int rightOfCol, int transLeft, int transRight, int n)
{
    if ((bottomOfRow - upperOfRow) < 512) {
        smallMatrixMult(upperOfRow, bottomOfRow, leftOfCol, rightOfCol, transLeft, transRight, n);
    } else {
        #pragma omp task
        {
            matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, leftOfCol, (leftOfCol + rightOfCol) / 2, transLeft, (transLeft + transRight) / 2, n);
            matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, leftOfCol, (leftOfCol + rightOfCol) / 2, (transLeft + transRight) / 2 + 1, transRight, n);
        }

        #pragma omp task
        {
            matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, transLeft, (transLeft + transRight) / 2, n);
            matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, (transLeft + transRight) / 2 + 1, transRight, n);
        }

        #pragma omp task
        {
            matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, leftOfCol, (leftOfCol + rightOfCol) / 2, transLeft, (transLeft + transRight) / 2, n);
            matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, leftOfCol, (leftOfCol + rightOfCol) / 2, (transLeft + transRight) / 2 + 1, transRight, n);
        }

        #pragma omp task
        {
            matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, transLeft, (transLeft + transRight) / 2, n);
            matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, (transLeft + transRight) / 2 + 1, transRight, n);
        }

        #pragma omp taskwait
    }
}

int main()
{
    // Case N = 1024
    matrixInit(N1);

    // Sequential execution
    clock_t t1 = clock();
    matrixMulti(0, N1 - 1, 0, N1 - 1, 0, N1 - 1, N1);
    clock_t t2 = clock();
    printf("N = 1024, Sequential Time: %.6f seconds\n", (double)(t2 - t1) / CLOCKS_PER_SEC);

    // Parallel execution
    matrixInit(N1);
    double t1_omp = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single nowait
        matrixMulti(0, N1 - 1, 0, N1 - 1, 0, N1 - 1, N1);
    }
    double t2_omp = omp_get_wtime();
    printf("N = 1024, Parallel Time: %.6f seconds\n", t2_omp - t1_omp);

    // Case N = 2048
    matrixInit(N2);

    // Sequential execution
    t1 = clock();
    matrixMulti(0, N2 - 1, 0, N2 - 1, 0, N2 - 1, N2);
    t2 = clock();
    printf("N = 2048, Sequential Time: %.6f seconds\n", (double)(t2 - t1) / CLOCKS_PER_SEC);

    // Parallel execution
    matrixInit(N2);
    t1_omp = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single nowait
        matrixMulti(0, N2 - 1, 0, N2 - 1, 0, N2 - 1, N2);
    }
    t2_omp = omp_get_wtime();
    printf("N = 2048, Parallel Time: %.6f seconds\n", t2_omp - t1_omp);

    // Case N = 4096
    matrixInit(N3);

    // Sequential execution
    t1 = clock();
    matrixMulti(0, N3 - 1, 0, N3 - 1, 0, N3 - 1, N3);
    t2 = clock();
    printf("N = 4096, Sequential Time: %.6f seconds\n", (double)(t2 - t1) / CLOCKS_PER_SEC);

    // Parallel execution
    matrixInit(N3);
    t1_omp = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single nowait
        matrixMulti(0, N3 - 1, 0, N3 - 1, 0, N3 - 1, N3);
    }
    t2_omp = omp_get_wtime();
    printf("N = 4096, Parallel Time: %.6f seconds\n", t2_omp - t1_omp);

    return 0;
}
