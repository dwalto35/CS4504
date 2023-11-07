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

void matrixMulti(int n)
{
    #pragma omp parallel for
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            double resultValue = 0;

            for (int transNumber = 0; transNumber < n; transNumber++)
            {
                resultValue += firstMatrix[row][transNumber] * secondMatrix[transNumber][col];
            }
            matrixMultiResult[row][col] = resultValue;
        }
    }
}

void matrixInit(int n)
{
    #pragma omp parallel for
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            srand(row + col);
            firstMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
            secondMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
        }
    }
}

int main()
{
    clock_t t1, t2;
    double t1_omp, t2_omp;
    
    // Case N = 1024
    matrixInit(N1);
    
    // Serial execution
    t1 = clock();
    matrixMulti(N1);
    t2 = clock();
    printf("N = 1024, Sequential Time: %.6f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
    
    // Parallel execution
    matrixInit(N1);
    t1_omp = omp_get_wtime();
    matrixMulti(N1);
    t2_omp = omp_get_wtime();
    printf("N = 1024, Parallel Time: %.6f\n", t2_omp - t1_omp);
    
    // Case N = 2048
    matrixInit(N2);
    
    // Serial execution
    t1 = clock();
    matrixMulti(N2);
    t2 = clock();
    printf("N = 2048, Sequential Time: %.6f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
    
    // Parallel execution
    matrixInit(N2);
    t1_omp = omp_get_wtime();
    matrixMulti(N2);
    t2_omp = omp_get_wtime();
    printf("N = 2048, Parallel Time: %.6f\n", t2_omp - t1_omp);
    
    // Case N = 4096
    matrixInit(N3);
    
    // Serial execution
    t1 = clock();
    matrixMulti(N3);
    t2 = clock();
    printf("N = 4096, Sequential Time: %.6f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
    
    // Parallel execution
    matrixInit(N3);
    t1_omp = omp_get_wtime();
    matrixMulti(N3);
    t2_omp = omp_get_wtime();
    printf("N = 4096, Parallel Time: %.6f\n", t2_omp - t1_omp);

    return 0;
}
