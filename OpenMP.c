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

void matrixMulti(int n, clock_t *serialTime, clock_t *parallelTime)
{
    // Serial execution
    clock_t t1, t2;
    
    // Reset the result matrix
    memset(matrixMultiResult, 0, sizeof(matrixMultiResult));
    
    matrixInit(n);
    t1 = clock();
    matrixMultiSerial(n);
    t2 = clock();
    *serialTime = t2 - t1;

    // Parallel execution
    t1 = clock();
    #pragma omp parallel
    {
        #pragma omp single
        matrixMultiParallel(n);
    }
    t2 = clock();
    *parallelTime = t2 - t1;
}

void matrixMultiSerial(int n)
{
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

void matrixMultiParallel(int n)
{
    #pragma omp for
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
    clock_t serialTime, parallelTime;
    
    // Case N = 1024
    matrixMulti(N1, &serialTime, &parallelTime);
    printf("N = 1024, Serial Time: %ld, Parallel Time: %ld\n", serialTime, parallelTime);
    
    // Case N = 2048
    matrixMulti(N2, &serialTime, &parallelTime);
    printf("N = 2048, Serial Time: %ld, Parallel Time: %ld\n", serialTime, parallelTime);

    // Case N = 4096
    matrixMulti(N3, &serialTime, &parallelTime);
    printf("N = 4096, Serial Time: %ld, Parallel Time: %ld\n", serialTime, parallelTime);

    return 0;
}
