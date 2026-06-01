#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE 16

#define CHECK_CUDA(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if(err != cudaSuccess)                                    \
    {                                                         \
        printf("CUDA ERROR: %s\n",                            \
               cudaGetErrorString(err));                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

void matMulCPU(const float *A,
               const float *B,
               float *C,
               int N)
{
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            float sum = 0.0f;

            for(int k=0;k<N;k++)
            {
                sum += A[i*N+k] * B[k*N+j];
            }

            C[i*N+j] = sum;
        }
    }
}

__global__ void matMulNaive(const float *A,
                            const float *B,
                            float *C,
                            int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N)
    {
        float sum = 0.0f;

        for(int k=0;k<N;k++)
        {
            sum += A[row*N+k] * B[k*N+col];
        }

        C[row*N+col] = sum;
    }
}

__global__ void matMulTiled(const float *A,
                            const float *B,
                            float *C,
                            int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for(int t=0; t<(N+TILE-1)/TILE; t++)
    {
        if(row < N && (t*TILE + threadIdx.x) < N)
            sA[threadIdx.y][threadIdx.x] =
                A[row*N + t*TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if(col < N && (t*TILE + threadIdx.y) < N)
            sB[threadIdx.y][threadIdx.x] =
                B[(t*TILE + threadIdx.y)*N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for(int k=0;k<TILE;k++)
        {
            sum += sA[threadIdx.y][k] *
                   sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if(row < N && col < N)
    {
        C[row*N+col] = sum;
    }
}

int verificar(const float *cpu,
              const float *gpu,
              int elementos)
{
    int errores = 0;

    for(int i=0;i<elementos;i++)
    {
        if(fabs(cpu[i] - gpu[i]) > 1e-3f)
        {
            errores++;
        }
    }

    return errores;
}

void benchmark(int N)
{
    printf("\n====================================\n");
    printf("MATRIZ %d x %d\n",N,N);
    printf("====================================\n");

    size_t bytes =
        (size_t)N * N * sizeof(float);

    float *h_A   = (float*)malloc(bytes);
    float *h_B   = (float*)malloc(bytes);
    float *h_CPU = (float*)malloc(bytes);
    float *h_GPU = (float*)malloc(bytes);

    for(int i=0;i<N*N;i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    clock_t cpu_start = clock();

    matMulCPU(h_A,h_B,h_CPU,N);

    clock_t cpu_end = clock();

    double cpu_ms =
        ((double)(cpu_end-cpu_start) /
         CLOCKS_PER_SEC) * 1000.0;

    printf("CPU           : %.2f ms\n",cpu_ms);

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(
        d_A,h_A,bytes,cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_B,h_B,bytes,cudaMemcpyHostToDevice));

    dim3 block(TILE,TILE);

    dim3 grid(
        (N+TILE-1)/TILE,
        (N+TILE-1)/TILE
    );

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float naive_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    matMulNaive<<<grid,block>>>(
        d_A,d_B,d_C,N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(
        &naive_ms,start,stop));

    CHECK_CUDA(cudaMemcpy(
        h_GPU,d_C,bytes,cudaMemcpyDeviceToHost));

    int err_naive =
        verificar(h_CPU,h_GPU,N*N);

    float tiled_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    matMulTiled<<<grid,block>>>(
        d_A,d_B,d_C,N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(
        &tiled_ms,start,stop));

    CHECK_CUDA(cudaMemcpy(
        h_GPU,d_C,bytes,cudaMemcpyDeviceToHost));

    int err_tiled =
        verificar(h_CPU,h_GPU,N*N);

    printf("GPU Naive     : %.2f ms\n",naive_ms);
    printf("GPU Tiled     : %.2f ms\n",tiled_ms);

    if(tiled_ms > 0.0f)
    {
        printf("Speedup       : %.2fx\n",
               naive_ms/tiled_ms);
    }

    printf("Errores Naive : %d\n",
           err_naive);

    printf("Errores Tiled : %d\n",
           err_tiled);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_CPU);
    free(h_GPU);
}

int main()
{
    int deviceCount = 0;

    cudaError_t err =
        cudaGetDeviceCount(&deviceCount);

    if(err != cudaSuccess || deviceCount == 0)
    {
        printf("No CUDA-capable device detected\n");
        return 0;
    }

    benchmark(512);
    benchmark(1024);

    return 0;
}
