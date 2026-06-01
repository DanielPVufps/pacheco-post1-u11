#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err__));              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

__global__ void vectorAddKernel(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

static void vectorAddCPU(const float *A, const float *B, float *C, int n)
{
    for (int i = 0; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }
}

static void fillVectors(float *A, float *B, int n)
{
    for (int i = 0; i < n; i++)
    {
        A[i] = 0.5f * (float)i;
        B[i] = 1.5f * (float)i;
    }
}

static int runOneSize(int n, float *cpu_ms, float *gpu_kernel_ms, float *gpu_total_ms, int *errors)
{
    size_t bytes = (size_t)n * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);
    float *h_C_gpu = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu)
    {
        fprintf(stderr, "Host malloc failed for N=%d\n", n);
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
        return 0;
    }

    fillVectors(h_A, h_B, n);

    double cpu_t0 = now_ms();
    vectorAddCPU(h_A, h_B, h_C_cpu, n);
    double cpu_t1 = now_ms();
    *cpu_ms = (float)(cpu_t1 - cpu_t0);

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    double gpu_total_t0 = now_ms();

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    CHECK_CUDA(cudaEventRecord(start));
    vectorAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(gpu_kernel_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    double gpu_total_t1 = now_ms();
    *gpu_total_ms = (float)(gpu_total_t1 - gpu_total_t0);

    int err = 0;
    for (int i = 0; i < n; i++)
    {
        if (fabsf(h_C_cpu[i] - h_C_gpu[i]) > 1e-4f)
        {
            err++;
        }
    }
    *errors = err;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 1;
}

int main(void)
{
    const int sizes[] = {1 << 20, 1 << 22, 1 << 24}; // 1M, 4M, 16M
    const char *labels[] = {"1M", "4M", "16M"};

    printf("========================================\n");
    printf(" CUDA BENCHMARK - VECTOR ADD\n");
    printf("========================================\n\n");

    printf("%-6s %-12s %-14s %-14s %-8s\n",
           "N", "CPU(ms)", "GPU kernel(ms)", "GPU total(ms)", "Errors");

    for (int i = 0; i < 3; i++)
    {
        float cpu_ms = 0.0f;
        float gpu_kernel_ms = 0.0f;
        float gpu_total_ms = 0.0f;
        int errors = -1;

        if (!runOneSize(sizes[i], &cpu_ms, &gpu_kernel_ms, &gpu_total_ms, &errors))
        {
            return EXIT_FAILURE;
        }

        printf("%-6s %-12.3f %-14.3f %-14.3f %-8d\n",
               labels[i], cpu_ms, gpu_kernel_ms, gpu_total_ms, errors);

        printf("Errores: %d\n\n", errors);
    }

    return EXIT_SUCCESS;
}
