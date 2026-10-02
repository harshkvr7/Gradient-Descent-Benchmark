#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

void computeGradientsCPU(const std::vector<float> &x, const std::vector<float> &y,
                         float w, float b,
                         float &grad_w, float &grad_b)
{
    int N = x.size();
    float sum_dw = 0.0f;
    float sum_db = 0.0f;

    for (int i = 0; i < N; ++i)
    {
        float y_pred = w * x[i] + b;
        float error = y_pred - y[i];

        sum_dw += (2.0f / N) * error * x[i];
        sum_db += (2.0f / N) * error;
    }

    grad_w = sum_dw;
    grad_b = sum_db;
}

__global__ void computeGradientsKernel(const float *x, const float *y,
                                       float w, float b,
                                       float *grad_w, float *grad_b,
                                       int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float x_val = x[idx];
        float y_true = y[idx];
        float y_pred = w * x_val + b;

        float error = y_pred - y_true;

        float dw = (2.0f / N) * error * x_val;
        float db = (2.0f / N) * error;

        atomicAdd(grad_w, dw);
        atomicAdd(grad_b, db);
    }
}

float computeLoss(const std::vector<float> &x, const std::vector<float> &y, float w, float b)
{
    float total_loss = 0.0f;
    int N = x.size();
    for (int i = 0; i < N; ++i)
    {
        float pred = w * x[i] + b;
        float diff = pred - y[i];
        total_loss += diff * diff;
    }
    return total_loss / N;
}

int main()
{
    // y = 3.5 * x + 2.0
    const int N = 1000000;
    const float true_w = 3.5f;
    const float true_b = 2.0f;

    std::vector<float> h_x(N);
    std::vector<float> h_y(N);

    for (int i = 0; i < N; ++i)
    {
        h_x[i] = static_cast<float>(i) / N * 10.0f;
        h_y[i] = true_w * h_x[i] + true_b;
    }

    float lr = 0.01f;
    int epochs = 1000;

    float cpu_w = 0.0f;
    float cpu_b = 0.0f;

    std::cout << "Starting CPU Gradient Descent (" << epochs << " epochs)..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch <= epochs; ++epoch)
    {
        float grad_w = 0.0f;
        float grad_b = 0.0f;

        computeGradientsCPU(h_x, h_y, cpu_w, cpu_b, grad_w, grad_b);

        cpu_w -= lr * grad_w;
        cpu_b -= lr * grad_b;
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU Training Complete." << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    float *d_x, *d_y, *d_grad_w, *d_grad_b;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_grad_w, sizeof(float));
    cudaMalloc(&d_grad_b, sizeof(float));

    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    float gpu_w = 0.0f;
    float gpu_b = 0.0f;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // CUDA event GPU timing
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    std::cout << "Starting GPU Gradient Descent (" << epochs << " epochs)..." << std::endl;
    cudaEventRecord(start_evt);

    for (int epoch = 0; epoch <= epochs; ++epoch)
    {
        cudaMemset(d_grad_w, 0, sizeof(float));
        cudaMemset(d_grad_b, 0, sizeof(float));

        computeGradientsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, gpu_w, gpu_b, d_grad_w, d_grad_b, N);
        cudaGetLastError();
        cudaDeviceSynchronize();

        float grad_w = 0.0f;
        float grad_b = 0.0f;
        cudaMemcpy(&grad_w, d_grad_w, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost);

        gpu_w -= lr * grad_w;
        gpu_b -= lr * grad_b;
    }

    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start_evt, stop_evt);
    std::cout << "GPU Training Complete." << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    std::cout << "=== PERFORMANCE & RESULTS COMPARISON ===" << std::endl;
    std::cout << "Dataset Size      : " << N << " samples" << std::endl;
    std::cout << "Epochs            : " << epochs << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "CPU Elapsed Time  : " << cpu_time_ms << " ms" << std::endl;
    std::cout << "GPU Elapsed Time  : " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup (CPU/GPU) : " << (cpu_time_ms / gpu_time_ms) << "x" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Target Parameters : w = " << true_w << ", b = " << true_b << std::endl;
    std::cout << "CPU Trained Result: w = " << cpu_w << ", b = " << cpu_b
              << " | Final Loss: " << computeLoss(h_x, h_y, cpu_w, cpu_b) << std::endl;
    std::cout << "GPU Trained Result: w = " << gpu_w << ", b = " << gpu_b
              << " | Final Loss: " << computeLoss(h_x, h_y, gpu_w, gpu_b) << std::endl;
    std::cout << "=============================================" << std::endl;

    // Cleanup
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_grad_w);
    cudaFree(d_grad_b);

    return 0;
}