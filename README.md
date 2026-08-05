# CUDA vs. CPU Gradient Descent for Linear Regression

![CUDA](https://img.shields.io/badge/CUDA-Runtime%20API-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Performance](https://img.shields.io/badge/Speedup-1.80x%20GPU-success?style=for-the-badge)

A high-performance C++ and CUDA benchmark project demonstrating **Linear Regression training via Batch Gradient Descent** across **1,000,000 samples**. This repository implements and benchmarks identical numerical optimization algorithms on both the CPU and an NVIDIA GPU to analyze parallel computing performance, accuracy convergence, and hardware utilization.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Architecture & Implementation Details](#-architecture--implementation-details)
  - [CPU Baseline Implementation](#1-cpu-baseline-implementation)
  - [GPU CUDA Kernel Implementation](#2-gpu-cuda-kernel-implementation)
- [Benchmark Results](#-benchmark-results)
  - [Raw Terminal Output](#raw-terminal-output)
  - [Performance & Accuracy Comparison](#performance--accuracy-comparison)
- [Prerequisites & Build Instructions](#-prerequisites--build-instructions)
- [Usage](#-usage)

---

## 📖 Overview

Training machine learning models over large datasets requires evaluating error gradients repeatedly across many iterations (epochs). In a 1D Linear Regression task ($y = w x + b$), computing batch gradients over $N = 1,000,000$ data points entails calculating independent sample errors and aggregating their gradient contributions.

This project contrasts:
1. **Sequential CPU Execution:** Iterates linearly over all $N$ samples in host memory for every epoch.
2. **Parallel GPU Execution:** Distributes sample evaluations across CUDA threads ($256$ threads/block) and sums gradient contributions using GPU atomic operations (`atomicAdd`).

---

## ⚙️ Architecture & Implementation Details

### 1. CPU Baseline Implementation
The CPU implementation (`computeGradientsCPU`) evaluates gradients sequentially over an O($N$) loop:
```cpp
void computeGradientsCPU(const std::vector<float>& x, const std::vector<float>& y,
                         float w, float b, float& grad_w, float& grad_b) {
    int N = x.size();
    float sum_dw = 0.0f;
    float sum_db = 0.0f;

    for (int i = 0; i < N; ++i) {
        float y_pred = w * x[i] + b;
        float error = y_pred - y[i];
        sum_dw += (2.0f / N) * error * x[i];
        sum_db += (2.0f / N) * error;
    }
    grad_w = sum_dw;
    grad_b = sum_db;
}
```

### 2. GPU CUDA Kernel Implementation
The GPU implementation (`computeGradientsKernel`) assigns each data sample $i$ to a dedicated CUDA thread. Each thread computes the local error and accumulates the partial gradients directly into device memory via `atomicAdd`:
```cpp
__global__ void computeGradientsKernel(const float* x, const float* y, 
                                       float w, float b, 
                                       float* grad_w, float* grad_b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
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
```

---

## 📊 Benchmark Results

### Raw Terminal Output

```text
Starting CPU Gradient Descent (1000 epochs)...
CPU Training Complete.
---------------------------------------------
Starting GPU Gradient Descent (1000 epochs)...
GPU Training Complete.
---------------------------------------------
=== PERFORMANCE & RESULTS COMPARISON ===
Dataset Size      : 1000000 samples
Epochs            : 1000
---------------------------------------------
CPU Elapsed Time  : 4414.24 ms
GPU Elapsed Time  : 2454.46 ms
Speedup (CPU/GPU) : 1.79846x
---------------------------------------------
Target Parameters : w = 3.5, b = 2
CPU Trained Result: w = 3.50161, b = 1.98936 | Final Loss: 2.83118e-05
GPU Trained Result: w = 3.50161, b = 1.98935 | Final Loss: 2.83111e-05
=============================================
```

### Performance & Accuracy Comparison

| Metric / Parameter | Target Ground Truth | CPU Execution | GPU Execution (CUDA) | Delta / Note |
| :--- | :---: | :---: | :---: | :--- |
| **Dataset Size ($N$)** | — | `1,000,000` | `1,000,000` | Identical dataset |
| **Epochs** | — | `1,000` | `1,000` | Identical iterations |
| **Total Training Time** | — | **4,414.24 ms** | **2,454.46 ms** | **1,959.78 ms saved** |
| **Speedup (CPU / GPU)** | — | `1.00x` | **1.80x** | **1.79846x speedup** |
| **Weight ($w$)** | `3.50000` | `3.50161` | `3.50161` | Error $ pprox 0.046\%$ |
| **Bias ($b$)** | `2.00000` | `1.98936` | `1.98935` | Error $ pprox 0.53\%$ |
| **Final Loss (MSE)** | `0.00000` | `2.83118e-05` | `2.83111e-05` | Converged to numerical parity |

---

## 🛠 Prerequisites & Build Instructions

### Requirements
- **NVIDIA CUDA Toolkit**: v11.0 or newer
- **C++ Compiler**: GCC, Clang, or MSVC supporting C++17
- **GPU**: NVIDIA GPU with Compute Capability 5.0+ (Maxwell, Pascal, Volta, Turing, Ampere, Ada Lovelace, or Hopper)

### Compiling with `nvcc`
Open a terminal and compile the program with maximum compiler optimizations (`-O3`):

```bash
nvcc -O3 -std=c++17 -o linear_regression_cuda main.cu
```

---

## 🚀 Usage

Execute the compiled binary directly:

```bash
./linear_regression_cuda
```

---