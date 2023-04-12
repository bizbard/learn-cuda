# GPU 硬件与 CUDA 程序开发工具

------

## GPU 硬件简介

GPU 计算不是指单独的 GPU 计算，而是指 CPU + GPU 的异构（heterogeneous）计算。在由 CPU 和 GPU 构成的异构计算平台中，通常将起控制作用的 CPU 称为**主机（host）**，将起加速作用的 GPU 称为**设备（device）**。

表征计算性能的一个重要参数是浮点数运算峰值，即每秒最多能执行的浮点数运算次数，英文为 Floating-point operations per second，缩写为 **FLOPS**。另一个影响计算性能的参数是 GPU 中的**内存带宽（memory bandwidth）**。最后，**显存容量**也是制约应用程序性能的一个因素。

-------

## CUDA 程序开发工具

1. CUDA；
2. OpenCL，更为通用的各种异构平台编写并行程序的框架，AMD 的 GPU 程序开发工具；
3. OpenACC，由多公司共同开发的异构并行编程标准。

CUDA 提供了两层 API（Application Programming Interface，应用程序编程接口）给程序员使用，即 CUDA 驱动（driver）API 和 CUDA 运行时（runtime）API。本书只涉及 CUDA 运行时 API。

<figure>
  <img src="/images/CUDA 编程开发环境概览.png" alt="CUDA 开发环境的主要组件" />
  <figcaption>CUDA 编程开发环境概览</figcaption>
</figure>

------

## CUDA 开发环境搭建

> linux 操作系统：[linux下cuda环境搭建](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

> windows10 操作系统：[windows10下cuda环境搭建](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

------

## nvidia-smi 检查与设置设备

    >> nvidia-smi
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 462.30       Driver Version: 462.30       CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  GeForce MX450      WDDM  | 00000000:2B:00.0 Off |                  N/A |
    | N/A   39C    P8    N/A /  N/A |    119MiB /  2048MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

1. **CUDA Version**， 11.2；
2. **GPU Name**，GeForce MX450，设备号为 0；如果系统中有多个 GPU 且只要使用其中某个特定的 GPU，  
可以通过设置环境变量 **CUDA_VISIBLE_DEVICES** 的值，从而可以在运行 CUDA 程序前选定 GPU;  

```bash
export CUDA_VISIBLE_DEVICES=1
```

这样设置的环境变量在当前 shell session 及其子进程中有效。
3. **TCC/WDDM**，WDDM（windows display driver model），其它包括 TCC（Tesla compute cluster）；  
可以通过命令行 `nvidia-smi -g GPU_ID -dm 0`，设置为 WDDM 模式（1 为 TCC 模式）；

```bash
sudo nvidia-smi -g GPU_ID -dm 0 # 设置为 WDDM 模式
sudo nvidia-smi -g GPU_ID -dm 1 # 设置为 TCC 模式
```

4. **Compute mode**, Default，此时同一个 GPU 中允许存在多个进程；其他模式包括 E.Process，  
指的是独占进程模式，在独占进程模式下，只能运行一个计算进程独占该 GPU，但不适用 WDDM 模式下的 GPU；  
可以通过命令行 `nvidia-smi -i GPU_ID -c 0`，设置为 Default 模式（1 为 E.Process 模式）;

```bash
sudo nvidia-smi -i GPU_ID -c 0 # 默认模式
sudo nvidia-smi -i GPU_ID -c 1 # 独占进程模式
```

5. **Perf**，p8（GPU 性能状态，最大p0~最小p12）；

更多关于 nvidia-smi 的资料：[nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface)

------

## CUDA 的官方手册

> CUDA C++ Programming Guide：(https://docs.nvidia.com/cuda/cuda-c-programming-guide)

> CUDA C++ Best Practices Guide：(https://docs.nvidia.com/cuda/cuda-c-best-practices-guide)

> CUDA运行时 API 的手册: (https://docs.nvidia.com/cuda/cuda-runtime-api)