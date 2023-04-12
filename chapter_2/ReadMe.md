# CUDA 中的线程组织

CUDA 虽然支持 C++ 但支持得并不充分，导致 C++ 代码中有很多 C 代码的风格。

CUDA 采用 **nvcc** 作为编译器，支持 C++ 代码；nvcc 在编译 CUDA 程序时，   
会将纯粹的 c++ 代码交给 c++ 编译器，自己负责编译剩下的 cu 代码。

------

## C++ 语言中的 Hello World 程序

```bash
vim hello.cpp
g++ hello.cpp -o hello
 ./hello
```

-------

## CUDA 中的 Hello World 程序

### 使用 核函数 的 CUDA 程序

一个利用了 GPU 的 CUDA 程序既有主机代码，又有设备代码（在设备中执行的代码）。  
主机对设备的调用是通过 **核函数（kernel function）** 实现的。

```cpp
int main()
{
    主机代码
    核函数的调用
    主机代码

    return 0；
}
```

CUDA 中的核函数与 C++ 中的函数是类似的，但一个显著的差别是：它必须被限定词（qualifier）**__global__** 修饰。其中 global 前后是双下划线。另外，核函数的返回类型必须是空类型，即 **void**。

```cpp
__global__ void hell_from__gpu()
{
    // 核函数不支持 c++ 的 iostream。
    // 限定符 __global__ 和 void 的次序可随意。
    printf("Hello World from the GPU!\n");
}
```

调用核函数的方式：

```cpp
hello_from_gpu<<<1, 1>>>
```

主机在调用一个核函数时，必须指明在设备中指派多少线程。核函数中的线程常组织为若干线程块： 
1. 三括号中第一个数字是线程块的个数（number of thread block）；
2. 三括号中第二个数字是每个线程块中的线程数（number of thread in per block）。

一个核函数的全部线程块构成一个`网格（grid）`，线程块的个数称为`网格大小（grid size）`。  
每个线程块中含有相同数目的线程，该数目称为`线程块大小（block size）`。

所以，核函数的总的线程数即 `总的线程数=网格大小*线程块大小`:

    hello_from_gpu<<<grid size, block size>>>

调用核函数后，调用 CUDA 运行时 API 函数，同步主机和设备：

    cudaDeviceSynchronize();

核函数中调用输出函数，输出流是先存放在缓冲区的，而缓冲区不会自动刷新。只有程序遇到某种同步操作时缓冲区才会刷新。函数 **cudaDeviceSynchronize** 的作用是同步主机与设备，所以能够促使缓冲区刷新。

------

## CUDA 中的线程组织

### 使用多个线程的核函数

实际上，总的线程数大于计算核心数时才能更充分地利用 GPU 中的计算资源。

核函数中代码的执行方式**单指令-多线程**，即每一个线程都执行同一串指令。

### 使用线程索引

    <<<grid_size, block_size>>>

这里的 grid_size（网格大小）和 block_size（线程块大小）一般来说是一个结构体类型的变量，但也可以是一个普通的整型变量。从开普勒架构开始，最大允许的线程块大小是 1024，而最大允许的网格大小是 2^31 − 1。需要指出的是，一个核函数中虽然可以指派如此巨大数目的线程数，但在执行时能够同时活跃（不活跃的线程处于等待状态）的线程数是由硬件（主要是 CUDA 核心数）和软件（即核函数中的代码）决定的。

每个线程在核函数中都有一个唯一的身份标识。

> gridDim.x：该变量的数值等于执行配置中变量 grid_size 的数值。
> blockDim.x：该变量的数值等于执行配置中变量 block_size 的数值。
> blockIdx.x：该变量指定一个线程在一个网格中的线程块指标，其取值范围是从 0 到 gridDim.x - 1。
> threadIdx.x：该变量指定一个线程在一个线程块中的线程指标，其取值范围是从 0 到 blockDim.x - 1 。

每个线程块的计算是相互独立的。

### 推广至多维网格

- blockIdx 和 threadIdx 是类型为 uint3 的变量。该类型是一个结构体，具有 x、y、z 这 3 个成员。结构体 uint3 在头文件 vector_types.h 中定义：

```cpp
struct __device_builtin__ uint3
{
unsigned int x, y, z;
};
typedef __device_builtin__ struct uint3 uint3;
```

-  gridDim 和 blockDim 是类型为 dim3 的变量。该类型是一个结构体，具有 x、y、z 这 3 个成员。结构体 dim3 也在头文件 vector_types.h 定义。

这些内建变量都只在核函数中有效（可见）：

> blockIdx.x 的取值范围是从 0 到 gridDim.x - 1。
> blockIdx.y 的取值范围是从 0 到 gridDim.y - 1。
> blockIdx.z 的取值范围是从 0 到 gridDim.z - 1。
> threadIdx.x 的取值范围是从 0 到 blockDim.x - 1 。
> threadIdx.y 的取值范围是从 0 到 blockDim.y - 1 。
> threadIdx.z 的取值范围是从 0 到 blockDim.z - 1 。

可以用结构体 dim3 定义“多维”的网格和线程块:

    dim3 grid_size(Gx, Gy, Gz);
    dim3 block_size(Bx, By, Bz);

多维的网格和线程块本质上还是一维的。多维网格线程在线程块上的 ID；
    
    tid = threadIdx.z * (blockDim.x * blockDim.y)  // 当前线程块上前面的所有线程数
        + threadIdx.y * (blockDim.x)               // 当前线程块上当前面上前面行的所有线程数
        + threadIdx.x                              // 当前线程块上当前面上当前行的线程数

多维网格线程块在网格上的 ID:

    bid = blockIdx.z * (gridDim.x * gridDim.y)
        + blockIdx.y * (gridDim.x)
        + blockIdx.x

x 维度是最内层的（变化最快），而 z 维度是最外层的（变化最慢）。

<figure>
  <img src="/images/CUDA 核函数中的线程组织示意图.png" alt="CUDA 核函数中的线程组织示意图" />
  <figcaption>CUDA 核函数中的线程组织示意图</figcaption>
</figure>

一个线程块中的线程还可以细分为不同的**线程束（thread warp）**。一个线程束（即一束线程）是同一个线程块中相邻的 warpSize 个线程。warpSize 也是一个内建变量，表示线程束大小，其值对于目前所有的 GPU 架构都是 32。

### 2.3.4 网格与线程块大小的限制

对于从开普勒架构到图灵架构的 GPU，网格大小在 x, y, z 方向的最大允许值为 （2^31 - 1, 2^16 - 1, 2^16 -1）；  
线程块大小在 x, y, z 方向的最大允许值为 （1024， 1024， 64），同时要求一个线程块最多有 1024 个线程。

## CUDA 中的头文件

CUDA 头文件的后缀一般是 “.cuh”；同时，同时可以包含c/cpp 的头文件 “.h”、“.hpp”，采用 nvcc 编译器会自动包含必要的 cuda 头文件，  如 <cuda.h>, <cuda_runtime.h>，同时前者也包含了c++头文件 <stdlib.h>。

## 用 nvcc 编译 CUDA 程序

CUDA 的编译器驱动（compiler driver）nvcc 先将全部源代码分离为主机代码和设备代码。主机代码完整地支持 C++ 语法，但设备代码只部分地支持 C++。

nvcc 先将设备代码编译为 PTX（Parallel Thread eXecution）伪汇编代码，再将 PTX 代码编译为二进制的 cubin 目标代码。

在将源代码编译为 PTX 代码时，需要用选项 `-arch=compute_XY` 指定一个虚拟架构的计算能力，用以确定代码中能够使用的 CUDA 功能。

在将 PTX 代码编译为 cubin 代码时，需要用选项 `-code=sm_ZW` 指定一个真实架构的计算能力，用以确定可执行文件能够使用的 GPU。

真实架构的计算能力必须等于或者大于虚拟架构的计算能力。

如果希望编译出来的可执行文件能够在更多的 GPU 中执行，可以同时指定多组计算能力:

```bash
-gencode arch=compute_35,code=sm_35
-gencode arch=compute_50,code=sm_50
-gencode arch=compute_60,code=sm_60
-gencode arch=compute_70,code=sm_70
```

同时，nvcc 有一种称为 **实时编译（just-in-time compilation）**机制，可以在运行可执行文件时从其中保留的PTX  代码中临时编译出一个 cubin 目标代码。因此， 需要通过选项 `-gencode arch=compute_XY, code=compute_XY`，指定所保留 PTX 代码的虚拟架构。

nvcc 编译有一个简化的编译选项 `-arch=sim_XY`，其等价于： 

```bash
-gencode arch=compute_XY, code=sm_XY  
    -gencode arch=compute_XY, code=compute_XY
```


