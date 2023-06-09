# CUDA gpu 编程学习，基于 《CUDA 编程——基础与实践》（樊哲勇）。

包含章节：

1. [GPU 硬件与 CUDA 程序开发工具](./capter1/ReadMe.md)
2. [CUDA 中的线程组织](./capter2/ReadMe.md)
3. [简单 CUDA 程序的基本框架](./capter3/ReadMe.md)
4. [CUDA 程序的错误检测](./capter4/ReadMe.md)
5. [GPU 加速的关键](./capter5/ReadMe.md)
6. [CUDA 内存组织](./capter6/ReadMe.md)
7. [全局内存的合理使用](./capter7/ReadMe.md)
8. [共享内存的合理使用](./capter8/ReadMe.md)
9. [原子函数的合理使用](./capter9/ReadMe.md)
10. [线程束基本函数与协作组](./capter10/ReadMe.md)
11. [CUDA 流](./capter11/ReadMe.md)
12. [使用同一内存编程]()
13. [分子动力学模型](./capter13/ReadMe.md)
14. [CUDA 标准库](./capter14/ReadMe.md)


## CUDA 官方文档

[CUDA c++编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
[CUDA c++最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)  
[CUDA 运行时API手册](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)  
[CUDA 数学函数库API手册](https://docs.nvidia.com/cuda/cuda-math-api/index.html)  


## CUDA 编程案例

[CUDA Samples](https://github.com/NVIDIA/cuda-samples)
+ Simple Reference
基础CUDA示例，适用于初学者， 反映了运用CUDA和CUDA runtime APIs的一些基本概念.
+ Utilities Reference
演示如何查询设备能力和衡量GPU/CPU 带宽的实例程序。
+ Graphics Reference
图形化示例展现的是 CUDA, OpenGL, DirectX 之间的互通性。
+ Imaging Reference
图像处理，压缩，和数据分析。
+ Finance Reference
金融计算的并行处理。
+ Simulations Reference
展现一些运用CUDA的模拟算法。
+ Advanced Reference
用CUDA实现的一些先进的算法。
+ Cudalibraries Reference
这类示例主要告诉我们该如何使用CUDA各种函数库(NPP, CUBLAS, CUFFT,CUSPARSE, and CURAND)。