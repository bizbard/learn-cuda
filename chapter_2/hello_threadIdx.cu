#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tid = threadIdx.x;
    // x 维度是最内层的（变化最快），而 z 维度是最外层的（变化最慢）。
    printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx,
ty);
}

int main(void)
{
    const dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}