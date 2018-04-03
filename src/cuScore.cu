#include <cuda_runtime.h>  
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "cuVocabulary.h"

#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void featureScoreKernel(cuSparseVector* features, size_t rows, size_t cols, cuSparseVector* score_faeture, size_t col)
{
    const uint32 tidx = threadIdx.x;
    const uint32 bidx = blockIdx.x;
    //uint32 max_len = cols > col ? cols: col;
    __shared__ float distance_array[256];
    __shared__ float norm;
    uint32 index_t = tidx;
    uint32 index_b = bidx;
    distance_array[tidx] = 0.0;
    if (bidx < rows)
    {
        while (index_t < cols)
        {
            uint32 pos = index_b * cols + index_t;
            distance_array[tidx] += features[pos].value;
            index_t += blockDim.x;
        }
        __syncthreads();

        uint32 offset = 128;
        while (offset > 0)
        {
            if (tidx < offset)
                distance_array[tidx] += distance_array[tidx + offset];
            offset = offset >> 1;
        }
        __syncthreads();

        if (tidx == 0)
        {
            /// 词袋向量数不多于块数-->暂时把词袋向量数量的控制踢给流
            /// 这么做的理由当然是共享内存的访存效率更高
            norm = distance_array[0];
            printf("%d: %f\n", bidx, norm);
        }
        //index_b += gridDim.x;
    }

}

__device__ uint32 BinarySearch(cuSparseVector* features, uint32 max, uint32 min)
{
    uint32 mid = (max + min) >> 1;
    while (mid <= max)
    {
        
    }
    return 0;
}

void cudaFeatureScore()
{
    struct cuSparseVector* dev_features;
    struct cuSparseVector* dev_key_feature;
    struct cuSparseVector* features = (cuSparseVector*)malloc(sizeof(cuSparseVector) * 20);
    struct cuSparseVector* key_feature = (cuSparseVector*)malloc(sizeof(cuSparseVector) * 6);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 5; j++)
        {
            features[i * 5 + j].id = i * 5 + j;
            features[i * 5 + j].value = 1;
        }
    for (int i = 0 ; i < 6 ; i++)
    {
        key_feature[i].id = i;
        key_feature[i].value = 1;
    }
    ERROR_CHECK( cudaMalloc((void**)&dev_features, sizeof(cuSparseVector) * 20) )
    ERROR_CHECK( cudaMalloc((void**)&dev_key_feature, sizeof(cuSparseVector) * 6) )
    ERROR_CHECK( cudaMemcpy(dev_features, features, sizeof(cuSparseVector) * 20, cudaMemcpyHostToDevice) )
    ERROR_CHECK( cudaMemcpy(dev_key_feature, key_feature ,sizeof(cuSparseVector) * 6, cudaMemcpyHostToDevice) )
    
    featureScoreKernel<<<4, 256>>>(dev_features, 4, 5, dev_key_feature, 6);

    cudaFree(dev_features);
    cudaFree(dev_key_feature);
    free(features);
    free(key_feature);
}