#include <float.h>

#include <cuda_runtime.h>  
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "cuVocabulary.h"

#define imin(a, b) (a < b? a: b)

#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define CHECK(status) if (status != cudaSuccess) { \
    fprintf(stderr, "error in cuda\n"); \
    goto Error; \
}

#define INFO_CHECK(status, err_info) if (status != cudaSuccess) { \
    fprintf(stderr, err_info); \
    goto Error; \
}

#define CHECK_MAIN(status) if (status != cudaSuccess) { \
    fprintf(stderr, "error in cuda\n"); \
    return NULL; \
}

#define INFO_CHECK_MAIN(status, err_info) if (status != cudaSuccess) { \
    fprintf(stderr, err_info); \
    return NULL; \
}

//const int thread_per_block = 256;
//const int block_per_grid = 1;
const int thread_rows = 16;
const int thread_cols = 16;

__device__ struct cuNode* dev_vocabulary;
__device__ float* dev_descriptor_map;
__device__ int* dev_children_map;

__device__ size_t descriptor_pitch;
__device__ size_t children_pitch;

float* dev_descriptor;
float* dev_feature;

/**
 * 使用cuda查询词典
 * @vec: 待查的向量
 * @vec_len: 向量长度
 * @k: 词典为k叉树
*/
__global__ void findWordKernel(float *vec, size_t vec_num, size_t vec_len, float* feature)
{
    /// 行
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    int offset = bidx;
    while (offset < vec_num)
    {
        /// 记录父节点
        //__shared__ uint32 index;
        __shared__ struct cuNode* parent;
        /// 存储向量求模结果
        __shared__ float norm_array[thread_rows][thread_cols];
        /// 储存求取最小距离向量结果
        //__shared__ int min_offsets[thread_cols] = { 0 };
        __shared__ float min_distance;
        __shared__ int nearest_child;

        if(tidx == 0 && tidy == 0)
        {
            parent = dev_vocabulary;
        }
        else {}
        __syncthreads();

        while (parent->children_num != 0)
        {
            int children_index_offset = tidy;
            int children_index = parent->children_index;
            int count = 0;
            int loop_count = parent->children_num / blockDim.y + 1;

            if(tidx == 0 && tidy == 0)
            {
                nearest_child = -1;
                min_distance = FLT_MAX;
                //printf("parent id in cuda: %d\n", parent->id);
            }
            else {}
            //__syncthreads();
            if (tidx < 16 && tidy < 16)
            {
                norm_array[tidx][tidy] = 0.;
            }
            
            while(count < loop_count)
            {
                if (children_index_offset < parent->children_num)
                {
                    int* p_child = (int*)((char*)dev_children_map + children_index * children_pitch) + children_index_offset;
                    //printf("(%d: %d) ", tidy, child);
                    if (*p_child != -1)
                    {
                        int descriptor_offset = tidx;
                        //if (tidx < vec_len && tidy < parent.children_num)
                            //printf("child %d: (%d, %d): %f ", child, tidy, tidx, dev_descriptor_map[child * descriptor_pitch + descriptor_offset]);                                
                        while (descriptor_offset < vec_len)
                        {
                            float* p_norm = (float*)((char*)dev_descriptor_map + (*p_child) * descriptor_pitch) + descriptor_offset;
                            float* p_vec = (float*)((char*)vec + vec_len * offset) + descriptor_offset;
                            //float* p_vec = vec + vec_len * offset + descriptor_offset;
                            float norm = *p_norm - *p_vec;
                            norm_array[tidy][tidx] += norm * norm;
                            descriptor_offset += blockDim.x;
                        }
                        //
                    } 
                    else {}
                }
                else {}
                __syncthreads();

                /// 归约求模
                int half_cols = thread_cols / 2;
                while (half_cols > 0)
                {
                    if (tidx < half_cols)
                    {
                        norm_array[tidy][tidx] += norm_array[tidy][tidx + half_cols];
                    }
                    else {}
                    __syncthreads();
                    half_cols /= 2;
                }

                /*if (tidy < parent.children_num && tidx < vec_len)
                {
                    printf("(%d, %d): %f ", tidy, tidx, norm_array[tidy][tidx]);
                }*/

                /// 归约求最小值
                /*int reserve = k - thread_cols * count > thread_cols ? thread_cols: k - thread_cols * count;
                int y_offset = tidy;
                min_offsets[tidy] = tidy;
                while (reserve > 1)
                { 
                    if (tidy * 2 < reserve && tidx == 0)
                    {
                        int after_tidy = tid + (reserve + 1) / 2;
                        if (norm_array[min_offsets[tidy]][0] > norm_array[min_offsets[after_tidy]][0])
                            min_offsets[tidy] = min_offsets[after_tidy];
                        else {}
                    }
                    __syncthreads();
                    reverse = (reverse + 1) / 2;
                }
                if (tidx == 0 && tidy == 0)
                {
                    if (min_distance > norm_array[min_offsets[0]][0])
                    {
                        min_distance = norm_array[min_offsets[0]][0];
                        dev_res = children_index * children_pitch + min_offsets[0] + count * thread_cols;
                    }
                    else {}
                }*/
                /// 暴力求最小值
                if (tidx == 0 && tidy == 0)
                {
                    int reserve = parent->children_num - thread_cols * count > thread_cols ? thread_cols: parent->children_num - thread_cols * count;
                    for (int i = 0; i < reserve; i++)
                    {
                        
                        if (norm_array[i][0] < min_distance)
                        {
                            min_distance = norm_array[i][0];
                            nearest_child = count * blockDim.y + i;
                        }
                        else {}
                    }
                }
                __syncthreads();
                count++;
                children_index_offset += blockDim.y;
            }

            if (tidx == 0 && tidy == 0)
            {
                int* p_parent_index = (int*)((char*)dev_children_map + children_index * children_pitch) + nearest_child;
                parent = dev_vocabulary + *p_parent_index;
            }
            else {}
            __syncthreads();
        
        }
        //
        if (tidx == 0 && tidy == 0)
        {
            atomicAdd(&(feature[parent->word_id]), parent->weight);
        }
        else {}
        __syncthreads();

        offset += gridDim.x;
    }    
}

int freeVocabulary()
{
    /// 记录数据的内存地址
    struct cuNode* dev_vocabulary_temp = NULL;
    float* dev_descriptor_map_temp = NULL;
    int* dev_children_map_temp = NULL;

    /// 由符号连接到数据的内存地址
    ERROR_CHECK( cudaMemcpyFromSymbol((void*)&dev_vocabulary_temp, dev_vocabulary, sizeof(cuNode*)) )
    ERROR_CHECK( cudaMemcpyFromSymbol((void*)&dev_descriptor_map_temp, dev_descriptor_map, sizeof(float*)) )
    ERROR_CHECK( cudaMemcpyFromSymbol((void*)&dev_children_map_temp, dev_children_map, sizeof(int*)) )

    NULL_CHECK( dev_vocabulary_temp );
    NULL_CHECK( dev_descriptor_map_temp );
    NULL_CHECK( dev_children_map_temp );

    /// 释放内存
    cudaFree(dev_vocabulary_temp);
    cudaFree(dev_descriptor_map_temp);
    cudaFree(dev_children_map_temp);

    printf("free cuda data success!\n");
    return 1;

Error:
    fprintf(stderr, "free data failed!");
    return 0;
}

int initVocabulary()
{

    cudaError_t cudaStatus;
    //cudaStatus = cudaSetDevice(0);
    //INFO_CHECK(cudaStatus, "cudaSetDevice failed: Do you have a CUDA-capable GPU installed?\n")

    struct cuNode* dev_vocabulary_temp;
    float* dev_descriptor_map_temp;
    int* dev_children_map_temp;

    size_t descriptor_pitch_temp;
    size_t children_pitch_temp;

    //float* dev_feature_temp;

    /*cudaStatus = cudaMalloc((void**)&dev_feature_temp, sizeof(float) * word_num);
    INFO_CHECK(cudaStatus, "cudaMalloc for dev_res failed!\n")
    cudaStatus = cudaMemcpyToSymbol(dev_feature, (void*)&dev_vocabulary_temp, sizeof(float*));
    CHECK(cudaStatus)*/

    /*cudaStatus = cudaMalloc((void**)&dev_descriptor_temp, sizeof(float) * cols * rows);
    INFO_CHECK(cudaStatus, "cudaMalloc for descriptor failed!\n")
    cudaStatus = cudaMemcpyToSymbol(dev_descriptor, (void*)&dev_vocabulary_temp, sizeof(cuNode*));
    CHECK(cudaStatus)*/

    /// 申请词典内存
    cudaStatus = cudaMalloc((void**)&dev_vocabulary_temp, node_num * sizeof(cuNode));
    INFO_CHECK(cudaStatus, "cudaMalloc for vocabulary failed!\n")
    cudaStatus = cudaMemcpyToSymbol(dev_vocabulary, (void*)&dev_vocabulary_temp, sizeof(cuNode*));
    CHECK(cudaStatus)

    /*/// 申请单词描述向量内存
    cudaStatus = cudaMalloc((void**)&dev_descriptor_map, node_num * vector_row * sizeof(float));
    INFO_CHECK(cudaStatus, "cudaMalloc for descriptor map failed!")

    /// 申请节点内存
    cudaStatus = cudaMalloc((void**)&dev_children_map, nonleaf_node_num * cu_k * sizeof(int));
    INFO_CHECK(cudaStatus, "cudaMalloc for children map failed!")
    */

    //printf("probe: %f\n", descriptor_map[20]);
    /// 申请内存
    cudaStatus = cudaMallocPitch((void**)&dev_descriptor_map_temp, &descriptor_pitch_temp, 
        vector_row * sizeof(float), node_num);
    INFO_CHECK(cudaStatus, "cudaMallocPitch for descriptor map failed!\n")
    cudaStatus = cudaMemcpyToSymbol(dev_descriptor_map, (void*)&dev_descriptor_map_temp, sizeof(float*), size_t(0));
    INFO_CHECK(cudaStatus, "cudaMemcpyToSymbol failed!")
    
    cudaStatus = cudaMallocPitch((void**)&dev_children_map_temp, &children_pitch_temp, 
        cu_k * sizeof(int), node_num - word_num);
    INFO_CHECK(cudaStatus, "cudaMallocPitch for children failed\n")
    cudaStatus = cudaMemcpyToSymbol(dev_children_map, (void*)&dev_children_map_temp, sizeof(int*), size_t(0));
    INFO_CHECK(cudaStatus, "cudaMemcpyToSymbol failed!")

    /// 将pitch复制到全局变量
    cudaStatus = cudaMemcpyToSymbol(descriptor_pitch, (void*)&descriptor_pitch_temp, sizeof(size_t));
    cudaStatus = cudaMemcpyToSymbol(children_pitch, (void*)&children_pitch_temp, sizeof(size_t));
    INFO_CHECK(cudaStatus, "cudaMemcpyToSymbol failed!")

    //cudaStatus = cudaMalloc((void**)&dev_res, node_num * sizeof(cuNode));
    //INFO_CHECK(cudaStatus, "cudaMalloc for dev_res failed!\n")

    /// 复制数据
    cudaStatus = cudaMemcpy(dev_vocabulary_temp, cu_vocabulary, node_num * sizeof(cuNode), cudaMemcpyHostToDevice);
    INFO_CHECK(cudaStatus, "cudaMemcpy2D for vocabulary failed!\n")

    cudaStatus = cudaMemcpy2D(dev_descriptor_map_temp, descriptor_pitch_temp, 
        descriptor_map, vector_row * sizeof(float), 
        vector_row * sizeof(float), node_num,
        cudaMemcpyHostToDevice);
    INFO_CHECK(cudaStatus, "cudaMemcpy2D for descriptor map failed!\n")
    
    cudaStatus = cudaMemcpy2D(dev_children_map_temp, children_pitch_temp,
        children_map, cu_k * sizeof(int),
        cu_k *sizeof(int), node_num - word_num,
        cudaMemcpyHostToDevice);
    INFO_CHECK(cudaStatus, "cudaMemcpy2D for children map failed!\n")

    printf("init cuda successe!\n");
    return 1;
Error:

    freeVocabulary();
    return 0;
}

float* cudaFindWord(float* host_descriptor, size_t rows, size_t cols)
{
    dim3 dimBlock(thread_rows, thread_cols);
    float* feature;
    size_t cols_pitch;

    ERROR_CHECK( cudaSetDevice(0) )
    //printf("word_num: %d\n", word_num);

    /// 分配2维内存
    ERROR_CHECK( cudaMallocPitch((void**)&dev_descriptor, (size_t*)&cols_pitch, (size_t)sizeof(float) * cols, rows) )

    /// 传入数据到设备
    ERROR_CHECK( cudaMemcpy2D(dev_descriptor, cols_pitch, 
        host_descriptor, sizeof(float) * cols, sizeof(float) * cols, rows, cudaMemcpyHostToDevice) )

    //ERROR_CHECK ( cudaHostAlloc((void**)&dev_descriptor, sizeof(float) * cols * rows, cudaHostAllocDefault) )

    ERROR_CHECK( cudaMemcpy(dev_descriptor, host_descriptor, sizeof(float) * cols * rows, cudaMemcpyHostToDevice) )

    ERROR_CHECK( cudaHostAlloc((void**)&dev_feature, sizeof(float) * word_num, cudaHostAllocDefault) )

    /// 运行kernel函数
    findWordKernel<<<256, dimBlock>>>(dev_descriptor, rows, cols, dev_feature);

    /// 传出数据到内存
    //struct cuNode *res_node = (cuNode*)malloc(sizeof(cuNode));
    feature = (float*)malloc(sizeof(float) * word_num);
    ERROR_CHECK( cudaMemcpy(feature, dev_feature, sizeof(float) * word_num, cudaMemcpyDeviceToHost) )

    //freeVocabulary();
Error:
    cudaFreeHost(dev_feature);
    cudaFreeHost(dev_descriptor);

    return feature;
}