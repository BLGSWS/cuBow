#include <cuda_runtime.h>  
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "cusparse.h"

#include "cuVocabulary.h"

#define tidx threadIdx.x
#define bidx blockIdx.x

#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define CUSPARSE_ERROR_CHECK(ans) { gpuSparseAssert((ans), __FILE__, __LINE__); }
inline void gpuSparseAssert(cusparseStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUSPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr,"GPUassert: %s %d\n", file, line);
        if (abort) exit(code);  
    }
}

const uint32 THREAD = 256;
const uint32 BLOCK = 256;

__global__ void normKernel(cudaTuple* feature_group, 
    uint32 m, uint32* csrRow,
    float* norms)
{
    __shared__ float distance_array[256];
    uint32 index_t = tidx;
    uint32 index_b = bidx;
    while (index_b < m)
    {
        distance_array[tidx] = 0.0;
        uint32 front = csrRow[index_b];
        uint32 rear = csrRow[index_b + 1];
        /*if (tidx == 0)
        {
            printf("front: %d, rear: %d, bidx: %d\n", front, rear, index_b);
        }*/
        while (index_t < rear - front)
        {
            float abs = feature_group[front + index_t].value;
            distance_array[tidx] +=  abs * abs;
            index_t += blockDim.x;
        }
        __syncthreads();
        index_t = tidx;

        uint32 offset = 128;
        while (offset > 0)
        {
            if (tidx < offset)
                distance_array[tidx] += distance_array[tidx + offset];
            __syncthreads();
            offset = offset >> 1;
        }

        if (tidx == 0)
        {
            norms[index_b] = __fsqrt_rn(distance_array[0]);
        }
        index_b += gridDim.x;
    }
}

/// 写二分查找竟然不能用无符号数！！！
__device__ uint32 binarySearch(cudaTuple* feature_group, int max, int min, uint32 index)
{
    while (min <= max)
    {
        uint32 mid = (max + min) >> 1;
        if (feature_group[mid].id < index) min = mid + 1;
        else if (feature_group[mid].id > index) max = mid - 1;
        else return mid;
    }
    return UINT_MAX;
}

__global__ void featureScoreKernel(cudaTuple* feature_group,
    uint32 m, uint32 n, uint32* csrRow, float* norms,
    float* score_list)
{
    //uint32 max_len = cols > col ? cols: col;
    __shared__ float distance_array[256];
    extern __shared__ uint8 flags[];
    uint32 index_t = tidx;
    uint32 index_b = bidx;
    while (index_b < m - 1)
    {
        distance_array[tidx] = 0.0;
        while (index_t < csrRow[1])
        {
            flags[index_t] = 0;
            index_t += blockDim.x;
        }
        index_t = tidx;
        __syncthreads();

        uint32 front = csrRow[index_b + 1];
        uint32 nnz = csrRow[index_b + 2] - front;
        float abs;
        while (index_t < nnz)
        {
            uint32 result = binarySearch(feature_group, csrRow[1] - 1, 0, 
                feature_group[front + index_t].id);
            if (result != UINT_MAX)
            {
                flags[result] = 1;
                abs = feature_group[result].value / norms[0]
                    - feature_group[index_t + front].value / norms[index_b + 1];
            }
            else
            {
                abs = fabsf(feature_group[index_t + front].value / norms[index_b + 1]);
            }
            distance_array[tidx] += abs * abs;
            index_t += blockDim.x;
        }
        index_t = tidx;

        while (index_t < csrRow[1])
        {
            if (flags[index_t] == 0)
            {
                abs = fabsf(feature_group[index_t].value / norms[0]);
                distance_array[tidx] += abs * abs;
            }
            index_t += blockDim.x;
        }
        __syncthreads();
        index_t = tidx;

        uint32 offset = 128;
        while (offset > 0)
        {
            if (tidx < offset)
                distance_array[tidx] += distance_array[tidx + offset];
            offset = offset >> 1;
            __syncthreads();
        }

        if (tidx == 0)
        {
            score_list[index_b] = 1.0 - 0.5 * __fsqrt_rn(distance_array[0]);
        }

        index_b += gridDim.x;
    }

}

/**
 * -0.5 * ||x - y| - x - y| if x->first == y->first
 * */
__global__ void featureScoreKernel_v2(cudaTuple* feature_group,
    uint32 m, uint32 n, uint32* csrRow,
    float* score_list)
{
    __shared__ float distance_array[256];
    uint32 index_b = bidx;
    while (index_b < m - 1)
    {
        uint32 index_t = tidx;
        distance_array[tidx] = 0.0;
        uint32 front = csrRow[index_b + 1];
        uint32 nnz = csrRow[index_b + 2] - front;
        while (index_t < nnz)
        {
            uint32 result = binarySearch(feature_group, csrRow[1] - 1, csrRow[0], 
                feature_group[front + index_t].id);
            float abs;
            if (result != UINT_MAX)
            {
                abs = fabs(feature_group[result].value) + fabs(feature_group[index_t + front].value)
                - fabsf(feature_group[result].value - feature_group[index_t + front].value);
            }
            else
            {
                abs = 0.;
            }
            distance_array[tidx] += abs;
            index_t += blockDim.x;
        }
        index_t = tidx;

        uint32 offset = 128;
        while (offset > 0)
        {
            if (tidx < offset)
                distance_array[tidx] += distance_array[tidx + offset];
            offset = offset >> 1;
            __syncthreads();
        }

        if (tidx == 0)
        {
            score_list[index_b] = 2 * distance_array[0];
        }

        index_b += gridDim.x;
    }
}

std::vector<float> cudaFeatureScore(cudaTuple* feature_group, uint32 m, uint32 n, uint32 nnz, uint32* csrRow)
{
    struct cudaTuple* dev_feature_group;
    float* dev_norms;
    uint32* dev_csrRow;
    float* dev_scores;

    ERROR_CHECK( cudaMalloc((void**)&dev_feature_group, sizeof(cudaTuple) * nnz) )
    ERROR_CHECK( cudaMalloc((void**)&dev_norms, sizeof(float) * m) )
    ERROR_CHECK( cudaMalloc((void**)&dev_csrRow, sizeof(uint32) * (m + 1)) )
    ERROR_CHECK( cudaMalloc((void**)&dev_scores, sizeof(float) * (m - 1)) )

    ERROR_CHECK( cudaMemcpy(dev_feature_group, feature_group, sizeof(cudaTuple) * nnz, cudaMemcpyHostToDevice) )
    ERROR_CHECK( cudaMemcpy(dev_csrRow, csrRow, sizeof(uint32) * (m + 1), cudaMemcpyHostToDevice) )

    normKernel<<<BLOCK, THREAD>>>(dev_feature_group, m, dev_csrRow, dev_norms);
    featureScoreKernel<<<BLOCK, THREAD, sizeof(uint8) * csrRow[1]>>>(dev_feature_group, m, n, dev_csrRow, dev_norms, dev_scores);

    float* host_scores = (float*)malloc(sizeof(float) * (m - 1));
    NULL_CHECK( host_scores )

    ERROR_CHECK( cudaMemcpy(host_scores, dev_scores, sizeof(float) * (m - 1), cudaMemcpyDeviceToHost) )

    cudaFree(dev_feature_group);
    cudaFree(dev_norms);
    cudaFree(dev_csrRow);
    cudaFree(dev_scores);

    std::vector<float> result;
    result.reserve(m - 1);
    for (int i = 0; i < m - 1; i++)
    {
        result.push_back(host_scores[i]);
    }
    free(host_scores);
    host_scores = NULL;

    return result;
}

std::vector<float> cudaFeatureScore_v2(cudaTuple* feature_group, uint32 m, uint32 n, uint32 nnz, uint32* csrRow)
{
    struct cudaTuple *dev_feature_group;
    uint32 *dev_csrRow;
    float *dev_scores;

    ERROR_CHECK( cudaMalloc((void**)&dev_feature_group, sizeof(cudaTuple) * nnz) )
    ERROR_CHECK( cudaMalloc((void**)&dev_csrRow, sizeof(uint32) * (m + 1)))
    ERROR_CHECK( cudaMalloc((void**)&dev_scores, sizeof(float) * (m - 1)) )

    ERROR_CHECK( cudaMemcpy(dev_feature_group, feature_group, sizeof(cudaTuple) * nnz, cudaMemcpyHostToDevice) )
    ERROR_CHECK( cudaMemcpy(dev_csrRow, csrRow, sizeof(uint32) * (m + 1), cudaMemcpyHostToDevice) )

    featureScoreKernel_v2<<<m, THREAD>>>(dev_feature_group, m, n, dev_csrRow, dev_scores);

    float* scores = new float[m - 1];
    NULL_CHECK( scores )

    ERROR_CHECK( cudaMemcpy(scores, dev_scores, sizeof(float) * (m - 1), cudaMemcpyDeviceToHost) )

    cudaFree(dev_feature_group);
    cudaFree(dev_csrRow);
    cudaFree(dev_scores);

    std::vector<float> result;
    result.reserve(m - 1);
    for (int i = 0; i < m - 1; i++)
    {
        result.push_back(scores[i]);
    }
    delete [] scores;

    return result;
}

uint32 findMaxNnz(const std::vector<QueryResult> &results)
{
    uint32 max_nnz = 0;
    for (auto result : results)
    {
        if (result.nnz > max_nnz) max_nnz = result.nnz;
    }
    return max_nnz;
}

std::vector<float> cudaFeatureScore_stream_v2(const std::vector<QueryResult> &results, const Eigen::SparseVector<float> &feature)
{
    struct cudaTuple *dev_feature_group1, *dev_feature_group2, *host_feature_group1, *host_feature_group2;
    uint32 *dev_csrRow1, *dev_csrRow2, *host_csrRow1, *host_csrRow2;
    float *dev_scores1, *dev_scores2, *host_scores1, *host_scores2;
    uint32 m = results[0].feature_ptrs.size() + 1;
    uint32 n = feature.size();
    uint32 nnz = findMaxNnz(results) + feature.nonZeros();
    std::vector<float> scores;
    if (1)
    {
        cudaStream_t stream1, stream2;
        ERROR_CHECK( cudaStreamCreate(&stream1) )
        ERROR_CHECK( cudaStreamCreate(&stream2) )

        ERROR_CHECK( cudaHostAlloc((void**)&host_feature_group1, 
                                    sizeof(cudaTuple) * nnz, cudaHostAllocDefault) )
        ERROR_CHECK( cudaHostAlloc((void**)&host_feature_group2, 
                                    sizeof(cudaTuple) * nnz, cudaHostAllocDefault) )
        ERROR_CHECK( cudaHostAlloc((void**)&host_csrRow1, sizeof(uint32) * (m + 1), cudaHostAllocDefault) )
        ERROR_CHECK( cudaHostAlloc((void**)&host_csrRow2, sizeof(uint32) * (m + 1), cudaHostAllocDefault) )
        ERROR_CHECK( cudaHostAlloc((void**)&host_scores1, sizeof(float) * (m - 1), cudaHostAllocDefault) )
        ERROR_CHECK( cudaHostAlloc((void**)&host_scores1, sizeof(float) * (m - 1), cudaHostAllocDefault) )

        ERROR_CHECK( cudaMalloc((void**)&dev_feature_group1, sizeof(cudaTuple) * nnz) )
        ERROR_CHECK( cudaMalloc((void**)&dev_csrRow1, sizeof(uint32) * (m + 1)) )
        ERROR_CHECK( cudaMalloc((void**)&dev_scores1, sizeof(float) * (m - 1)) )
        ERROR_CHECK( cudaMalloc((void**)&dev_feature_group2, sizeof(cudaTuple) * nnz) )
        ERROR_CHECK( cudaMalloc((void**)&dev_csrRow2, sizeof(uint32) * (m + 1)) )
        ERROR_CHECK( cudaMalloc((void**)&dev_scores2, sizeof(float) * (m - 1)) )

        for (uint32 i = 0; i < results.size(); i += 2)
        {
            uint32 m1 = results[i].feature_ptrs.size() + 1;
            uint32 nnz1 = results[i].nnz + feature.nonZeros();
            uint32 m2, nnz2;

            for (Eigen::SparseVector<float>::InnerIterator it(feature); it; ++it)
            {
                host_feature_group1[i].id = it.index();
                host_feature_group1[i].value = it.value();
                host_feature_group2[i].id = it.index();
                host_feature_group2[i].value = it.value();
            }

            host_csrRow1[0] = 0; host_csrRow1[1] = feature.nonZeros();
            uint32 offset = feature.nonZeros();
            for (uint32 j = 0; j < results[i].feature_ptrs.size(); j++)
            {
                results[i].feature_ptrs[i].copyData(host_feature_group1 + offset);
                host_csrRow1[j + 2] = host_csrRow1[j + 1] + results[i].feature_ptrs[j].offset;
                offset += results[i].feature_ptrs[j].offset;
            }

            if (i + 1 != results.size())
            {
                host_csrRow2[0] = 0; host_csrRow2[2] = feature.nonZeros();
                offset = feature.nonZeros();
                for (uint32 j = 0; j < results[i + 1].feature_ptrs.size(); j++)
                {
                    results[i + 1].feature_ptrs[i].copyData(host_feature_group2 + offset);
                    host_csrRow2[j + 2] = host_csrRow2[j + 1] + results[i + 1].feature_ptrs[j].offset;
                    offset += results[i + 1].feature_ptrs[j].offset;
                }
                m2 = results[i + 1].feature_ptrs.size() + 1;
                nnz2 = results[i + 1].nnz + feature.nonZeros();
            }
            else
            {
                m2 = 1;
                nnz2 = feature.nonZeros();
            }

            ERROR_CHECK( cudaMemcpyAsync(dev_feature_group1, host_feature_group1,
                                        nnz1 * sizeof(cudaTuple),
                                        cudaMemcpyHostToDevice, stream1) )
            ERROR_CHECK( cudaMemcpyAsync(dev_feature_group2, host_feature_group2,
                                        nnz2 * sizeof(cudaTuple),
                                        cudaMemcpyHostToDevice, stream2) )
            ERROR_CHECK( cudaMemcpyAsync(dev_csrRow1, host_csrRow1,
                                        (m1 + 1) * sizeof(uint32),
                                        cudaMemcpyHostToDevice, stream1) )
            ERROR_CHECK( cudaMemcpyAsync(dev_csrRow2, host_csrRow2,
                                        (m2 + 1) * sizeof(uint32),
                                        cudaMemcpyHostToDevice, stream2) )
            
            featureScoreKernel_v2<<<128, 256, 0, stream1>>>(dev_feature_group1, m1, n, dev_csrRow1, dev_scores1);
            featureScoreKernel_v2<<<128, 256, 0, stream2>>>(dev_feature_group2, m2, n, dev_csrRow2, dev_scores2);

            ERROR_CHECK( cudaMemcpyAsync(host_scores1, dev_scores1, sizeof(float) * (m1 - 1),
                                         cudaMemcpyDeviceToHost, stream1) )
            ERROR_CHECK( cudaMemcpyAsync(host_scores2, dev_scores2, sizeof(float) * (m2 - 1),
                                         cudaMemcpyDeviceToHost, stream2) )
            
            for (uint32 j = 0; j < m1 - 1; j++)
            {
                scores.push_back(host_scores1[j]);
            }
            for (uint32 j = 0; j < m2 - 1; j++ )
            {
                scores.push_back(host_scores2[j]);
            }
        }
        ERROR_CHECK( cudaStreamSynchronize(stream1) )
        ERROR_CHECK( cudaStreamSynchronize(stream2) )

        cudaFreeHost(host_feature_group1);
        cudaFreeHost(host_feature_group2);
        cudaFreeHost(host_csrRow1);
        cudaFreeHost(host_csrRow2);
        cudaFreeHost(host_scores1);
        cudaFreeHost(host_scores2);
        cudaFree(dev_feature_group1);
        cudaFree(dev_feature_group2);
        cudaFree(dev_csrRow1);
        cudaFree(dev_csrRow2);
        cudaFree(dev_scores1);
        cudaFree(dev_scores2);
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
    }
    return scores;
}