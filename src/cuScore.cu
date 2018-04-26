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

__device__ uint32 binarySearch(cudaTuple* feature_group, uint32 max, uint32 min, uint32 index)
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
    // const uint32 f_front = csrRow[0];
    // const uint32 f_rear = csrRow[1];
    float abs;
    while (index_b < m - 1)
    {
        distance_array[tidx] = 0.0;
        while (index_t < csrRow[1] - csrRow[0])
        {
            flags[index_t] = 0;
            index_t += blockDim.x;
        }
        index_t = tidx;

        if (tidx == 0)
        {
            score_list[index_b] = - norms[0] - norms[index_b + 1];
            //printf("%d: %f, %f, %f\n", index_b, norms[0], norms[index_b + 1], score_list[index_b]);
        }

        uint32 front = csrRow[index_b + 1];
        uint32 rear = csrRow[index_b + 2];
        while (index_t < rear - front)
        {
            uint32 result = binarySearch(feature_group, csrRow[1] - csrRow[0], 0, feature_group[front + index_t].id);
            if (result != UINT_MAX)
            {
                flags[result] = 1;
                abs = fabsf(feature_group[result].value - feature_group[index_t + front].value);
            }
            else
            {
                abs = fabsf(feature_group[index_t + front].value);
            }
            distance_array[tidx] += abs * abs;
            index_t += blockDim.x;
        }
        index_t = tidx;

        while (index_t < csrRow[1] - csrRow[0])
        {
            if (flags[index_t] == 0)
            {
                abs = fabsf(feature_group[index_t].value);
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
            score_list[index_b] += __fsqrt_rn(distance_array[0]);
            //printf("%d: %f   ", index_b, score_list[index_b]);
        }

        index_b += gridDim.x;
    }

}

#ifdef SCORE_DEBUG
void cudaFeatureScoreTest()
{
    struct cudaTuple* feature_group = (cudaTuple*)malloc(sizeof(cudaTuple) * 9);
    float* norms = new float[8];
    float* normB = new float;
    uint32* csrRow = new uint32[5];
    
    feature_group[0].id = 0; feature_group[0].value = 1;
    feature_group[1].id = 3; feature_group[1].value = 3;
    feature_group[2].id = 1; feature_group[2].value = 7;
    feature_group[3].id = 3; feature_group[3].value = 5;
    feature_group[4].id = 2; feature_group[4].value = 2;
    feature_group[5].id = 3; feature_group[5].value = 4;
    feature_group[6].id = 5; feature_group[6].value = 1;
    feature_group[7].id = 1; feature_group[7].value = 1;
    feature_group[8].id = 2; feature_group[8].value = 2;

    csrRow[0] = 0; csrRow[1] = 2; csrRow[2] = 4; csrRow[3] = 7; csrRow[4] = 9;

    struct cudaTuple* dev_feature_group;
    float* dev_norms;
    float* dev_normB;
    uint32* dev_csrRow;
    float* dev_score_list;

    ERROR_CHECK( cudaMalloc((void**)&dev_feature_group, sizeof(cudaTuple) * 9) )
    ERROR_CHECK( cudaMalloc((void**)&dev_norms, sizeof(float) * 4) )
    ERROR_CHECK( cudaMalloc((void**)&dev_normB, sizeof(float)) )
    ERROR_CHECK( cudaMalloc((void**)&dev_csrRow, sizeof(uint32) * 5) )
    ERROR_CHECK( cudaMalloc((void**)&dev_score_list, sizeof(float) * 3) )

    ERROR_CHECK( cudaMemcpy(dev_feature_group, feature_group, sizeof(cudaTuple) * 9, cudaMemcpyHostToDevice) )
    ERROR_CHECK( cudaMemcpy(dev_csrRow, csrRow, sizeof(uint32) * 5, cudaMemcpyHostToDevice) )
    
    normKernel<<<2, THREAD>>>(dev_feature_group, 4, dev_csrRow, dev_norms);
    featureScoreKernel<<<2, THREAD, sizeof(uint8) * 2>>>(dev_feature_group, 4, 6, dev_csrRow, dev_norms, dev_score_list);
 
    cudaFree(dev_feature_group);
    cudaFree(dev_norms);
    cudaFree(dev_normB);
    cudaFree(dev_csrRow);
    cudaFree(dev_score_list);
    free(feature_group);
    free(norms);
    free(normB);
    free(csrRow);
}

void cuSparseTest()
{
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;

    CUSPARSE_ERROR_CHECK( cusparseCreateMatDescr(&descr) )
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    CUSPARSE_ERROR_CHECK( cusparseCreate(&handle) )

    std::cout << "successed!" << std::endl;
}

#endif

std::vector<float> cudaFeatureScore(cudaTuple* feature_group, uint32 m, uint32 n, uint32 nnz, uint32* csrRow)
{
    struct cudaTuple* dev_feature_group;
    float* dev_norms;
    uint32* dev_csrRow;
    float* dev_scores;

    ERROR_CHECK( cudaMalloc((void**)&dev_feature_group, sizeof(cudaTuple) * nnz) )
    ERROR_CHECK( cudaMalloc((void**)&dev_norms, sizeof(float) * m) )
    ERROR_CHECK( cudaMalloc((void**)&dev_csrRow, sizeof(uint32) * (nnz + 1)) )
    ERROR_CHECK( cudaMalloc((void**)&dev_scores, sizeof(float) * (m - 1)) )

    ERROR_CHECK( cudaMemcpy(dev_feature_group, feature_group, sizeof(cudaTuple) * nnz, cudaMemcpyHostToDevice) )
    ERROR_CHECK( cudaMemcpy(dev_csrRow, csrRow, sizeof(uint32) * (nnz + 1), cudaMemcpyHostToDevice) )

    normKernel<<<BLOCK, THREAD>>>(dev_feature_group, m, dev_csrRow, dev_norms);
    featureScoreKernel<<<BLOCK, THREAD, sizeof(uint8) * 2>>>(dev_feature_group, m, n, dev_csrRow, dev_norms, dev_scores);

    float* scores = new float[m - 1];
    NULL_CHECK( scores )

    ERROR_CHECK( cudaMemcpy(scores, dev_scores, sizeof(float) * (m - 1), cudaMemcpyDeviceToHost) )

    cudaFree(dev_feature_group);
    cudaFree(dev_norms);
    cudaFree(dev_csrRow);
    cudaFree(dev_scores);

    std::vector<float> result;
    result.reserve(m - 1);
    for (int i = 0; i < m - 1; i++)
    {
        result.push_back(scores[i]);
    }

    delete [] scores;
    cudaFree(dev_feature_group);
    cudaFree(dev_norms);
    cudaFree(dev_csrRow);
    cudaFree(dev_scores);

    return result;
}