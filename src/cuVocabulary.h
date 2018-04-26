#pragma once

#include <iostream>
#include <vector>

#define uint32 unsigned int
#define uint16 unsigned long
#define uint8 unsigned char

/// 检查空指针
#define VNAME(name) (#name)
#define NULL_CHECK(ptr) if(ptr == NULL) { \
    printf("null ptr: %s %s %d\n", VNAME(ptr), __FILE__, __LINE__); \
    exit(0); }

#define imin(a, b) (a < b? a: b)
#define imax(a, b) (a < b? b: a)

#ifdef __cplusplus
extern "C" {
#endif

struct cudaNode
{
    float weight;
    size_t children_num;
    int children_index;
    uint32 parent;
    uint32 word_id;
};

struct cudaTuple
{
    uint32 id;
    float value;
};

#ifdef __cplusplus
}
#endif

extern uint32 word_num;
extern uint32 node_num;
extern uint32 vector_row;
extern uint32 cu_k;

extern int* children_map;
extern float* descriptor_map;
extern struct cudaNode* node_map;

std::vector<cudaTuple> cudaFindWord(float* descriptors, size_t rows, size_t cols);
std::vector<float> cudaFeatureScore(cudaTuple* feature_group, uint32 m, uint32 n, uint32 nnz, uint32* csrRow);

/// test
void cudaFeatureScoreTest();
void cuSparseTest();

/**
 * 释放内存
*/
void deleteData();
/**
 * 初始化GPU 
 */
int initVocabulary();
/**
 * 释放GPU 
 */
int freeVocabulary();


