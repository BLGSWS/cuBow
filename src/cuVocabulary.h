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

struct cuNode
{
    uint32 id;
    float weight;
    size_t children_num;
    int children_index;
    uint32 parent;
    uint32 word_id;
};

struct cuSparseVector
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
extern struct cuNode* cu_vocabulary;

std::vector<cuSparseVector> cudaFindWord(float* descriptors, size_t rows, size_t cols);
/// test
void cudaFeatureScore();

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


