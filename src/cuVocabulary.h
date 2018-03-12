#ifndef _CUBOWVECTOR_H
#define _CUBOWVECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

#define uint32 unsigned int
#define VNAME(name) (#name)
#define NULL_CHECK(ptr) if(ptr == NULL) { null_check(ptr, __FILE__, __LINE__); }

void null_check(void* ptr, const char *file, int line);

struct cuNode
{
    uint32 id;
    float weight;
    size_t children_num;
    int children_index;
    uint32 parent;
    uint32 word_id;
};

extern uint32 word_num;
extern uint32 node_num;
extern uint32 vector_row;
extern uint32 cu_k;

extern int* children_map;
extern float* descriptor_map;
extern struct cuNode* cu_vocabulary;

/**
 * 释放内存
*/
void deleteData();

float* cudaFindWord(float* descriptors, size_t rows, size_t cols);
int initVocabulary();
int freeVocabulary();

#ifdef __cplusplus
}
#endif

#endif


