#include "cuVocabulary.h"

uint32 node_num;
uint32 vector_row;
uint32 word_num;
uint32 cu_k;

struct cudaNode* node_map = nullptr;
float* descriptor_map = nullptr;
int* children_map = nullptr;

void deleteData()
{
    NULL_CHECK( children_map )
    free(children_map);
    children_map = NULL;

    NULL_CHECK( descriptor_map )
    free(descriptor_map);
    descriptor_map = NULL;

    NULL_CHECK( node_map )
    free(node_map);
    node_map = NULL;
    
}