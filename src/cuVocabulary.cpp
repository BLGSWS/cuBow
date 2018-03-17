#include "cuVocabulary.h"

uint32 node_num;
uint32 vector_row;
uint32 word_num;
uint32 cu_k;

struct cuNode* cu_vocabulary = NULL;
float* descriptor_map = NULL;
int* children_map = NULL;

void deleteData()
{
    NULL_CHECK( children_map )
    free(children_map);
    children_map = NULL;

    NULL_CHECK( descriptor_map )
    free(descriptor_map);
    descriptor_map = NULL;

    NULL_CHECK( cu_vocabulary )
    free(cu_vocabulary);
    cu_vocabulary = NULL;
    
}