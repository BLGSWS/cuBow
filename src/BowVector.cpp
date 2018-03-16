#include "BowVector.h"

namespace cuBoW{
BowVector::BowVector(void)
{

}

BowVector::~BowVector(void)
{

}

void BowVector::transformData(cuBowVector* cuvec) const
{
    NULL_CHECK( cuvec )
    uint32 i = 0;
    BowVector::const_iterator it = this->begin();
    for (it; it != this->end(); it++)
    {
        cuvec[i].id = it->first;
        cuvec[i].value = it->second;
        i++;
    }
}

}