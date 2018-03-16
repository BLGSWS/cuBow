#pragma once
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Core>
#include <opencv2/core/eigen.hpp> /// cv to eigen
#include <eigen3/Eigen/Dense>
#include <vector>
#include <set>
#include <iostream>

#include "cuVocabulary.h"

namespace cuBoW{

//typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

typedef unsigned int NodeId;
typedef float WordValue;
typedef unsigned int WordId;

enum LNorm { L1, L2 };

enum WeightingType
{
    TF_IDF,
    TF,
    IDF,
    BINARY
};

enum ScoringType
{
    L1_NORM,
    L2_NORM,
    CHI_SQUARE,
    KL,
    BHATTACHARYYA,
    DOT_PRODUCT
};

enum ClusterType
{
    KMEANS,
    HKMEANS
};

class BowVector: public std::map<WordId, WordValue>
{
public:
    BowVector(void);

    ~BowVector(void);

    /**
    * INLINE
    */
    void addWeight(WordId id, WordValue value)
    {
        BowVector::iterator pos = this->lower_bound(id);
        if (pos != this->end() && pos->first == id)
        {
            pos->second += value;
        }
        else
        {
            this->insert(pos, BowVector::value_type(id, value));
        }
    }

    void normalize(LNorm norm_type);

    void transformData(cuBowVector* cuvec) const;

private:


};


} // namespace CUBoW