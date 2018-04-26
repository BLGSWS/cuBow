#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Core>

#include <vector>
#include <set>
#include <string>
#include <iostream>

namespace cuBoW{

typedef unsigned int uint32;
typedef unsigned int NodeId;
typedef float WordValue;
typedef unsigned int WordId;
typedef unsigned int FrameId;

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
    HKMEANS,
    FCM
};

}