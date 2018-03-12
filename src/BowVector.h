#pragma once
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Core>
#include <opencv2/core/eigen.hpp> /// cv to eigen
#include <eigen3/Eigen/Dense>
#include <vector>
#include <set>
#include <iostream>

namespace cuBoW{

typedef unsigned int uint32;
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


} // namespace CUBoW