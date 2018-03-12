#pragma once
#include "BowVector.h"

namespace cuBoW{

class Cluster
{
public:
    Cluster(uint32 k = 3);
    virtual void initiateClusters() = 0;
    virtual void cluster(const std::vector<Eigen::VectorXf> &descriptors) = 0;
    const uint32& get_k() const { return cluster_k; }
    const std::vector<std::vector<NodeId> >& get_result() const { return result; }
    const std::vector<Eigen::VectorXf>& get_centers() const { return centers; }
    float value();
    virtual ~Cluster() = 0;
protected:
    uint32 cluster_k;
    std::vector<std::vector<NodeId> > result;
    std::vector<Eigen::VectorXf> centers;
};

class KmeansCluster: public Cluster
{
public:
    KmeansCluster(uint32 k = 3);
    void initiateClusters() {}
    void cluster(const std::vector<Eigen::VectorXf> &descriptors);
protected:
    bool change_center(const std::vector<Eigen::VectorXf> &descriptors);
private:
};

class HKmeansCluster: public Cluster
{
public:
    HKmeansCluster(uint32 k);
    void initiateClusters() {}
    void cluster(const std::vector<Eigen::VectorXf> &descriptors);
};

class DescManip
{
public:
    static float distance(const Eigen::VectorXf &a, const Eigen::VectorXf &b);
    static std::vector<std::vector<NodeId> > find_nearest_center(const std::vector<Eigen::VectorXf> &descriptors, const std::vector<Eigen::VectorXf> &centers);
};

} //namspace cuBow