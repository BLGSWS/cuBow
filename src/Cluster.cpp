#include "Cluster.h"
#include <stdlib.h>
#define SEED 1984

namespace cuBoW {

Cluster::Cluster(uint32 k): cluster_k(k) 
{
    result.reserve(k);
    centers.reserve(k);
    srand(SEED);
}

Cluster::~Cluster()
{}

KmeansCluster::KmeansCluster(uint32 k): Cluster(k)
{}

void KmeansCluster::cluster(const std::vector<Eigen::VectorXf> &descriptors)
{
    result.clear();
    centers.clear();

    if(descriptors.size() <= cluster_k)
    {
        result.resize(cluster_k);
        for(size_t i = 0; i < descriptors.size(); i++)
        {
            std::vector<NodeId> group;
            group.push_back(i);
            centers.push_back(descriptors[i]);
            result[i] = group;
        }
    }
    else
    {
        bool finish_flag = false;
        std::set<NodeId> origin_centers_id;

        /// 生成初始聚类中心
        while(1)
        {
            NodeId id = rand()%descriptors.size();
            origin_centers_id.insert(id);
            if (origin_centers_id.size() == cluster_k) break;
            else {}
        }
        for (NodeId id : origin_centers_id)
        {
            centers.push_back(descriptors[id]);
        }

        /// 聚类
        while (!finish_flag)
        {
            result = DescManip::find_nearest_center(descriptors, centers);
            finish_flag = change_center(descriptors);
        }
    }
}

bool KmeansCluster::change_center(const std::vector<Eigen::VectorXf> &descriptors)
{
    int length = descriptors[0].rows();
    bool not_changed_flag = true;

    for (size_t i = 0; i < centers.size(); i++)
    {
        /// 聚类中心可能重复的情况
        /// 选择一个新中心, 但新选择的聚类中心也有可能是重复的- - 
        if (result[i].size() == 0)
        {
            bool less_flag = true;/// 唯一的特征是否少于聚类数
            for (Eigen::VectorXf feature : descriptors)
            {
                bool repate_flag = false;/// 某特征是否已被记录到聚类中心中
                for (Eigen::VectorXf center : centers)
                {
                    if (center != feature) continue;
                    else
                    {
                        repate_flag = true;
                        break;
                    }
                }
                /// 如果该特征未在聚类中心中出现, 则将此特征设为新的聚类中心
                if (repate_flag == false)
                {
                    centers[i] = feature;
                    less_flag = false;
                    break;
                }
                else {}
            }
            if (less_flag == true)
            {
                /// 方案1: 减少聚类中心
                result.erase(result.begin() + i);
                centers.erase(centers.begin() + i);

                /// 2: 允许相同的聚类中心存在，随机选取一个已有的聚类中心
                /*size_t index;
                do
                {
                    index = rand()%result.size();
                } while (result[index].size() != 0);
                result[i].assign(result[index].begin(), result[index].end());
                centers[i] = centers[index];*/

                std::cout << "KmeansCluster: number of clusters reduced" << std::endl;
                not_changed_flag = true;
            }
            else
                not_changed_flag = false;
        }
        else
        {
            /// 调整聚类中心
            Eigen::VectorXf new_center = Eigen::VectorXf::Zero(length);
            for (size_t j = 0; j < result[i].size(); j++)
            {
                new_center += descriptors[result[i][j]];
            }
            new_center /= float(result[i].size());
            if (centers[i] == new_center) continue;
            else
            {
                centers[i] = new_center;
                not_changed_flag = false;
            }
        }
    }
    return not_changed_flag;
}

HKmeansCluster::HKmeansCluster(uint32 k): Cluster(k)
{}

void HKmeansCluster::cluster(const std::vector<Eigen::VectorXf> &descriptors)
{}

float DescManip::distance(const Eigen::VectorXf &a, const Eigen::VectorXf &b)
{
    return (a - b).norm();
}

std::vector<std::vector<NodeId> > DescManip::find_nearest_center
(const std::vector<Eigen::VectorXf> &descriptors, const std::vector<Eigen::VectorXf> &centers)
{
    std::vector<std::vector<NodeId> > nearest_center;
    nearest_center.resize(centers.size());

    for (uint32 i = 0; i < descriptors.size(); i++)
    {
        float min_distance = distance(descriptors[i], centers[0]);
        uint32 index = 0;
        for (uint32 j = 1; j < centers.size(); j++)
        {
            float dis = distance(descriptors[i], centers[j]);
            if (dis < min_distance)
            {
                min_distance = dis;
                index = j;
            }
            else {}
        }
        nearest_center[index].push_back(i);
    }

    return nearest_center;
}

}// nampespace cuBow