#pragma once

#include "Type.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp> /// cv to eigen

#include "Cluster.h"

namespace cuBoW {

class Vocabulary
{
public:
    Vocabulary(int k, int L, 
    WeightingType weighting = TF_IDF, 
    ScoringType scoring = L1_NORM, 
    ClusterType cluster = KMEANS);
    Vocabulary();
    virtual ~Vocabulary();
public:
    virtual void create(const std::vector<cv::Mat> &training_features);
    virtual void create(const std::vector<std::vector<Eigen::VectorXf> > &training_features);
    virtual void create(const std::vector<std::vector<Eigen::VectorXf> > &training_features, int k, int L);
    virtual void create(const std::vector<cv::Mat> &training_features, int k, int L);
    virtual void initGPU();
    //virtual void create(const std::vector<Eigen::VectorXf> &training_features);
    /**
     * 向二进制文件中保存词典树
     * @param filepath: 二进制文件路径
     * */
    virtual void save(const std::string &filepath) const;
    virtual void read(const std::string &filepath);
    //static Vocabulary create_from_file(const std::string &filepath);
public:
    virtual Eigen::SparseVector<float> getFeature(const cv::Mat &mat) const;
    virtual Eigen::SparseVector<float> getFeature(const std::vector<Eigen::VectorXf> &descriptors) const;
    virtual Eigen::SparseVector<float> cudaGetFeature(const cv::Mat &mat) const;
    virtual Eigen::SparseVector<float> cudaGetFeature(const std::vector<Eigen::VectorXf> &descriptors) const;
protected:
    Eigen::SparseVector<float> cudaGetFeature(float* host_descriptors, uint32 rows, uint32 cols) const;
public:
    struct Node
    {
        /// 节点id
        NodeId id;
        /// 
        WordValue weight;
        /// 此节点的孩子节点
        std::vector<NodeId> children;
        /// 此节点的父节点
        NodeId parent;
        /// 描述子(聚类中心描述)
        Eigen::VectorXf descriptor;
        /// 
        WordId word_id;

        Node(): id(0), weight(0), parent(0), word_id(0) {}

        Node(NodeId _id): id(_id), weight(0), parent(0), word_id(0) {}

        //Node(NodeId _id, NodeId _parent, std::vector<NodeId> _children, )

        bool isLeaf() const { return children.empty(); }
    };

protected:
    void createScoringObject();
    void createClusterObject();
    void createLayer(NodeId parent_id, const std::vector<Eigen::VectorXf> &descriptors, uint32 current_level);
    void createWords();
    void setNodeWeight(const std::vector<std::vector<Eigen::VectorXf> > &descriptors);
    /**
     * 将数据转换成C形式
    */
    void transformData() const;
    WordId findWord(const Eigen::VectorXf &feature) const;
protected:
    /// 聚类数
    uint32 m_k;
    /// 树的深度
    uint32 m_d;
    /// Cluster method
    ClusterType m_cluster;
    /// Weighting method
    WeightingType m_weighting;
    /// Scoring method
    ScoringType m_scoring;
    /// Object for computing scores
    //GeneralScoring* m_scoring_object;
    /// 数组表示的树结构
    std::vector<Node> m_nodes;
    /// 叶子节点
    /// this condition holds: m_words[wid]->word_id == wid
    std::vector<Node*> m_words;
    /// 聚类器
    Cluster* m_cluster_object;
protected:
    bool GPU_flag;
public:
    const std::vector<Node>& getVocabulary() const { return m_nodes; }
    const std::vector<Node*>& getWords() const { return m_words; }
    //for debug (REMOVE)
    Node* getNodeWord(uint32 idx) const { return m_words[idx]; }
    WordId DBfindWord(const Eigen::VectorXf &feature) const;
};

} // namespcae CUBoW