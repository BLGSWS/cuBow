#include "Vocabulary.h"
#include "cuVocabulary.h"

namespace cuBoW{

Vocabulary::Vocabulary(
    int k, int L, WeightingType weighting,  ScoringType scoring, ClusterType cluster):
    m_k(k), m_d(L), m_weighting(weighting), m_scoring(scoring), m_cluster(cluster), 
    m_cluster_object(0)
{
    createClusterObject();
}


void Vocabulary::createScoringObject()
{

}

void Vocabulary::createClusterObject()
{
    if (m_cluster_object != 0)
    {
        delete m_cluster_object;
        m_cluster_object = 0;
    }
    else {}

    switch(m_cluster)
    {
        case KMEANS:
            m_cluster_object = new KmeansCluster(m_k);
            break;
        
        case HKMEANS:
            m_cluster_object = new HKmeansCluster(m_k);
            break;
        
        default:
            std::cerr << "Vocabulary::createClusterObject: no such cluster type: " << m_cluster;
            std::cerr << std::endl;
            throw std::exception();
    }
}

void Vocabulary::create(const std::vector<cv::Mat> &training_features)
{
    std::vector<std::vector<Eigen::VectorXf> > vtb(training_features.size());
    for (auto mat : training_features)
    {
        std::vector<Eigen::VectorXf> temp;
        for (uint32 j = 0; j < mat.rows; j++)
        {
            cv::Mat slice = mat.rowRange(j, j + 1);
            Eigen::RowVectorXf vec;
            cv::cv2eigen(slice, vec);
            temp.push_back(vec.transpose());
        }
        vtb.push_back(temp);
    }
    create(vtb);
}

void Vocabulary::create(
    const std::vector<std::vector<Eigen::VectorXf> > &training_features)
{
    m_nodes.clear();
    m_words.clear();
    /// 词典为完全m_k叉树
    int expected_nodes = (int)(pow((double)m_k, (double)m_d) - 1) / (m_k - 1);
    m_nodes.reserve(expected_nodes);

    std::vector<Eigen::VectorXf> features;
    for (size_t i = 0; i < training_features.size(); i++)
    {
        for (size_t j = 0; j < training_features[i].size(); j++)
        {
            features.push_back(training_features[i][j]);
        }
    }
    /// 建立第一个节点, 第一个节点为第0层
    Node root(0);
    root.descriptor = Eigen::MatrixXf::Zero(features[0].rows(), 1);
    m_nodes.push_back(root);
    /// 递归地建立词典
    createLayer(0, features, 1);
    /// 建立单词
    createWords();
    /// 设置IDF权重
    setNodeWeight(training_features);
    /// 转移数据
    //transformData();
}

void Vocabulary::create(
    const std::vector<std::vector<Eigen::VectorXf> > &training_features, int k, int L)
{
    m_k = k;
    m_d = L;
    create(training_features);
}

void Vocabulary::create(
    const std::vector<cv::Mat> &training_features, int k, int L)
{
    m_k = k;
    m_d = L;
    create(training_features);
}

void Vocabulary::createLayer(
    NodeId parent_id, const std::vector<Eigen::VectorXf> &descriptors, uint32 current_level)
{
    if (descriptors.empty()) return;
    else {}
    /// 聚类
    m_cluster_object->cluster(descriptors);
    std::vector<std::vector<NodeId> > result = m_cluster_object->get_result();
    std::vector<Eigen::VectorXf> centers = m_cluster_object->get_centers();
    for (size_t i = 0; i < centers.size(); i++)
    {
        uint32 current_id = m_nodes.size();
        Node node(current_id);
        node.parent = parent_id;
        node.descriptor = centers[i];
        //std::cout << node.descriptor.rows() << " " << node.descriptor.cols() << std::endl;
        //std::cout << centers[i] << std::endl << std::endl;
        m_nodes.push_back(node);
        m_nodes[parent_id].children.push_back(current_id);
    }

    if (current_level == m_d) return;
    else {}

    /// 递归地对聚类的结果再进行聚类
    for(size_t i = 0; i < result.size(); i++)
    {
        std::vector<Eigen::VectorXf> sub_descriptors;
        sub_descriptors.reserve(result[i].size());
        NodeId id = m_nodes[parent_id].children[i];
        for(size_t j = 0; j < result[i].size(); j++)
        {
            int index = result[i][j];
            sub_descriptors.push_back(descriptors[index]);
        }
        if(sub_descriptors.size() > 1)
            createLayer(id, sub_descriptors, current_level + 1);
        else {}
    }

}

void Vocabulary::createWords()
{
    m_words.resize(0);
    if (!m_nodes.empty())
    {
        m_words.reserve((int)pow((double)m_k, (double)m_d));
        auto it = m_nodes.begin();
        for (++it; it != m_nodes.end(); it++)
        {
            if(it->isLeaf())
            {
                it->word_id = m_words.size();
                m_words.push_back(&(*it));
            }
            else {}
        }
    }
    else {}
}

void Vocabulary::setNodeWeight(
    const std::vector<std::vector<Eigen::VectorXf> > &descriptors)
{
    int word_num = m_words.size();
    int docs_num = descriptors.size();

    /// TF和BINARY两种情况与IDF值无关
    if (m_weighting == TF || m_weighting == BINARY)
    {
        for (uint32 i = 0; i < word_num; i++)
        {
            m_words[i]->weight = 1.;
        }
    }
    /// 设置IDF值
    else if (m_weighting == IDF || m_weighting == TF_IDF)
    {
        /// 记录单词在多少篇文档里面出现过
        std::vector<int> count(word_num, 0);
        /// 遍历文档
        for (const std::vector<Eigen::VectorXf>& doc : descriptors)
        {
            /// 哈希表，记录单词是否在同一文档中重复出现
            std::vector<bool> repate(word_num, false);
            for (const Eigen::VectorXf& feature : doc)
            {
                WordId id = findWord(feature);
                if (repate[id] == false)
                {
                    repate[id] = true;
                    count[id] += 1;
                }
                else {}
            }
        }
        for (uint32 i = 0; i < word_num; i++)
        {
            m_words[i]->weight = log((float)docs_num / ((float)count[i] + 1));
            /// 按照DoW3的意思，count不存在等于0的元素
        }
    }
    else {}
}

WordId Vocabulary::findWord(const Eigen::VectorXf &feature) const
{
    NodeId final_id = 0;
    while (!m_nodes[final_id].isLeaf())
    {
        const std::vector<NodeId>& children = m_nodes[final_id].children;
        /// 找到最近的聚类中心或单词
        float min_distance = DescManip::distance(feature, m_nodes[children[0]].descriptor);
        final_id = children[0];
        for (size_t i = 1; i < children.size(); i++)
        {
            float distance = DescManip::distance(feature, m_nodes[children[i]].descriptor);
            if (distance < min_distance)
            {
                min_distance = distance;
                final_id = children[i];
            }
            else {}
        }
    }
    /// 返回wordid
    return m_nodes[final_id].word_id;
}

WordId Vocabulary::DBfindWord(const Eigen::VectorXf &feature)
{
    NodeId final_id = 0;
    while (!m_nodes[final_id].isLeaf())
    {
        //std::cout << "parent id: " << m_nodes[final_id].id << std::endl;
        std::vector<NodeId>& children = m_nodes[final_id].children;
        /// 找到最近的聚类中心或单词
        float min_distance = DescManip::distance(feature, m_nodes[children[0]].descriptor);
        final_id = children[0];
        for (size_t i = 1; i < children.size(); i++)
        {
            float distance = DescManip::distance(feature, m_nodes[children[i]].descriptor);
            if (distance < min_distance)
            {
                min_distance = distance;
                final_id = children[i];
            }
            else {}
        }
        //std::cout << "min_distance: " << min_distance << std::endl;
    }
    /// 返回wordid
    std::cout << "id: " << m_nodes[final_id].id << std::endl;
    return m_nodes[final_id].word_id;
}

Vocabulary::~Vocabulary()
{
    //deleteData();
    delete m_cluster_object;
    m_cluster_object = nullptr;
}

 CudaVocabulary::CudaVocabulary(int k, int L, 
    WeightingType weighting, ScoringType scoring, ClusterType cluster) : Vocabulary(k, L, weighting, scoring, cluster)
{

}

void CudaVocabulary::transformData() const
{
    node_num = m_nodes.size();
    vector_row = m_nodes[0].descriptor.rows();
    //nonleaf_node_num = node_num - m_words.size();
    word_num = m_words.size();
    cu_k = m_k;

    cu_vocabulary = (cuNode*)malloc(sizeof(cuNode) * node_num);
    descriptor_map = (float*)malloc(sizeof(float) * node_num * vector_row);
    children_map = (int*)malloc(sizeof(int) * (node_num - word_num) * m_k);

    int children_index = 0;

    for (size_t i = 0; i < node_num; i++)
    {
        cu_vocabulary[i].id = i;
        cu_vocabulary[i].parent = m_nodes[i].parent;
        cu_vocabulary[i].weight = m_nodes[i].weight;
        cu_vocabulary[i].word_id = m_nodes[i].word_id;
 
        /*descriptor_map[i] = (float*)malloc(sizeof(float) * vector_row);
        for (uint32 j = 0; j < vector_row; j++)
        {
            descriptor_map[i][j] = m_nodes[i].descriptor[j];
        }*/
        
        for (uint32 j = 0; j < vector_row; j++)
        {
            descriptor_map[i * vector_row + j] = m_nodes[i].descriptor(j);
            //std::cout << m_nodes[i].descriptor(j) << " ";
        }

        size_t children_num = m_nodes[i].children.size();
        /// 非叶子节点
        if (children_num != 0)
        {
            cu_vocabulary[i].children_num = children_num;
            cu_vocabulary[i].children_index = children_index;
            //children_map[children_index] = (int*)malloc(sizeof(uint32) * m_k);
            int j = 0;
            while(j < children_num)
            {
                children_map[children_index * m_k + j] = m_nodes[i].children[j];
                j++;
            }
            while(j < m_k)
            {
                children_map[children_index * m_k + j] = -1;
                j++;
            }
            children_index++;
        }
        /// 叶子节点
        else
        {
            cu_vocabulary[i].children_num = 0;
            cu_vocabulary[i].children_index = -1;
        }

    }
}

void CudaVocabulary::create(
    const std::vector<std::vector<Eigen::VectorXf> > &training_features)
{
    m_nodes.clear();
    m_words.clear();
    /// 词典为完全m_k叉树
    int expected_nodes = (int)(pow((double)m_k, (double)m_d) - 1) / (m_k - 1);
    m_nodes.reserve(expected_nodes);

    std::vector<Eigen::VectorXf> features;
    for (size_t i = 0; i < training_features.size(); i++)
    {
        for (size_t j = 0; j < training_features[i].size(); j++)
        {
            features.push_back(training_features[i][j]);
        }
    }
    /// 建立第一个节点, 第一个节点为第0层
    Node root(0);
    root.descriptor = Eigen::MatrixXf::Zero(features[0].rows(), 1);
    m_nodes.push_back(root);
    /// 递归地建立词典
    createLayer(0, features, 1);
    /// 建立单词
    createWords();
    /// 设置IDF权重
    setNodeWeight(training_features);
    /// 转移数据
    transformData();
    int result = initVocabulary();
    if (result != 1) 
    {
        std::cout << "error in initlize vocabulary" << std::endl;
        exit(0);
    }
}

Eigen::VectorXf CudaVocabulary::cudaGetFeature(const cv::Mat &mat) const
{
    /*std::vector<Eigen::VectorXf> temp;
    for (uint32 j = 0; j < mat.rows; j++)
    {
        cv::Mat slice = mat.rowRange(j, j + 1);
        Eigen::RowVectorXf vec;
        cv::cv2eigen(slice, vec);
        temp.push_back(vec.transpose());
    }*/

    if (mat.channels() != 1) std::cout << "descriptor mat must have 1 channel" << std::endl;

    uint32 rows = mat.rows;
    uint32 cols = mat.cols;

    float* host_descriptors = new float[cols * rows];
    NULL_CHECK( host_descriptors )

    for (uint32 i = 0; i < rows; i++)
    {
        const uchar* head = mat.ptr<uchar>(i);
        for (uint32 j = 0; j < cols; j++)
        {
            host_descriptors[i * cols + j] = static_cast<float>(*(head + j));
        }
    }

    return cudaGetFeature(host_descriptors, rows, cols);
}

Eigen::VectorXf CudaVocabulary::cudaGetFeature(const std::vector<Eigen::VectorXf> &descriptors) const
{
    size_t rows, cols;
    rows = descriptors.size();
    if (rows == 0)
    {
        std::cerr << "empty descriptor" << std::endl;
        throw std::exception();
    }
    else 
    { 
        cols = descriptors[0].rows(); 
    }


    float* host_descriptors = new float[rows * cols];
    NULL_CHECK( host_descriptors )
    for (uint32 i = 0; i < rows; i++)
    {   
        for (uint32 j =0; j < cols; j++)
        {
            host_descriptors[cols * i + j] = descriptors[i](j); 
        }
    }

    return cudaGetFeature(host_descriptors, rows, cols);
}

Eigen::VectorXf CudaVocabulary::cudaGetFeature(float* host_descriptors, uint32 rows, uint32 cols) const
{
    //vector<float> feature;

    Eigen::VectorXf feature = Eigen::VectorXf::Zero(m_words.size());

    float* cuda_feature = cudaFindWord(host_descriptors, rows, cols);
    NULL_CHECK( cuda_feature )

    for (uint32 i = 0; i < m_words.size(); i++)
    {
        feature[i] = cuda_feature[i];
    }

    delete [] cuda_feature;
    cuda_feature = nullptr;

    delete [] host_descriptors;
    host_descriptors = nullptr;

    return feature;
}

Eigen::VectorXf CudaVocabulary::getFeature(const cv::Mat &mat) const
{
    std::vector<Eigen::VectorXf> temp;
    for (uint32 j = 0; j < mat.rows; j++)
    {
        cv::Mat slice = mat.rowRange(j, j + 1);
        Eigen::RowVectorXf vec;
        cv::cv2eigen(slice, vec);
        temp.push_back(vec.transpose());
    }
    return getFeature(temp);
}

Eigen::VectorXf CudaVocabulary::getFeature(const std::vector<Eigen::VectorXf> &descriptors) const
{
    Eigen::VectorXf feature = Eigen::VectorXf::Zero(m_words.size());
    for (auto descriptor : descriptors)
    {
        WordId word_id = findWord(descriptor);
        Node* p_node = getNodeWord(word_id);
        feature[word_id] += p_node->weight;
    }
    return feature;
}

CudaVocabulary::~CudaVocabulary()
{
    if (!freeVocabulary())
    {
        std::cout << "error in free vocabulary" << std::endl;
    }
    deleteData();
    delete m_cluster_object;
    m_cluster_object = nullptr;
}

} // namespace cuBoW

