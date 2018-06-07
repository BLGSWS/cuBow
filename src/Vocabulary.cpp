#include "Vocabulary.h"
#include "cuVocabulary.h"
#include "Tools.hpp"
#include <fstream>

namespace cuBoW{

Vocabulary::Vocabulary(): m_cluster_object(nullptr)
{ }

Vocabulary::Vocabulary(
    int k, int L, WeightingType weighting,  ScoringType scoring, ClusterType cluster):
    m_k(k), m_d(L), m_weighting(weighting), m_scoring(scoring), m_cluster(cluster), 
    m_cluster_object(nullptr)
{
    createClusterObject();
}


void Vocabulary::createScoringObject()
{

}

void Vocabulary::createClusterObject()
{
    if (m_cluster_object != nullptr)
    {
        delete m_cluster_object;
        m_cluster_object = nullptr;
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

void Vocabulary::initGPU()
{
    transformData();
    initVocabulary();
    deleteData();
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
        // std::cout << final_id << " ";
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

WordId Vocabulary::DBfindWord(const Eigen::VectorXf &feature) const
{
    NodeId final_id = 0;
    while (!m_nodes[final_id].isLeaf())
    {
        //std::cout << "parent id: " << m_nodes[final_id].id << std::endl;
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
        //std::cout << "min_distance: " << min_distance << std::endl;
    }
    /// 返回wordid
    std::cout << "id: " << m_nodes[final_id].id << std::endl;
    return m_nodes[final_id].word_id;
}

void Vocabulary::save(const std::string &filepath) const
{
    std::ofstream ofile, onode, ochildren, odescriptor;
    ofile.open(filepath + "/paraments.txt", std::ios::out);
    onode.open(filepath + "/nodes", std::ios::binary);
    ochildren.open(filepath + "/children", std::ios::binary);
    odescriptor.open(filepath + "/descriptors", std::ios::binary);
    if (!ofile.is_open() || !onode.is_open() || !ochildren.is_open() || !odescriptor.is_open())
    {
        std::cerr << "Vocabulary::save: fail in open file: " << filepath;
        std::cerr << std::endl;
        throw std::exception();
    }

    /// 写入参数
    ofile << "#word_number" << std::endl << "word_num = " << m_words.size() << std::endl;
    ofile << "#node_number" << std::endl << "node_num = " << m_nodes.size() << std::endl;
    ofile << "#descriptor_demension" << std::endl << "vector_row = " << vector_row << std::endl;
    ofile << "#cluster_number_of_tree"  << std::endl << "m_k = " << m_k << std::endl;
    ofile << "#depth_of_tree" << std::endl << "m_d = " << m_d << std::endl;
    ofile << "#cluster_type" << std::endl << "m_cluster = " << m_cluster << std::endl;
    ofile << "#weighting_type" << std::endl << "m_weighting = " << m_weighting << std::endl;
    ofile << "#scoring_type" << std::endl << "m_scoring = " << m_scoring << std::endl;

    transformData();
    /// 写入二进制文件
    onode.write((char*)node_map, node_num * sizeof(cudaNode));
    ochildren.write((char*)children_map, (node_num - word_num) * m_k * sizeof(int));
    odescriptor.write((char*)descriptor_map, sizeof(float) * node_num * vector_row);

    ofile.close();
    onode.close();
    ochildren.close();
    odescriptor.close();

    deleteData();
}

void Vocabulary::read(const std::string &filepath)
{
    if (m_nodes.size() != 0)
    {
        std::cerr << "Vocabulary::read: there is data in vocabulary already" << std::endl;
        throw std::exception();
    }

    std::ifstream inode, ichildren, idescriptor;
    inode.open(filepath + "/nodes", std::ios::binary);
    ichildren.open(filepath + "/children", std::ios::binary);
    idescriptor.open(filepath + "/descriptors", std::ios::binary);
    if (!inode.is_open() || !ichildren.is_open() || !idescriptor.is_open())
    {
        std::cerr << "Vocabulary::read: fail to open the file: " << filepath;
        std::cerr << std::endl;
        throw std::exception();
    }

    Param_reader reader(filepath + "/paraments.txt");

    m_k = reader.get_param<uint32>("m_k");
    m_d = reader.get_param<uint32>("m_d");
    node_num = reader.get_param<uint32>("node_num");
    word_num = reader.get_param<uint32>("word_num");
    vector_row = reader.get_param<uint32>("vector_row");
    m_cluster = ClusterType(reader.get_param<uint32>("m_cluster"));
    m_weighting = WeightingType(reader.get_param<uint32>("m_weighting"));
    m_scoring = ScoringType(reader.get_param<uint32>("m_scoring"));

    createClusterObject();
    cu_k = m_k;

    if (node_map != nullptr || descriptor_map != nullptr || children_map != nullptr)
    {
        std::cerr << "Vocabulary::read: global ptr is not null" << std::endl;
        throw std::exception();
    }

    node_map = new cudaNode[node_num];
    children_map = new int[(node_num - word_num) * m_k];
    descriptor_map = new float[node_num * vector_row];
    
    inode.read((char*)node_map, node_num * sizeof(cudaNode));
    ichildren.read((char*)children_map, (node_num - word_num) * m_k * sizeof(int));
    idescriptor.read((char*)descriptor_map, node_num * vector_row * sizeof(float));

    m_nodes.reserve(node_num);
    m_words.reserve(word_num);
    cudaNode* p_vocabulary = node_map;
    float* p_descriptor = descriptor_map;

    for (int i = 0; i < node_num; i++)
    {
        Node node(i);
        node.id = i;
        node.parent = p_vocabulary->parent;
        node.weight = p_vocabulary->weight;
        node.word_id = p_vocabulary->word_id;

        std::vector<NodeId> children;
        if (p_vocabulary->children_index != -1)
        {
            children.reserve(p_vocabulary->children_num);
            for (int j = 0; j < p_vocabulary->children_num; j++)
            {
                children.push_back(children_map[p_vocabulary->children_index * cu_k + j]);
            }
        }
        node.children = children;

        /// 应该都用Map的
        Eigen::Map<Eigen::VectorXf> temp(p_descriptor, vector_row);
        Eigen::VectorXf descriptor = Eigen::VectorXf::Map(temp.data(), vector_row);
        node.descriptor = descriptor;

        m_nodes.push_back(node);

        if (node.isLeaf())
        {
            m_words.push_back(&m_nodes[i]);
        }

        ++p_vocabulary;
        p_descriptor += vector_row;
    }

    std::cout << "Read data from file " << filepath << " success!" << std::endl;
}

Vocabulary::~Vocabulary()
{
    //deleteData();
    delete m_cluster_object;
    m_cluster_object = nullptr;
}

void Vocabulary::transformData() const
{
    node_num = m_nodes.size();
    vector_row = m_nodes[0].descriptor.rows();
    //nonleaf_node_num = node_num - m_words.size();
    word_num = m_words.size();
    cu_k = m_k;

    node_map = (cudaNode*)malloc(sizeof(cudaNode) * node_num);
    descriptor_map = (float*)malloc(sizeof(float) * node_num * vector_row);
    children_map = (int*)malloc(sizeof(int) * (node_num - word_num) * m_k);

    cudaNode* p_node = node_map;
    float* p_descriptor = descriptor_map;
    int* p_child = children_map;
    uint32 children_index = 0;
    std::vector<Node>::const_iterator it = m_nodes.begin();

    for (it; it != m_nodes.end(); it++)
    {
        p_node->parent = it->parent;
        p_node->weight = it->weight;
        p_node->word_id = it->word_id;
 
        /*descriptor_map[i] = (float*)malloc(sizeof(float) * vector_row);
        for (uint32 j = 0; j < vector_row; j++)
        {
            descriptor_map[i][j] = m_nodes[i].descriptor[j];
        }*/
        
        for (uint32 j = 0; j < vector_row; j++)
        {
            *(p_descriptor++) = it->descriptor(j);
            //std::cout << m_nodes[i].descriptor(j) << " ";
        }

        size_t children_num = it->children.size();
        /// 非叶子节点
        if (children_num != 0)
        {
            p_node->children_num = children_num;
            p_node->children_index = children_index;
            //children_map[children_index] = (int*)malloc(sizeof(uint32) * m_k);
            int j = 0;
            while(j < children_num)
            {
                *(p_child++) = it->children[j];
                ++j;
            }
            while(j < m_k)
            {
                *(p_child++) = -1;
                ++j;
            }
            ++children_index;
        }
        /// 叶子节点
        else
        {
            p_node->children_num = 0;
            p_node->children_index = -1;
        }

        ++p_node;
    }
}

Eigen::SparseVector<float> Vocabulary::cudaGetFeature(const cv::Mat &mat) const
{
    uint32 rows = mat.rows;
    uint32 cols = mat.cols;

    float* host_descriptors = new float[cols * rows];
    NULL_CHECK( host_descriptors )

    if (mat.type() == CV_8UC1)
    {
        for (uint32 i = 0; i < rows; i++)
        {
            const uint8* head = mat.ptr<uint8>(i);
            for (uint32 j = 0; j < cols; j++)
            {
                host_descriptors[i * cols + j] = static_cast<float>(*(head + j));
            }
        }
    }
    else if (mat.type() == CV_32FC1)
    {
        float* p_descriptor = host_descriptors;
        for (uint32 i = 0; i < rows; i++)
        {
            const float* head = mat.ptr<float>(i);
            memcpy(p_descriptor, head, sizeof(float) * cols);
            p_descriptor += cols;
        }
    }
    else
    {
        /// https://blog.csdn.net/jeffdeen/article/details/52401526
        std::cerr << "Vocabulary::cudaGetFeature: mat type is " << mat.type() << std::endl;
        throw std::exception();
    }

    return cudaGetFeature(host_descriptors, rows, cols);
}

Eigen::SparseVector<float> Vocabulary::cudaGetFeature(const std::vector<Eigen::VectorXf> &descriptors) const
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

Eigen::SparseVector<float> Vocabulary::cudaGetFeature(float* host_descriptors, uint32 rows, uint32 cols) const
{
    //vector<float> feature;

    Eigen::SparseVector<float> sp_feature(m_words.size()); 

    std::vector<cudaTuple> cuda_features = cudaFindWord(host_descriptors, rows, cols);

    for (auto cuda_feature : cuda_features)
    {
        //sp_feature.insert(cuda_feature.id) = cuda_feature.value;
        sp_feature.coeffRef(cuda_feature.id) += cuda_feature.value;
    }

    delete [] host_descriptors;
    host_descriptors = nullptr;

    return sp_feature;
}

Eigen::SparseVector<float> Vocabulary::getFeature(const cv::Mat &mat) const
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

Eigen::SparseVector<float> Vocabulary::getFeature(const std::vector<Eigen::VectorXf> &descriptors) const
{
    Eigen::SparseVector<float> sp_feature(m_words.size());
    for (auto descriptor : descriptors)
    {
        WordId word_id = findWord(descriptor);
        Node* p_node = getNodeWord(word_id);
        //feature[word_id] += p_node->weight;
        sp_feature.coeffRef(word_id) += p_node->weight;
    }
    return sp_feature;
}

} // namespace cuBoW

