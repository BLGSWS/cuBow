#include "LoopCheck.h"
#include <cmath>

namespace cuBoW{

const uint32 GROUP_SIZE = 256;
const uint32 CUDA_GROUP_SIZE = 1024;

LoopCheck::LoopCheck()
{
    data_set = std::make_shared<DataSet>();
}

LoopCheck::LoopCheck(const std::shared_ptr<DataSet> _data_set): 
data_set(_data_set)
{ }

void LoopCheck::addKeyFrame(const Eigen::SparseVector<float> &feature)
{
        data_set->insertFeature(feature);
}

std::vector<float> LoopCheck::dataSetQuery(const Eigen::SparseVector<float> &feature) const
{
    std::vector<float> scores;
    uint32 num = data_set->getFeatureNum(), i = 0;
    if (num == 0) return scores;
    uint32 reverse = num % GROUP_SIZE;
    for (i; i < num - reverse; i += GROUP_SIZE)
    {
        QueryResult query_result = data_set->groupQuery(i, i + GROUP_SIZE);
        std::vector<float> score_result = L1score(query_result, feature);
        scores.insert(scores.end(), score_result.begin(), score_result.end());
    }
    QueryResult query_result = data_set->groupQuery(num - reverse, num);
    std::vector<float> score_result = L1score(query_result, feature);
    scores.insert(scores.end(), score_result.begin(), score_result.end());
    return scores;
}

std::vector<float> LoopCheck::dataSetQuery_v2(const Eigen::SparseVector<float> &feature) const
{
    std::vector<float> scores;
    uint32 num = data_set->getFeatureNum(), i = 0;
    if (num == 0) return scores;
    uint32 reverse = num % GROUP_SIZE;
    for (i; i < num - reverse; i += GROUP_SIZE)
    {
        QueryResult query_result = data_set->groupQuery(i, i + GROUP_SIZE);
        std::vector<float> score_result = L1score_v2(query_result, feature);
        scores.insert(scores.end(), score_result.begin(), score_result.end());
    }
    QueryResult query_result = data_set->groupQuery(num - reverse, num);
    std::vector<float> score_result = L1score_v2(query_result, feature);
    scores.insert(scores.end(), score_result.begin(), score_result.end());
    return scores;
}

inline std::vector<float> LoopCheck::L1score(const QueryResult &res, 
    const Eigen::SparseVector<float> &feature) const
{
    std::unique_ptr<cudaTuple> tuples(new cudaTuple[res.nnz + feature.nonZeros()]);
    cudaTuple* p_tuple = tuples.get();
    std::unique_ptr<uint32> csr_row(new uint32[res.feature_ptrs.size() + 2]);
    for (Eigen::SparseVector<float>::InnerIterator it(feature); it; ++it)
    {
        p_tuple->id = it.index();
        p_tuple->value = it.value();
        ++p_tuple;
    }
    csr_row.get()[0] = 0; csr_row.get()[1] = feature.nonZeros();
    for (uint32 i = 0; i < res.feature_ptrs.size(); ++i)
    {
        res.feature_ptrs[i].copyData(p_tuple);
        csr_row.get()[2 + i] = csr_row.get()[1 + i] + res.feature_ptrs[i].offset;
        p_tuple += res.feature_ptrs[i].offset;
    }
    std::vector<float> result =  cudaFeatureScore(tuples.get(), res.feature_ptrs.size() + 1, 
        feature.size(), res.nnz + feature.nonZeros(), csr_row.get());
    return result;
}

inline std::vector<float> LoopCheck::L1score_v2(const QueryResult &res, 
    const Eigen::SparseVector<float> &feature) const
{
    std::unique_ptr<cudaTuple> tuples(new cudaTuple[res.nnz + feature.nonZeros()]);
    cudaTuple* p_tuple = tuples.get();
    std::unique_ptr<uint32> csr_row(new uint32[res.feature_ptrs.size() + 2]);
    for (Eigen::SparseVector<float>::InnerIterator it(feature); it; ++it)
    {
        p_tuple->id = it.index();
        p_tuple->value = it.value();
        ++p_tuple;
    }
    csr_row.get()[0] = 0; csr_row.get()[1] = feature.nonZeros();
    for (uint32 i = 0; i < res.feature_ptrs.size(); ++i)
    {
        res.feature_ptrs[i].copyData(p_tuple);
        csr_row.get()[2 + i] = csr_row.get()[1 + i] + res.feature_ptrs[i].offset;
        p_tuple += res.feature_ptrs[i].offset;
    }
    std::vector<float> result =  cudaFeatureScore_v2(tuples.get(), res.feature_ptrs.size() + 1, 
        feature.size(), res.nnz + feature.nonZeros(), csr_row.get());
    return result;
}

DataSet::DataSet()
{

}

void DataSet::insertFeature(const Eigen::SparseVector<float> &feature)
{
    std::vector<cudaTuple> tuple_set;
    vector_row = feature.size();
    tuple_set.reserve(feature.nonZeros());
    Eigen::SparseVector<float>::InnerIterator it(feature);
    for (it; it; ++it)
    {
        cudaTuple tuple;
        tuple.id = it.index();
        tuple.value = it.value();
        tuple_set.push_back(tuple);
    }
    feature_set.push_back(tuple_set);
}

/*void DataSet::directIndexUpdate(const std::vector<WordId> &direct_index_row)
{
    direct_index.push_back(direct_index_row);
}*/

QueryResult DataSet::query(FrameId id) const
{
    QueryResult result;
    result.feature_ptrs.push_back(findPtr(id));
    result.nnz = result.feature_ptrs[0].offset;
    return result;
}

QueryResult DataSet::groupQuery(const std::vector<FrameId> &id_group) const
{
    QueryResult result;
    result.feature_ptrs.reserve(id_group.size());
    result.nnz = 0;
    for (uint32 i = 0; i < id_group.size(); i++)
    {
        QueryResult::TuplePtr ptr = findPtr(id_group[i]);
        result.feature_ptrs.push_back(ptr);
        result.nnz += ptr.offset;
    }
    return result;
}

QueryResult DataSet::groupQuery(const uint32 start, const uint32 stop, const uint32 stride) const
{
    QueryResult result;
    result.feature_ptrs.reserve((stop - start) / stride);
    result.nnz = 0; /// ?
    for (uint32 i = start; i < stop; i += stride)
    {
        QueryResult::TuplePtr ptr = findPtr(i);
        result.feature_ptrs.push_back(ptr);
        result.nnz += ptr.offset;
    }
    return result;
}

Eigen::SparseVector<float> DataSet::query2Eigen(FrameId id) const
{
    std::vector<std::vector<cudaTuple> >::const_iterator it = feature_set.begin() + id;
    Eigen::SparseVector<float> eigen_feature(vector_row);
    std::vector<cudaTuple>::const_iterator tuple_it = it->begin();
    for (tuple_it; tuple_it != it->end(); ++tuple_it)
    {
        eigen_feature.insert(tuple_it->id) = tuple_it->value;
    }
    return eigen_feature;
}

std::vector<Eigen::SparseVector<float>> DataSet::groupQuery2Eigen(const std::vector<FrameId> &id_group) const
{
    std::vector<Eigen::SparseVector<float> > eigen_feature_set;
    eigen_feature_set.reserve(id_group.size());
    std::vector<FrameId>::const_iterator it = id_group.begin();
    for (it; it != id_group.end(); ++it)
    {
        eigen_feature_set.push_back(query2Eigen(*it));
    }
    return eigen_feature_set;
}

CPULoopCheck::CPULoopCheck()
{
    data_set = std::make_shared<DataSet>();
}

CPULoopCheck::CPULoopCheck(const std::shared_ptr<DataSet> _data_set): 
data_set(_data_set)
{ }

void CPULoopCheck::addKeyFrame(const Eigen::SparseVector<float> &feature)
{
    data_set->insertFeature(feature);
    updateInverseIndex();
}

std::vector<float> CPULoopCheck::dataSetQuery(const Eigen::SparseVector<float> &feature) const
{
    std::vector<float> scores;
    uint32 num = data_set->getFeatureNum();
    scores.reserve(num);
    for (FrameId i = 0; i < num; i++)
    {
        Eigen::SparseVector<float> key_frame_feature = data_set->query2Eigen(i);
        float norm1 = feature.norm();
        float norm2 = key_frame_feature.norm();
        float distance = (feature / norm1 - key_frame_feature / norm2).norm();
        scores.push_back(1. - 0.5 * distance);
    }
    return scores;
}

std::vector<float> CPULoopCheck::dataSetQuery_v2(const Eigen::SparseVector<float> &feature) const
{
    /*for (auto map_it = inverse_index.begin(); map_it != inverse_index.end(); map_it++)
    {
        std::cout << map_it->first << std::endl;
        auto vec = map_it->second;
        for (uint32 i = 0; i < vec.size(); i++)
        {
            std::cout << vec[i].frame_id << " "<< vec[i].tuple_ptr->id << " " << vec[i].tuple_ptr->value << std::endl; 
        }
    }*/
    std::vector<float> scores(data_set->getFeatureNum(), 0.);
    for (Eigen::SparseVector<float>::InnerIterator it(feature); it; ++it)
    {
        if (inverse_index.count(it.index()) == 0) continue;
        auto map_it = inverse_index.find(it.index());
        for (auto index_item: map_it->second)
        {
            float absolute = fabs(fabs(it.value() - index_item.tuple_ptr->value) - it.value() - index_item.tuple_ptr->value);
            scores[index_item.frame_id - 1] += absolute;
        }
    }
    return scores;
}

inline void CPULoopCheck::updateInverseIndex()
{
    const std::vector<cudaTuple> &last_frame = data_set->getLastFrame();
    for (uint32 i = 0; i < last_frame.size(); i++)
    {
        IndexItem item;
        item.tuple_ptr = &last_frame[i];
        item.frame_id = data_set->getFeatureNum();
        inverse_index[last_frame[i].id].push_back(item);
    }
}

}