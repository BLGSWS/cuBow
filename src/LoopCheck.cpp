#include "LoopCheck.h"

namespace cuBoW{

const uint32 GROUP_SIZE = 2048;
const uint32 CUDA_GROUP_SIZE = 256;

LoopCheck::LoopCheck() {}

void LoopCheck::addKeyFrame(const Eigen::SparseVector<float> &feature)
{
    data_set.frameSetUpdate(feature);
}

std::vector<float> LoopCheck::dataSetQuery(const Eigen::SparseVector<float> &feature)
{
    std::vector<float> scores;
    uint32 num = data_set.getFeatureNum(), i = 0;
    uint32 reverse = num % GROUP_SIZE;
    for (i; i < num - reverse; i += GROUP_SIZE)
    {
        DataSet::QueryResult query_result = data_set.cudaGroupQuery(i, i + GROUP_SIZE);
        std::vector<float> score_result = score(query_result, feature);
        scores.insert(scores.end(), score_result.begin(), score_result.end());
    }
    DataSet::QueryResult query_result = data_set.cudaGroupQuery(i, num);
    std::vector<float> score_result = score(query_result, feature);
    scores.insert(scores.end(), score_result.begin(), score_result.end());
    return scores;
}

std::vector<float> LoopCheck::score(const DataSet::QueryResult &res, 
    const Eigen::SparseVector<float> &feature) const
{
    cudaTuple* tuples = new cudaTuple[res.nnz + feature.nonZeros()];
    cudaTuple* p_tuple = tuples;
    uint32* csr_row = new uint32[res.feature_ptrs.size() + 2];
    for (Eigen::SparseVector<float>::InnerIterator it(feature); it; ++it)
    {
        p_tuple->id = it.index();
        p_tuple->value = it.value();
        ++p_tuple;
    }
    csr_row[0] = 0; csr_row[1] = feature.nonZeros();
    for (uint32 i = 0; i < res.feature_ptrs.size(); ++i)
    {
        res.feature_ptrs[i].copyData(p_tuple);
        csr_row[2 + i] = csr_row[1 + i] + res.feature_ptrs[i].offset;
    }
    std::vector<float> result =  cudaFeatureScore(tuples, res.feature_ptrs.size() + 1, 
        feature.size(), res.nnz, csr_row);
    delete [] tuples;
    return result;
}

float Scoring::score(const Eigen::SparseVector<float> &a, const Eigen::SparseVector<float> &b)
{
    uint32 nnz = a.nonZeros() + b.nonZeros();
    uint32* csrRow = new uint32[3];
    cudaTuple* feature_group = new cudaTuple[nnz];
    uint32 i = 0; 
    csrRow[0] = 0;
    for (Eigen::SparseVector<float>::InnerIterator it(a); it; ++it)
    {
        feature_group[i].id = it.index();
        feature_group[i].value = it.value(); 
        ++i;
    }
    csrRow[1] = i;
    for (Eigen::SparseVector<float>::InnerIterator it(b); it; ++it)
    {
        feature_group[i].id = it.index();
        feature_group[i].value = it.value(); 
        ++i;
    }
    csrRow[2] = i;

    std::vector<float> result =  cudaFeatureScore(feature_group, 2, a.size(), nnz, csrRow);

    delete [] csrRow;
    csrRow = nullptr;
    delete [] feature_group;
    feature_group = nullptr;

    return result[0];
}

std::vector<float> Scoring::group_score(const std::vector<Eigen::SparseVector<float> > &feature_group,
    const Eigen::SparseVector<float> &feature)
{
    uint32 nnz = feature.nonZeros();
    for (size_t i = 0; i < feature_group.size(); i++)
    {
        nnz += feature_group[i].nonZeros();
    }

    uint32* csrRow = new uint32[feature_group.size() + 2];
    uint32* p_csr = csrRow;
    cudaTuple* host_feature_group = new cudaTuple[nnz];
    uint32 i = 0; 
    *p_csr = 0;
    for (Eigen::SparseVector<float>::InnerIterator it(feature); it; ++it)
    {
        host_feature_group[i].id = it.index();
        host_feature_group[i].value = it.value(); 
        ++i;
    }
    *(++p_csr) = i;
    for (auto raw_feature : feature_group)
    {
        for (Eigen::SparseVector<float>::InnerIterator it(raw_feature); it; ++it)
        {
            host_feature_group[i].id = it.index();
            host_feature_group[i].value = it.value(); 
            ++i;
        }
        *(++p_csr) = i;
    }

    std::vector<float> result =  cudaFeatureScore(host_feature_group, feature_group.size() + 1,
        feature.size(), nnz, csrRow);
    
    delete [] csrRow;
    csrRow = nullptr; p_csr = nullptr;
    delete [] host_feature_group;
    host_feature_group = nullptr;

    return result;
}

DataSet::DataSet()
{

}

void DataSet::frameSetUpdate(const Eigen::SparseVector<float> &feature)
{
    std::vector<cudaTuple> tuple_set;
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

void DataSet::directIndexUpdate(const std::vector<WordId> &direct_index_row)
{
    direct_index.push_back(direct_index_row);
}

DataSet::QueryResult DataSet::cudaQuery(FrameId id) const
{
    QueryResult result;
    result.feature_ptrs.push_back(findPtr(id));
    result.nnz = result.feature_ptrs[0].offset;
    return result;
}

DataSet::QueryResult DataSet::cudaGroupQuery(const std::vector<FrameId> &id_group) const
{
    QueryResult result;
    result.feature_ptrs.reserve(id_group.size());
    std::vector<FrameId>::const_iterator it = id_group.begin();
    for (it; it != id_group.end(); ++it)
    {
        TuplePtr ptr = findPtr(*it);
        result.feature_ptrs.push_back(ptr);
        result.nnz += ptr.offset;
    }
    return result;
}

DataSet::QueryResult DataSet::cudaGroupQuery(const uint32 start, const uint32 stop, const uint32 stride) const
{
    QueryResult result;
    result.feature_ptrs.reserve((stop - start) / stride);
    for (uint32 i = start; i < stop; i += stride)
    {
        TuplePtr ptr = findPtr(i);
        result.feature_ptrs.push_back(ptr);
        result.nnz += ptr.offset;
    }
    return result;
}

Eigen::SparseVector<float> DataSet::query(FrameId id) const
{
    std::vector<std::vector<cudaTuple> >::const_iterator it = feature_set.begin() + id;
    Eigen::SparseVector<float> eigen_feature(it->size());
    std::vector<cudaTuple>::const_iterator tuple_it = it->begin();
    for (tuple_it; tuple_it != it->end(); ++tuple_it)
    {
        eigen_feature.insert(tuple_it->id) = tuple_it->id;
    }
    return eigen_feature;
}

std::vector<Eigen::SparseVector<float>> DataSet::groupQuery(const std::vector<FrameId> &id_group) const
{
    std::vector<Eigen::SparseVector<float> > eigen_feature_set;
    eigen_feature_set.reserve(id_group.size());
    std::vector<FrameId>::const_iterator it = id_group.begin();
    for (it; it != id_group.end(); ++it)
    {
        eigen_feature_set.push_back(query(*it));
    }
    return eigen_feature_set;
}

}