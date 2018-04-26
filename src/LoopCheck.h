#include "Type.h"
#include "cuVocabulary.h"

namespace cuBoW{

class DataSet;

class Scoring
{
public:
    static float score(const Eigen::SparseVector<float> &a, const Eigen::SparseVector<float> &b);
    static std::vector<float> group_score(const std::vector<Eigen::SparseVector<float> > &feature_group,
        const Eigen::SparseVector<float> &feature);
};
    
class DataSet
{
public:
    struct TuplePtr
    {
        const cudaTuple* pos;
        uint32 offset;
        TuplePtr(const cudaTuple* _pos, uint32 _offset):
            pos(_pos), offset(_offset) {}
        void copyData(cudaTuple* buffer) const
        {
            NULL_CHECK( buffer )
            memcpy(buffer, pos, offset * sizeof(float));
            buffer += offset;
        }
    };
    struct QueryResult
    {
        std::vector<TuplePtr> feature_ptrs;
        uint32 nnz;

    };
public:
    DataSet();
    void frameSetUpdate(const Eigen::SparseVector<float> &feature);
    void directIndexUpdate(const std::vector<WordId> &direct_index_row);
    QueryResult cudaQuery(FrameId id) const;
    QueryResult cudaGroupQuery(const std::vector<FrameId> &id_group) const;
    QueryResult cudaGroupQuery(const uint32 start, const uint32 stop, const uint32 stride = 1) const;
    Eigen::SparseVector<float> query(FrameId id) const;
    std::vector<Eigen::SparseVector<float>> groupQuery(const std::vector<FrameId> &id_group) const;
    uint32 getFeatureNum() const { return feature_set.size(); }
protected:
    /* inline */
    TuplePtr findPtr(const FrameId id) const
    {
        return TuplePtr(&feature_set[id][0], feature_set[id].size());
    }
    uint32 nnz;
    std::vector<std::vector<cudaTuple> > feature_set;
    std::vector<std::vector<WordId> > direct_index;
};

class LoopCheck
{
public:
    LoopCheck();
    //float score(const FrameId i, const FrameId j);
    //std::vector<float> score(const FrameId i, const std::vector<FrameId> &id_group);
    void addKeyFrame(const Eigen::SparseVector<float> &feature);
    std::vector<float> dataSetQuery(const Eigen::SparseVector<float> &feature);
protected:
    std::vector<float> score(const DataSet::QueryResult &res, 
        const Eigen::SparseVector<float> &feature) const;
    DataSet data_set;
};

}