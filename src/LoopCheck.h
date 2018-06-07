#include "Type.h"
#include "cuVocabulary.h"
#include <memory>

namespace cuBoW{

class DataSet;

class DataSet
{
public:
    DataSet();
    void insertFeature(const Eigen::SparseVector<float> &feature);
    //void directIndexUpdate(const std::vector<WordId> &direct_index_row);
    QueryResult query(FrameId id) const;
    QueryResult groupQuery(const std::vector<FrameId> &id_group) const;
    QueryResult groupQuery(const uint32 start, const uint32 stop, const uint32 stride = 1) const;
    Eigen::SparseVector<float> query2Eigen(FrameId id) const;
    std::vector<Eigen::SparseVector<float> > groupQuery2Eigen(const std::vector<FrameId> &id_group) const;
    std::vector<Eigen::SparseVector<float> > groupQuery2Eigen(const uint32 start, const uint32 stop, const uint32 stride = 1) const;
    uint32 getFeatureNum() const { return feature_set.size(); }
    uint32 getVectorSize() const { return vector_row; }
    const std::vector<cudaTuple>& getLastFrame() const { return *(feature_set.end() - 1); }
protected:
    /* inline */
    QueryResult::TuplePtr findPtr(const FrameId id) const
    {
        return QueryResult::TuplePtr(&feature_set[id][0], feature_set[id].size());
    }
    uint32 nnz;
    uint32 vector_row;
    std::vector<std::vector<cudaTuple> > feature_set;
    //std::vector<std::vector<WordId> > direct_index;
};

class LoopCheck
{
public:
    LoopCheck();
    LoopCheck(const std::shared_ptr<DataSet> _data_set);
    //float score(const FrameId i, const FrameId j);
    //std::vector<float> score(const FrameId i, const std::vector<FrameId> &id_group);
    void addKeyFrame(const Eigen::SparseVector<float> &feature);
    std::vector<float> dataSetQuery(const Eigen::SparseVector<float> &feature) const;
    std::vector<float> dataSetQuery_v2(const Eigen::SparseVector<float> &feature) const;
    const std::shared_ptr<DataSet> getData() const { return data_set; }
protected:
    inline std::vector<float> L1score(const QueryResult &res, 
        const Eigen::SparseVector<float> &feature) const;
    inline std::vector<float> L1score_v2(const QueryResult &res, 
        const Eigen::SparseVector<float> &feature) const;
    std::shared_ptr<DataSet> data_set;
};

class CPULoopCheck
{
public:
    CPULoopCheck();
    CPULoopCheck(const std::shared_ptr<DataSet> _data_set);
    void addKeyFrame(const Eigen::SparseVector<float> &feature);
    std::vector<float> dataSetQuery(const Eigen::SparseVector<float> &feature) const;
    std::vector<float> dataSetQuery_v2(const Eigen::SparseVector<float> &feature) const;
    DataSet* getData() const { return data_set.get(); }
    struct IndexItem
    {
        const cudaTuple* tuple_ptr;
        FrameId frame_id;
    };
protected:
    inline void updateInverseIndex();
    std::shared_ptr<DataSet> data_set;
    std::map<WordId, std::vector<IndexItem> > inverse_index;
};

}