#include<iostream>
#include<memory>
#include<math.h>
#include "LoopCheck.h"
using namespace std;
using namespace cuBoW;

int main()
{
    /*
    1 0 0 3 0 0
    0 7 0 5 0 1
    0 2 4 0 1 0
    0 1 2 1 0 0
    */
    Eigen::SparseVector<float> v1(6), v2(6), v3(6), v4(6);
    v1.insert(0) = 1; v1.insert(3) = 3;
    v2.insert(1) = 7; v2.insert(3) = 5; v2.insert(5) = 1;
    v3.insert(1) = 2; v3.insert(2) = 4; v3.insert(4) = 1;
    v4.insert(1) = 1; v4.insert(2) = 2; v4.insert(3) = 1;
    shared_ptr<DataSet> data_set = make_shared<DataSet>();
    LoopCheck check(data_set);
    CPULoopCheck cpu_check(data_set);
    cpu_check.addKeyFrame(v2);
    cpu_check.addKeyFrame(v3);
    cpu_check.addKeyFrame(v4);
    vector<float> result = check.dataSetQuery_v2(v1);
    vector<float> cpu_result = cpu_check.dataSetQuery_v2(v1);
    for (uint32 i = 0; i < result.size(); ++i)
    {
        cout << result[i] << " ";
    }
    cout << endl;
    for (uint32 i = 0; i < result.size(); i++)
    {
        cout << cpu_result[i] << " ";
    }
    cout << endl;
    cout << 1. - 0.5 * sqrt(49. / 74. + 1. / 10. + (5. / sqrt(74.) - 3. / sqrt(10.)) * (5. / sqrt(74.) - 3. / sqrt(10.)));
    cout << endl;
    //cudaFeatureScoreTest();
}