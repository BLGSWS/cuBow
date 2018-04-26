#include<iostream>
#include "LoopCheck.h"
using namespace std;
using namespace cuBoW;

int main()
{
    /*
    1 0 0 3 0 0
    0 7 0 5 0 0
    0 2 4 0 1 0
    0 1 2 0 0 0
    */
    Eigen::SparseVector<float> v1(6), v2(6), v3(6), v4(6);
    v1.insert(0) = 1; v1.insert(3) = 3;
    v2.insert(1) = 7; v2.insert(3) = 5;
    v3.insert(1) = 2; v3.insert(2) = 4; v3.insert(4) = 1;
    v4.insert(1) = 1; v4.insert(2) = 2;
    LoopCheck check;
    check.addKeyFrame(v2);
    check.addKeyFrame(v3);
    check.addKeyFrame(v4);
    vector<float> result = check.dataSetQuery(v1);
    for (uint32 i = 0; i < result.size(); ++i)
    {
        cout << result[i] << " ";
    }
    cout << endl;
    //cudaFeatureScoreTest();
}