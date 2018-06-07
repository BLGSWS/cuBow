#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <time.h>
#include "Vocabulary.h"
#include "LoopCheck.h"

/// opencv for feature detection
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cuBoW;
using namespace Eigen;

cv::Ptr<cv::FeatureDetector> detector;
cv::Ptr<cv::DescriptorExtractor> descriptor;
vector<cv::KeyPoint> key_points;

vector<string> readImageName(const string &txtfilepath, uint32 num = UINT32_MAX)
{
    vector<string> images;

    ifstream fin;
    fin.open(txtfilepath, std::ios::in);
    if (!fin)
    {
        cerr << "error in read file" << endl; 
        throw exception();
    }

    uint32 count = 0;
    while (!fin.eof())
    {
        string str;
        getline(fin, str);
        if (str[0] == '#') continue;
        
        int pos = str.find(" ");
        if (pos == -1) continue;

        string image_name = str.substr(pos + 1, str.length());
        images.push_back(image_name);

        count++;
        if (!fin.good() || count > num) break;
    }

    return images;
}

void train()
{
    vector<cv::Mat> features;
    cv::initModule_nonfree();
    detector = cv::FeatureDetector::create("SURF");
    descriptor = cv::DescriptorExtractor::create("SURF");

    vector<string> images = readImageName("train_image/rgb.txt");

    for (uint32 i = 0; i < images.size(); i++)
    {
        cv::Mat image = cv::imread("train_image/" + images[i], -1);
        cv::Mat feature;
        vector<cv::KeyPoint> key_points;

        if (image.empty())
        {
            cerr << "error in read image: " << "train_image/" + images[i] << endl;
            throw exception();
        }

        detector->detect(image, key_points);
        descriptor->compute(image, key_points, feature);
        features.push_back(feature);
    }

    Vocabulary vocabulary(8, 6);
    vocabulary.create(features);
    vocabulary.save("tur_tree");
}

cv::Mat detect_feature(const string &path)
{
    cv::initModule_nonfree();
    detector = cv::FeatureDetector::create("SURF");
    descriptor = cv::DescriptorExtractor::create("SURF");

    cv::Mat feature;
    cv::Mat image = cv::imread(path, -1);
    if (image.empty())
    {
        cerr << "error in read image: " << path << endl;
        throw exception();
    }

    detector->detect(image, key_points);
    descriptor->compute(image, key_points, feature);

    return feature;
}

int main()
{
    //train();

    Vocabulary vocabulary;
    vocabulary.read("tur_tree_surf");
    vocabulary.initGPU();

    vector<string> images = readImageName("test_image/rgb.txt", 1000);

    shared_ptr<DataSet> data_set = make_shared<DataSet>();
    LoopCheck check(data_set);
    CPULoopCheck cpu_check(data_set);
    /*cv::Mat feature = detect_feature("test_image/" + images[0]);
    Eigen::SparseVector<float> result = vocabulary.cudaGetFeature(feature);
    data_set->insertFeature(result);
    vector<float> scores1 = check.dataSetQuery(result);*/

    ofstream fo;
    fo.open("total_time.txt", ios::out | ios::app);
    if (!fo.is_open())
    {
        cout << "open file fail" << endl;
    }

    for (uint32 i = 1; i < images.size(); i++)
    {
        clock_t stage1, stage2, stage3, stage4, stage5;
        cv::Mat feature = detect_feature("test_image/" + images[i]);

        stage1 = clock();
        Eigen::SparseVector<float> result = vocabulary.cudaGetFeature(feature);
        vector<float> scores1 = check.dataSetQuery_v2(result);
        stage2 = clock();
        Eigen::SparseVector<float> result2 = vocabulary.getFeature(feature);
        vector<float> scores2 = cpu_check.dataSetQuery_v2(result2);
        stage3 = clock();
        Eigen::SparseVector<float> result3 = vocabulary.cudaGetFeature(feature);
        vector<float> scores3 = check.dataSetQuery(result3);
        stage4 = clock();
        Eigen::SparseVector<float> result4 = vocabulary.getFeature(feature);
        vector<float> scores4 = cpu_check.dataSetQuery(result4);
        stage5 = clock();

        cpu_check.addKeyFrame(result4);
        /*
        stage1 = clock();
        vector<float> scores1 = check.dataSetQuery_v2(result);
        stage2 = clock();
        vector<float> scores2 = cpu_check.dataSetQuery_v2(result);
        stage3 = clock();

        cpu_check.addKeyFrame(result);
        */

        float elapsed_time1 = (float)(stage2 - stage1) / CLOCKS_PER_SEC;
        float elapsed_time2 = (float)(stage3 - stage2) / CLOCKS_PER_SEC;
        float elapsed_time3 = (float)(stage4 - stage3) / CLOCKS_PER_SEC;
        float elapsed_time4 = (float)(stage5 - stage4) / CLOCKS_PER_SEC;

        /*cout << "scores" << i << ": ";
        for (uint32 j = 0; j < scores1.size(); j++)
        {
            cout << scores1[j] << ": " << scores2[j] << ", ";
        }
        cout << endl;*/

        cout << "GPU: " << elapsed_time1 << ", CPU: " << elapsed_time2 << endl;
        fo << elapsed_time1 << "," << elapsed_time2 << "," << elapsed_time3 << "," << elapsed_time4 << endl;

        //Eigen::SparseVector<float> result2 = vocabulary.getFeature(feature);
        
    }
    fo.close();

    /*clock_t stage1, stage2, stage3;

    //const vector<Vocabulary::Node*>& words = vocabulary.getWords();
    //VectorXf vec = words[5]->descriptor;

    stage1 = clock();

    Eigen::SparseVector<float> result1 = vocabulary.cudaGetFeature(feature);

    stage2 = clock();
    
    Eigen::SparseVector<float> result2 = vocabulary.getFeature(feature);

    stage3 = clock();

    float elapsed_time1 = (float)(stage2 - stage1) / CLOCKS_PER_SEC;
    float elapsed_time2 = (float)(stage3 - stage2) / CLOCKS_PER_SEC;

    cout << result1 << endl << endl;
    cout << result2 << endl;

    cout << elapsed_time1 << ": " << elapsed_time2 << endl;
    */

    return 0;
}