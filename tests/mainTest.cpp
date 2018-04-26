//#define IRIS
#define IMAGE

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <time.h>
#include "Vocabulary.h"

/// opencv for feature detection
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cuBoW;
using namespace Eigen;

int main()
{
#ifdef IMAGE
    vector<cv::KeyPoint> key_points;
    vector<cv::Mat> features;
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor;
    cv::initModule_nonfree();
    detector = cv::FeatureDetector::create("SURF");
    descriptor = cv::DescriptorExtractor::create("SURF");

    for (uint32 i = 0; i < 4; i++)
    {
        cv::Mat feature;
        string image_path = "example_image/image" + to_string(i) + ".png";
        cv::Mat image = cv::imread(image_path, -1);
        if (image.empty())
        {
            cerr << "error in read image: " << image_path << endl;
            throw exception();
        }
        detector->detect(image, key_points);
        descriptor->compute(image, key_points, feature);
        features.push_back(feature);
    }

    Vocabulary vocabulary(5, 4);
    //vocabulary.create(features);
    vocabulary.read("tree");

    clock_t stage1, stage2, stage3;

    cv::Mat feature;

    string image_path = "example_image/image0.png";
    cv::Mat image = cv::imread(image_path, -1);
    if (image.empty())
    {
        cerr << "error in read image: " << image_path << endl;
        throw exception();
    }

    detector->detect(image, key_points);
    descriptor->compute(image, key_points, feature);

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


#endif 
    return 0;
}