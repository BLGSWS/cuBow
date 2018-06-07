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

    for (uint32 i = 0; i < 100; i++)
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

    Vocabulary vocabulary(6, 4);
    vocabulary.create(features);
    vocabulary.save("tur_tree_surf_64");
}

int main()
{
    train();
}