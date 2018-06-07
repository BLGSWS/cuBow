#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <time.h>
#include "Vocabulary.h"
#include "LoopCheck.h"

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
    cv::Mat rgb1 = cv::imread( "./test_image/rgb/1305031536.875419.png");
    cv::Mat rgb2 = cv::imread( "./test_image/rgb/1305031533.575643.png");

    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor;
    cv::initModule_nonfree();
    detector = cv::FeatureDetector::create( "SURF" );
    descriptor = cv::DescriptorExtractor::create( "SURF" );

    vector< cv::KeyPoint > kp1, kp2;
    detector->detect( rgb1, kp1 );
    detector->detect( rgb2, kp2 );

    cv::Mat imgShow1, imgShow2;
    cv::drawKeypoints( rgb1, kp1, imgShow1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    cv::drawKeypoints( rgb2, kp2, imgShow2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    //cv::imshow( "keypoints", imgShow );
    cv::imwrite( "./img1.png", rgb1 );
    cv::imwrite( "./img2.png", rgb2 );
    cv::imwrite( "./keypoints1.png", imgShow1 );
    cv::imwrite( "./keypoints2.png", imgShow2 );

    Vocabulary vocabulary;
    vocabulary.read("tur_tree_surf_84");
    vocabulary.initGPU();

    cv::Mat feature1, feature2;
    descriptor->compute( rgb1, kp1, feature1 );
    descriptor->compute( rgb2, kp2, feature2 );

    vector< cv::DMatch > matches; 
    cv::FlannBasedMatcher matcher;
    matcher.match( feature1, feature2, matches );

    cv::Mat imgMatches;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matches, imgMatches );
    cv::imwrite( "./matches.png", imgMatches );

    Eigen::SparseVector<float> result1 = vocabulary.cudaGetFeature(feature1);
    Eigen::SparseVector<float> result2 = vocabulary.cudaGetFeature(feature2);

    ofstream fo1, fo2;
    fo1.open("bowvec1.txt", ios::out | ios::app);
    fo2.open("bowvec2.txt", ios::out | ios::app);
    for (Eigen::SparseVector<float>::InnerIterator it(result1); it; ++it)
    {
        fo1 << it.index() << "," << it.value() << endl;
    }
    for (Eigen::SparseVector<float>::InnerIterator it(result2); it; ++it)
    {
        fo2 << it.index() << "," << it.value() << endl;
    }
    fo1.close();
    fo2.close();
}