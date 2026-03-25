#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

cv::Mat toBinary(const cv::Mat& img, int binaryThreshold);
cv::Mat keepLargestComponent(const cv::Mat& binary);
cv::Mat skeletonize(const cv::Mat& binary);

cv::Mat detectNonLinear(const cv::Mat& skeleton);
cv::Mat growRegions(const cv::Mat& img, int closeKernel, int dilateKernel);

std::vector<cv::Rect> extractRegions(
    const cv::Mat& img,
    int minArea,
    int margin,
    double mergeIouThreshold
);