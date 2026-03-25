#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct Detection
{
    cv::Rect box;
    std::string label;
};

std::string classify(const cv::Mat& roi);

void drawCandidates(
    cv::Mat& img,
    const std::vector<Detection>& detections);

std::vector<Detection> classifyRegions(
    const cv::Mat& clean,
    const std::vector<cv::Rect>& boxes);