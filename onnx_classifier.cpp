#include "onnx_classifier.h"

#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <vector>

namespace
{
cv::dnn::Net g_net;
std::vector<std::string> g_labels;
bool g_ready = false;
cv::HOGDescriptor g_hog(
    cv::Size(64, 64),
    cv::Size(16, 16),
    cv::Size(8, 8),
    cv::Size(8, 8),
    9
);

cv::Mat preprocessRoi(const cv::Mat& roi)
{
    cv::Mat gray;
    if (roi.channels() == 3)
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    else
        gray = roi.clone();

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(64, 64));
    cv::Mat binary;
    cv::threshold(
        resized,
        binary,
        0,
        255,
        cv::THRESH_BINARY_INV | cv::THRESH_OTSU
    );
 
    return binary;
}

cv::Mat extractFeatures(const cv::Mat& preprocessed)
{
    std::vector<float> descriptors;
    g_hog.compute(preprocessed, descriptors);
    cv::Mat features(1, static_cast<int>(descriptors.size()), CV_32F, descriptors.data());
    return features.clone();
}
} // namespace

bool initOnnxClassifier(const std::string& modelPath, const std::string& labelsPath)
{
    try
    {
        g_net = cv::dnn::readNetFromONNX(modelPath);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to load ONNX model: " << e.what() << std::endl;
        g_ready = false;
        return false;
    }

    g_labels.clear();
    std::ifstream labelsFile(labelsPath);
    if (!labelsFile.is_open())
    {
        std::cerr << "Failed to load labels file: " << labelsPath << std::endl;
        g_ready = false;
        return false;
    }

    std::string line;
    while (std::getline(labelsFile, line))
    {
        if (!line.empty())
            g_labels.push_back(line);
    }

    if (g_labels.empty())
    {
        std::cerr << "Labels file is empty: " << labelsPath << std::endl;
        g_ready = false;
        return false;
    }

    g_ready = true;
    return true;
}

std::string classifyOnnx(const cv::Mat& roi)
{
    if (!g_ready)
        return "error";

    const cv::Mat processed = preprocessRoi(roi);
    const cv::Mat features = extractFeatures(processed);

    g_net.setInput(features);

    std::vector<cv::String> outNames = g_net.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outs;
    g_net.forward(outs, outNames);

    cv::Mat scores;
    for (const auto& out : outs)
    {
        if (out.total() == static_cast<size_t>(g_labels.size()))
        {
            scores = out.reshape(1, 1);
            break;
        }
    }


    cv::Point classIdPoint;
    double maxScore = 0.0;
    cv::minMaxLoc(scores, nullptr, &maxScore, nullptr, &classIdPoint);

    const int classId = classIdPoint.x;
    if (classId < 0 || classId >= static_cast<int>(g_labels.size()))
        return "error";

    return g_labels[classId];
}
