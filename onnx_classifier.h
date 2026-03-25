#pragma once

#include <opencv2/opencv.hpp>
#include <string>

bool initOnnxClassifier(
    const std::string& modelPath = "model.onnx",
    const std::string& labelsPath = "labels.txt"
);

std::string classifyOnnx(const cv::Mat& roi);
