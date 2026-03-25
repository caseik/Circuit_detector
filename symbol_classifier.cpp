#include "symbol_classifier.h"
#include "onnx_classifier.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

void drawCandidates(cv::Mat& img,
                    const std::vector<Detection>& detections)
{
    for(const auto& det : detections)
    {
        cv::rectangle(img, det.box, cv::Scalar(0,0,255), 2);

        cv::putText(
            img,
            det.label,
            cv::Point(det.box.x, det.box.y + 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0,255,0),
            2
        );
    }
}

std::vector<Detection> classifyRegions(
    const cv::Mat& clean,
    const std::vector<cv::Rect>& boxes)
{
    std::vector<Detection> results;

    for(const auto& box : boxes)
    {
        cv::Mat roi = clean(box);

        std::string label = classify(roi);

        results.push_back({box, label});
    }

    return results;
}


std::string classify(const cv::Mat& roi)
{
    static bool initialized = false;
    static bool initOk = false;
    if (!initialized)
    {
        initOk = initOnnxClassifier("model.onnx", "labels.txt");
        initialized = true;
    }

    if (!initOk)
        return "error";
    return classifyOnnx(roi);
}