#pragma once

#include <string>

struct CliConfig
{
    std::string dataDir = "data";
    int startImage = 11;
    int endImage = 20;
    int resizeWidth = 323;
    int resizeHeight = 172;
    int binaryThreshold = 20;
    int closeKernel = 19;
    int dilateKernel = 11;
    int minArea = 180;
    int margin = 7;
    double mergeIou = 0.30;
};
