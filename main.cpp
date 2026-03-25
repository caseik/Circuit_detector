#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include "app_config.h"
#include "pipeline.h"

std::string requireEnv(const char* key)
{
    const char* value = std::getenv(key);
    if (!value)
        throw std::runtime_error(std::string("Missing required env var: ") + key);
    return std::string(value);
}

int parseIntEnv(const char* key)
{
    return std::stoi(requireEnv(key));
}

double parseDoubleEnv(const char* key)
{
    return std::stod(requireEnv(key));
}

CliConfig initConfigFromEnv()
{
    CliConfig config;
    config.dataDir = requireEnv("DATA_DIR");
    config.startImage = parseIntEnv("START_IMAGE");
    config.endImage = parseIntEnv("END_IMAGE");
    config.resizeWidth = parseIntEnv("RESIZE_WIDTH");
    config.resizeHeight = parseIntEnv("RESIZE_HEIGHT");
    config.binaryThreshold = parseIntEnv("BINARY_THRESHOLD");
    config.closeKernel = parseIntEnv("CLOSE_KERNEL");
    config.dilateKernel = parseIntEnv("DILATE_KERNEL");
    config.minArea = parseIntEnv("MIN_AREA");
    config.margin = parseIntEnv("BOX_MARGIN");
    config.mergeIou = parseDoubleEnv("MERGE_IOU");
    return config;
}

void normalizeConfig(CliConfig& config)
{
    if (config.closeKernel % 2 == 0)
        config.closeKernel++;
    if (config.dilateKernel % 2 == 0)
        config.dilateKernel++;

    if (config.mergeIou < 0.0)
        config.mergeIou = 0.0;
    if (config.mergeIou > 1.0)
        config.mergeIou = 1.0;
}

int main()
{
    try
    {
        CliConfig config = initConfigFromEnv();
        normalizeConfig(config);
        runPipeline(config);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}