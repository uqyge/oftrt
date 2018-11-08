#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"

#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;
#include "common.h"

// static Logger gLogger;
// static int gDLA{0};

std::string locateFile(const std::string &input);

ICudaEngine *loadModelAndCreateEngine(const char *uffFile,
                                      int maxBatchSize,
                                      IUffParser *parser);

void execute(ICudaEngine &engine);