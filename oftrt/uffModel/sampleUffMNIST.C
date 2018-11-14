#include "sampleUffMNIST.H"

using namespace nvuffparser;
using namespace nvinfer1;

static Logger gLogger;
static int gDLA{0};

#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                   \
    do                                                                           \
    {                                                                            \
        std::string error_message = "sample_uff_mnist: " + std::string(message); \
        gLogger.log(ILogger::Severity::k##severity, error_message.c_str());      \
        return (ret);                                                            \
    } while (0)

inline int64_t volume(const Dims &d)
{
    int64_t v = 1;
    for (int64_t i = 0; i < d.nbDims; i++)
        v *= d.d[i];
    return v;
}

inline unsigned int elementSize(DataType t)
{
    switch (t)
    {
    case DataType::kINT32:
        // Fallthrough, same as kFLOAT
    case DataType::kFLOAT:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT8:
        return 1;
    }
    assert(0);
    return 0;
}

static const int INPUT_H = 2;
static const int INPUT_W = 1;
static const int OUTPUT_SIZE = 10;

std::string locateFile(const std::string &input)
{
    std::vector<std::string> dirs{"data/mnist/", "data/samples/mnist/", "./data/"};
    return locateFile(input, dirs);
}

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string &filename, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(locateFile(filename), buffer, INPUT_H, INPUT_W);
}

void *safeCudaMalloc(size_t memSize)
{
    void *deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine &engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

void *createMnistCudaBuffer(int64_t eltCount, DataType dtype, int run)
{
    /* in that specific case, eltCount == INPUT_H * INPUT_W */
    assert(eltCount == INPUT_H * INPUT_W);
    assert(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    float *inputs = new float[eltCount];

    /* read PGM file */
    uint8_t fileData[INPUT_H * INPUT_W];
    readPGMFile(std::to_string(run) + ".pgm", fileData);

    /* display the number in an ascii representation */
    std::cout << "\n\n\n---------------------------"
              << "\n\n\n"
              << std::endl;
    for (int i = 0; i < eltCount; i++)
        std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

    /* initialize the inputs buffer */
    for (int i = 0; i < eltCount; i++)
        inputs[i] = 1.0 - float(fileData[i]) / 255.0;

    void *deviceMem = safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));

    delete[] inputs;
    return deviceMem;
}

void *createRealCudaBuffer(float *input_p_he, int64_t eltCount, DataType dtype, int run)
{
    /* in that specific case, eltCount == INPUT_H * INPUT_W */
    std::cout << eltCount << '\n';
    // std::cout << input_p_he[0] << '\n'
    //           << input_p_he[1] << '\n';
    //   << input_p_he[2] << '\n'
    //   << input_p_he[3] << '\n';
    // assert(eltCount == INPUT_H * INPUT_W);
    assert(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    float *inputs = new float[eltCount];

    for (int i = 0; i < eltCount; i++)
    {
        inputs[i] = input_p_he[i];
        std::cout << inputs[i] << '\n';
    }
    // inputs = input_p_he;

    void *deviceMem = safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));

    delete[] inputs;
    return deviceMem;
}

void printOutput(int64_t eltCount, DataType dtype, void *buffer, float out_arr[])
{
    // std::cout << eltCount << " eltCount" << std::endl;
    assert(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    float *outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(out_arr, buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = 0;
    for (int i = 0; i < eltCount; ++i)
        if (outputs[i] > outputs[maxIdx])
            maxIdx = i;

    std::cout << std::endl;
    delete[] outputs;
}

ICudaEngine *loadModelAndCreateEngine(const char *uffFile, int maxBatchSize,
                                      IUffParser *parser)
{
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetwork();

#if 1
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
#else
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
    builder->setFp16Mode(true);
#endif
    // std::cout << "network:" << network->getInput() << '\n';
    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);
    if (gDLA > 0)
        samplesCommon::enableDLA(builder, gDLA);

    ICudaEngine *engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}

void execute(ICudaEngine &engine, int batchSize, std::vector<float> &input_p_he, std::vector<float> &output_real)
{
    IExecutionContext *context = engine.createExecutionContext();
    // int batchSize = 1;

    int nbBindings = engine.getNbBindings();
    assert(nbBindings == 2);

    std::vector<void *> buffers(nbBindings);
    auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    int bindingIdxInput = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        if (engine.bindingIsInput(i))
            bindingIdxInput = i;
        else
        {
            auto bufferSizesOutput = buffersSizes[i];
            buffers[i] = safeCudaMalloc(bufferSizesOutput.first *
                                        elementSize(bufferSizesOutput.second));
        }
    }

    auto bufferSizesInput = buffersSizes[bindingIdxInput];

    int iterations = 1;
    int numberRun = 2;
    numberRun = input_p_he.size() / bufferSizesInput.first;
    std::cout << "rounds:" << numberRun << '\n';
    float input_batch[batchSize * INPUT_H * INPUT_W];
    for (int i = 0; i < iterations; i++)
    {
        float total = 0, ms;
        for (int run = 0; run < numberRun; run++)
        {
            // buffers[bindingIdxInput] = createMnistCudaBuffer(bufferSizesInput.first,
            //                                                  bufferSizesInput.second, run);
            for (int i = 0; i < batchSize * INPUT_H * INPUT_W; i++)
            {
                input_batch[i] = input_p_he[i + batchSize * INPUT_H * INPUT_W * run];
            }
            buffers[bindingIdxInput] = createRealCudaBuffer(input_batch,
                                                            bufferSizesInput.first,
                                                            bufferSizesInput.second, run);

            auto t_start = std::chrono::high_resolution_clock::now();
            context->execute(batchSize, &buffers[0]);
            auto t_end = std::chrono::high_resolution_clock::now();
            ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            total += ms;

            for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
            {
                if (engine.bindingIsInput(bindingIdx))
                    continue;

                auto bufferSizesOutput = buffersSizes[bindingIdx];
                float out_arr[bufferSizesOutput.first];
                printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                            buffers[bindingIdx], out_arr);
                output_real.insert(
                    output_real.end(),
                    out_arr,
                    out_arr + bufferSizesOutput.first);
            }
            CHECK(cudaFree(buffers[bindingIdxInput]));
        }

        total /= numberRun;
        std::cout << "Average over " << numberRun << " runs is " << total << " ms." << std::endl;
    }

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
    context->destroy();
}
