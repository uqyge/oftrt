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
// static const int OUTPUT_SIZE = 10;

std::string locateFile(const std::string &input)
{
    std::vector<std::string> dirs{"data/mnist/", "data/samples/mnist/", "./data/"};
    return locateFile(input, dirs);
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

void *createRealCudaBuffer(float *input_p_he, int64_t eltCount, DataType dtype, int run)
{
    /* in that specific case, eltCount == INPUT_H * INPUT_W */
    // std::cout << eltCount << '\n';
    // std::cout << input_p_he[0] << '\n'
    //           << input_p_he[1] << '\n';

    // assert(eltCount == INPUT_H * INPUT_W);
    assert(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    float *inputs = new float[eltCount];

    for (int i = 0; i < eltCount; i++)
    {
        inputs[i] = input_p_he[i];
        // std::cout << inputs[i] << '\n';
    }

    void *deviceMem = safeCudaMalloc(memSize);
    // CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceMem, input_p_he, memSize, cudaMemcpyHostToDevice));

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
    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        out_arr[eltIdx] = outputs[eltIdx];
        // std::cout << eltIdx << " => " << outputs[eltIdx] << "\t : ";
        // if (eltIdx == maxIdx)
        //     std::cout << "***";
        // std::cout << "\n";
    }
    // std::cout << std::endl;
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

    int numberRun;
    numberRun = ceil(float(input_p_he.size()) / bufferSizesInput.first);
    std::cout << "rounds:in " << input_p_he.size() << '\n';
    std::cout << "rounds:" << numberRun << '\n';
    float input_batch[batchSize * INPUT_H * INPUT_W] = {0};
    // int iterations = 1;
    // for (int i = 0; i < iterations; i++)
    {
        float total = 0, ms, coutt, cint;
        float tot_in = 0;
        float tot_out = 0;
        for (int run = 0; run < numberRun; run++)
        {
            auto t_start = std::chrono::high_resolution_clock::now();
            // buffers[bindingIdxInput] = createMnistCudaBuffer(bufferSizesInput.first,
            //                                                  bufferSizesInput.second, run);
            for (int i = 0; i < batchSize * INPUT_H * INPUT_W; i++)
            {
                input_batch[i] = input_p_he[i + batchSize * INPUT_H * INPUT_W * run];
            }
            buffers[bindingIdxInput] = createRealCudaBuffer(input_batch,
                                                            bufferSizesInput.first,
                                                            bufferSizesInput.second, run);
            auto t_cin_e = std::chrono::high_resolution_clock::now();
            cint = std::chrono::duration<float, std::milli>(t_cin_e - t_start).count();
            tot_in += cint;
            std::cout << "memory copy d2h time is " << cint << " ms." << std::endl;

            // auto t_start = std::chrono::high_resolution_clock::now();
            context->execute(batchSize, &buffers[0]);
            // cudaStream_t stream;
            // CHECK(cudaStreamCreate(&stream));
            // context->enqueue(batchSize, &buffers[0], stream, nullptr);
            // Wait for the work in the stream to complete
            // cudaStreamSynchronize(stream);

            // Release stream
            // cudaStreamDestroy(stream);

            // auto t_end = std::chrono::high_resolution_clock::now();
            // ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            // total += ms;

            for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
            {
                if (engine.bindingIsInput(bindingIdx))
                    continue;

                auto bufferSizesOutput = buffersSizes[bindingIdx];
                float out_arr[bufferSizesOutput.first];
                auto t_cout_s = std::chrono::high_resolution_clock::now();
                printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                            buffers[bindingIdx], out_arr);
                auto t_cout_e = std::chrono::high_resolution_clock::now();
                coutt = std::chrono::duration<float, std::milli>(t_cout_e - t_cout_s).count();
                std::cout << "memory copy h2d time is " << coutt << " ms." << std::endl;
                tot_out += coutt;
                output_real.insert(
                    output_real.end(),
                    out_arr,
                    out_arr + bufferSizesOutput.first);
            }
            CHECK(cudaFree(buffers[bindingIdxInput]));
            auto t_end = std::chrono::high_resolution_clock::now();
            ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            total += ms;
        }

        // total /= numberRun;
        // std::cout << "Average over " << numberRun << " runs is " << total << " ms." << std::endl;
        std::cout << "Total time is " << total << " ms." << std::endl;
        float memTime = tot_in + tot_out;
        float annTime = total - memTime;
        std::cout << "Total memory time is " << memTime << " ms." << std::endl;
        std::cout << "Total network running time is " << annTime << " ms." << std::endl;
    }

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
    context->destroy();
}
