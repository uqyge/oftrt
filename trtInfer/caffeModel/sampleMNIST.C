#include <sampleMNIST.H>

bool SampleMNIST::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
        return false;

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
        return false;

    auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    if (!parser)
        return false;

    constructNetwork(builder, network, parser);
    builder->setMaxBatchSize(mParams.batchSize);
    builder->setMaxWorkspaceSize(16_MB);
    builder->allowGPUFallback(true);
    if (mParams.dlaID > 0)
        samplesCommon::enableDLA(builder.get(), mParams.dlaID);

    mEngine = std::move(std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network), samplesCommon::InferDeleter()));

    if (!mEngine)
        return false;

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    // mOutputDims = network->getOutput(0)->getDimensions();
    // mOutputDims = mInputDims;
    outSize = network->getOutput(0)->getDimensions().d[0];
    std::cout << network->getOutput(0)->getDimensions().d[0];

    assert(mInputDims.nbDims == 3);

    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool SampleMNIST::processInput(const samplesCommon::BufferManager &buffers, const std::string &inputTensorName, int inputFileIdx) const
{
    std::cout << "d0:" << mInputDims.d[0] << '\n'
              << "d1:" << mInputDims.d[1] << '\n'
              << "d2:" << mInputDims.d[2] << '\n';
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];

    // Read a random digit file
    srand(unsigned(time(nullptr)));
    uint8_t fileData[inputH * inputW];
    readPGMFile(locateFile(std::to_string(inputFileIdx) + ".pgm", mParams.dataDirs), fileData, inputH, inputW);

    std::cout << "input buffer : " << buffers.size(mParams.inputTensorNames[0]) / sizeof(float) << '\n';
    // Print ASCII representation of digit
    std::cout << "\nInput:\n"
              << std::endl;
    for (int i = 0; i < inputH * inputW; i++)
        std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");

    float *hostInputBuffer = static_cast<float *>(buffers.getHostBuffer(inputTensorName));

    for (int i = 0; i < inputH * inputW; i++)
        hostInputBuffer[i] = float(fileData[i]);

    return true;
}

//!
//! \brief Verifies that the output is correct and prints it
//!
bool SampleMNIST::verifyOutput(const samplesCommon::BufferManager &buffers, const std::string &outputTensorName, int groundTruthDigit) const
{
    const float *prob = static_cast<const float *>(buffers.getHostBuffer(outputTensorName));

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";

    std::cout << "outPutTensorName is " << outputTensorName << '\n';
    int vec_num = buffers.size(outputTensorName) / sizeof(*prob);
    std::cout << "length is: " << vec_num << '\n';
    std::vector<float> vec_prob(prob, prob + vec_num);

    float val{0.0f};
    int idx{0};
    for (unsigned int i = 0; i < 10; i++)
    {
        val = std::max(val, prob[i]);
        if (val == prob[i])
            idx = i;
        std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }

    return (idx == groundTruthDigit && val > 0.9f);
}

//!
//! \brief This function uses a caffe parser to create the MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the MNIST network
//!
//! \param builder Pointer to the engine builder
//!
void SampleMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder, SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvcaffeparser1::ICaffeParser> &parser)
{
    const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor = parser->parse(
        locateFile(mParams.prototxtFileName, mParams.dataDirs).c_str(),
        locateFile(mParams.weightsFileName, mParams.dataDirs).c_str(),
        *network,
        nvinfer1::DataType::kFLOAT);

    for (auto &s : mParams.outputTensorNames)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // nvinfer1::Dims outDims;
    // outDims = network->getOutput(0)->getDimensions();
    // std::cout << "outdims" << outDims.d[0] << '\n';

    // add mean subtraction to the beginning of the network
    Dims inputDims = network->getInput(0)->getDimensions();
    mMeanBlob = SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob>(parser->parseBinaryProto(locateFile(mParams.meanFileName, mParams.dataDirs).c_str()));
    Weights meanWeights{DataType::kFLOAT, mMeanBlob->getData(), inputDims.d[1] * inputDims.d[2]};

    auto mean = network->addConstant(Dims3(1, inputDims.d[1], inputDims.d[2]), meanWeights);
    auto meanSub = network->addElementWise(*network->getInput(0), *mean->getOutput(0), ElementWiseOperation::kSUB);
    network->getLayer(0)->setInput(0, *meanSub->getOutput(0));
}

//!
//! \brief This function runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleMNIST::infer(std::vector<float> &out)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
        return false;

    // Pick a random digit to try to infer
    srand(time(NULL));
    // const int digit = rand() % 10;

    //prepare input
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
    uint8_t fileData_1[inputH * inputW];
    uint8_t fileData_3[inputH * inputW];
    uint8_t fileData_5[inputH * inputW];
    readPGMFile(locateFile(std::to_string(1) + ".pgm", mParams.dataDirs), fileData_1, inputH, inputW);
    readPGMFile(locateFile(std::to_string(3) + ".pgm", mParams.dataDirs), fileData_3, inputH, inputW);
    readPGMFile(locateFile(std::to_string(5) + ".pgm", mParams.dataDirs), fileData_5, inputH, inputW);
    const int tot_n = 3;
    float data[tot_n * inputH * inputW];
    for (int i = 0; i < inputH * inputW; i++)
    {
        data[i] = float(fileData_1[i]);
        data[1 * inputH * inputW + i] = float(fileData_3[i]);
        data[2 * inputH * inputW + i] = float(fileData_5[i]);
    }

    // output holder
    std::vector<float> out_vec;
    for (int n = 0; n < tot_n; n += mParams.batchSize)
    {

        // Read the input data into the managed buffers
        // There should be just 1 input tensor
        assert(mParams.inputTensorNames.size() == 1);
        std::cout << "input name: " << mParams.inputTensorNames[0] << '\n';
        float *hostInputBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
        for (int i = 0; i < mParams.batchSize * inputH * inputW; i++)
            hostInputBuffer[i] = float(data[i + n * inputH * inputW]);

        // Create CUDA stream for the execution of this inference.
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        
            // Asynchronously copy data from host input buffers to device input buffers
            buffers.copyInputToDeviceAsync(stream);

        // Asynchronously enqueue the inference work
        if (!context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr))
            return false;

        // Asynchronously copy data from device output buffers to host output buffers
        buffers.copyOutputToHostAsync(stream);

        // Wait for the work in the stream to complete
        cudaStreamSynchronize(stream);

        // Release stream
        cudaStreamDestroy(stream);

        // Check and print the output of the inference
        // There should be just one output tensor
        assert(mParams.outputTensorNames.size() == 1);

        // prepare output
        const float *prob_out = static_cast<const float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

        out_vec.insert(out_vec.end(),
                       prob_out,
                       prob_out + mParams.batchSize * outSize);
    }
    // std::cout << "m out size: " << mOutputDims.d[0] << '\n';
    out_vec.resize(tot_n * outSize);
    out = out_vec;
    return 0;
}

//!
//! \brief This function can be used to clean up any state created in the sample class
//!
bool SampleMNIST::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

void printHelpInfo()
{
    std::cout << "Usage: ./sample_mnist [-h or --help] [-d or --datadir=<path to data directory>]\n";
    std::cout << "--help     Display help information\n";
    std::cout << "--datadir  Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
}
