#include "uffModel.H"

bool uffModel::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
        return false;

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
        return false;

    // auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    parser->registerInput("input_1", Dims3(1, 1, 2), nvuffparser::UffInputOrder::kNCHW);
    // parser->registerInput("input_1", Dims2(2, 1), UffInputOrder::kNCHW);
    parser->registerOutput("dense_2/BiasAdd");

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

    assert(mInputDims.nbDims == 3);

    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool uffModel::processInput(const samplesCommon::BufferManager &buffers, const std::string &inputTensorName, int inputFileIdx) const
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
bool uffModel::verifyOutput(const samplesCommon::BufferManager &buffers, const std::string &outputTensorName, int groundTruthDigit) const
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
void uffModel::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder, SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvuffparser::IUffParser> &parser)
{
    parser->parse(locateFile(mParams.uffFileName, mParams.dataDirs).c_str(), *network, nvinfer1::DataType::kFLOAT);
}

//!
//! \brief This function runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool uffModel::infer(std::vector<float> &data_in, std::vector<float> &out)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
        return false;

    //prepare input

    const int inputH = 2;
    const int inputW = 1;
    // float data[4] = {0.0, 0.0, 0.0, 0.0};
    const int tot_n = ceil(float(data_in.size()) / mParams.batchSize);
    // output holder
    std::vector<float> out_vec;
    assert(mParams.inputTensorNames.size() == 1);
    // std::cout << "input name: " << mParams.inputTensorNames[0] << '\n';
    std::cout << "There are " << tot_n << " batches.\n";
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < tot_n; n += mParams.batchSize)
    {
        float *hostInputBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
        for (int i = 0; i < mParams.batchSize * inputH * inputW; i++)
            hostInputBuffer[i] = float(data_in[i + n * inputH * inputW]);

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
    auto t_end = std::chrono::high_resolution_clock::now();
    auto total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "totol time is " << total << "ms."
              << "\n";
    // std::cout << "m out size: " << mOutputDims.d[0] << '\n';
    out_vec.resize(tot_n * outSize);
    out = out_vec;
    return 0;
}

//!
//! \brief This function can be used to clean up any state created in the sample class
//!
bool uffModel::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    // nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}
