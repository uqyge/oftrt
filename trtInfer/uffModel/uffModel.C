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
    // parser->registerInput("input_1", Dims3(1, 1, 2), nvuffparser::UffInputOrder::kNCHW);
    parser->registerInput("input_1", Dims3(3, 1, 1), nvuffparser::UffInputOrder::kNCHW);

    parser->registerOutput("dense_2/BiasAdd");

    if (!parser)
        return false;

    constructNetwork(builder, network, parser);
    builder->setMaxBatchSize(mParams.batchSize);
    builder->setMaxWorkspaceSize(64_MB);
    builder->allowGPUFallback(true);
    // builder->setFp16Mode(builder->platformHasFastFp16());
    // builder->setFp16Mode(true);
    // std::cout << builder->platformHasFastFp16() << '\n';
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
    std::cout << "out size is " << outSize << '\n';
    assert(mInputDims.nbDims == 3);

    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
// bool uffModel::processInput(const samplesCommon::BufferManager &buffers, const std::string &inputTensorName, int inputFileIdx) const
// {
//     std::cout << "d0:" << mInputDims.d[0] << '\n'
//               << "d1:" << mInputDims.d[1] << '\n'
//               << "d2:" << mInputDims.d[2] << '\n';
//     const int inputH = mInputDims.d[1];
//     const int inputW = mInputDims.d[2];

//     // Read a random digit file
//     srand(unsigned(time(nullptr)));
//     uint8_t fileData[inputH * inputW];
//     readPGMFile(locateFile(std::to_string(inputFileIdx) + ".pgm", mParams.dataDirs), fileData, inputH, inputW);

//     std::cout << "input buffer : " << buffers.size(mParams.inputTensorNames[0]) / sizeof(float) << '\n';
//     // Print ASCII representation of digit
//     std::cout << "\nInput:\n"
//               << std::endl;
//     for (int i = 0; i < inputH * inputW; i++)
//         std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");

//     float *hostInputBuffer = static_cast<float *>(buffers.getHostBuffer(inputTensorName));

//     for (int i = 0; i < inputH * inputW; i++)
//         hostInputBuffer[i] = float(fileData[i]);

//     return true;
// }

//!
//! \brief Verifies that the output is correct and prints it
//!
// bool uffModel::verifyOutput(const samplesCommon::BufferManager &buffers, const std::string &outputTensorName, int groundTruthDigit) const
// {
//     const float *prob = static_cast<const float *>(buffers.getHostBuffer(outputTensorName));

//     // Print histogram of the output distribution
//     std::cout << "\nOutput:\n\n";

//     std::cout << "outPutTensorName is " << outputTensorName << '\n';
//     int vec_num = buffers.size(outputTensorName) / sizeof(*prob);
//     std::cout << "length is: " << vec_num << '\n';
//     std::vector<float> vec_prob(prob, prob + vec_num);

//     float val{0.0f};
//     int idx{0};
//     for (unsigned int i = 0; i < 10; i++)
//     {
//         val = std::max(val, prob[i]);
//         if (val == prob[i])
//             idx = i;
//         std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
//     }

//     return (idx == groundTruthDigit && val > 0.9f);
// }

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
    const int inputH = 3;
    const int inputW = 1;
    // float data[4] = {0.0, 0.0, 0.0, 0.0};
    const int tot_n = ceil(float(data_in.size()) / (inputH * inputW));
    const int tot_b = ceil(float(tot_n) / mParams.batchSize);

    // output holder
    std::vector<float> out_vec;
    assert(mParams.inputTensorNames.size() == 1);
    // std::cout << "input name: " << mParams.inputTensorNames[0] << '\n';

    std::cout << "There are " << tot_b << " batches.\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < tot_n; n += mParams.batchSize)
    {
        // std::cout << n << "\n";
        float *hostInputBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
        // std::cout << mParams.inputTensorNames[0] << "\n";
        for (int i = 0; i < mParams.batchSize * inputH * inputW; i++)
        {
            // std::cout << i << "\n";
            // std::cout << "in " << float(data_in[i + n * inputH * inputW]) << "\n";
            // hostInputBuffer[i] = float(data_in[i + n * inputH * inputW]);
            hostInputBuffer[i] = float(data_in[i + n * inputH * inputW]);
            // std::cout << "host" << hostInputBuffer[i] << "\n";
        }

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
    out_vec.resize(data_in.size() / (inputH * inputW) * outSize);
    out = out_vec;
    std::cout << "out is " << out.size() << "\n";
    return 0;
}

bool uffModel::infer_new(std::vector<float> &data_in, std::vector<float> &out)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
    // auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    // if (!context)
    //     return false;
    // std::cout << mEngine->getBindingIndex("input_1") << "\n";
    // std::cout << mEngine->getBindingIndex("dense_2/BiasAdd") << "\n";

    //prepare input
    const int inputH = 3;
    const int inputW = 1;

    const int tot_n = ceil(float(data_in.size()) / (inputH * inputW));
    const int tot_b = ceil(float(tot_n) / mParams.batchSize);
    std::cout << "There are " << tot_b << " batches.\n";

    // output holder
    std::vector<float> out_vec;
    assert(mParams.inputTensorNames.size() == 1);

    const int nStreams = 10;
    cudaStream_t stream[nStreams];
    std::vector<IExecutionContext *> contexts;
    for (int i = 0; i < nStreams; i++)
    {
        // auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        auto context = mEngine->createExecutionContext();
        contexts.push_back(context);
    }
    // std::cout << contexts.size() << " number of streams\n";

    const int bPs = ceil(float(tot_b) / nStreams);
    std::cout << bPs << " batch per stream\n";

    //size of batch
    const int streamSize_a = mParams.batchSize * (inputH * inputW);
    const int streamBytes_a = streamSize_a * sizeof(float);
    const int streamSize_b = mParams.batchSize * outSize;
    const int streamBytes_b = streamSize_b * sizeof(float);

    // void *buffers_stream[2];
    const int dataSize_a = tot_b * mParams.batchSize * (inputH * inputW) * sizeof(float);
    const int dataSize_b = tot_b * mParams.batchSize * outSize * sizeof(float);
    // const int dataSize_a = nStreams * bPs * mParams.batchSize * (inputH * inputW) * sizeof(float);
    // const int dataSize_b = nStreams * bPs * mParams.batchSize * outSize * sizeof(float);

    float *a, *d_a;
    float *b, *d_b;

    a = &data_in[0];
    // cudaMallocHost((void **)&a, dataSize_a);
    cudaMalloc((void **)&d_a, dataSize_a);
    cudaMallocHost((void **)&b, dataSize_b);
    cudaMalloc((void **)&d_b, dataSize_b);
    void *buffers_stream[2];
    // std::cout << a[0] << " a0\n";
    // std::cout << b[0] << " b0\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < nStreams; ++i)
        cudaStreamCreate(&stream[i]);
    for (int i = 0; i < nStreams; ++i)
    {
        // std::cout << i << " i\n";
        for (int j = 0; j < bPs; ++j)
        {
            // std::cout << j << " j\n";
            int offset_a = j * streamSize_a + i * bPs * streamSize_a;
            int offset_b = j * streamSize_b + i * bPs * streamSize_b;
            if (i * bPs + j + 1 >= tot_b)
                break;

            // std::cout << offset_a << " off_a\n";
            // // std::cout << offset_b << " off_b\n";
            cudaMemcpyAsync(&d_a[offset_a], &a[offset_a],
                            streamBytes_a, cudaMemcpyHostToDevice,
                            stream[i]);

            // prepare buffers

            buffers_stream[0] = &d_a[offset_a];
            buffers_stream[1] = &d_b[offset_b];

            //kernel
            if (!contexts[i]->enqueue(mParams.batchSize, buffers_stream, stream[i], nullptr))
                return false;
            // std::cout << " there\n";
            cudaMemcpyAsync(&b[offset_b], &d_b[offset_b],
                            streamBytes_b, cudaMemcpyDeviceToHost,
                            stream[i]);
        }
        // cudaStreamSynchronize(stream[i]);
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < nStreams; ++i)
    {
        // cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
        contexts[i]->destroy();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Time for gpu inference is " << total << "ms.\n";

    // std::cout << d_b[0] << "\n";
    // std::cout << a[0] << " a0\n";
    std::cout << b[0] << " b0\n";
    out_vec.insert(out_vec.end(),
                   b,
                   b + data_in.size() / (inputH * inputW) * outSize);

    out = out_vec;
    // std::cout << "out is " << out.size() << "\n";

    cudaFree(d_a);
    // cudaFreeHost(a);
    cudaFree(d_b);
    cudaFreeHost(b);

    return 0;
}

// bool uffModel::infer_new(std::vector<float> &data_in, std::vector<float> &out)
// {
//     // Create RAII buffer manager object
//     // samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
//     samplesCommon::BufferManager buffers(mEngine, data_in.size());

//     auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
//     if (!context)
//         return false;

//     //prepare input
//     const int inputH = 3;
//     const int inputW = 1;
//     // float data[4] = {0.0, 0.0, 0.0, 0.0};
//     const int tot_n = ceil(float(data_in.size()) / (inputH * inputW));
//     const int tot_b = ceil(float(tot_n) / mParams.batchSize);

//     // output holder
//     std::vector<float> out_vec;
//     assert(mParams.inputTensorNames.size() == 1);
//     // std::cout << "input name: " << mParams.inputTensorNames[0] << '\n';

//     std::cout << "There are " << tot_b << " batches.\n";

//     auto t_start = std::chrono::high_resolution_clock::now();
//     // for (int n = 0; n < tot_n; n += mParams.batchSize)
//     {
//         // std::cout << n << "\n";
//         float *hostInputBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
//         // std::cout << mParams.inputTensorNames[0] << "\n";
//         std::cout << "a\n";
//         for (unsigned i = 0; i < data_in.size(); i++)
//         // for (unsigned i = 0; i < data_in.size() / 100; i++)
//         {
//             // std::cout << i << "\n";
//             // std::cout << "in " << float(data_in[i + n * inputH * inputW]) << "\n";
//             // hostInputBuffer[i] = float(data_in[i + n * inputH * inputW]);
//             hostInputBuffer[i] = float(data_in[i]);
//             // std::cout << "host" << hostInputBuffer[i] << "\n";
//         }

//         // Create CUDA stream for the execution of this inference.
//         cudaStream_t stream;
//         CHECK(cudaStreamCreate(&stream));
//         std::cout << "b\n";
//         // Asynchronously copy data from host input buffers to device input buffers
//         buffers.copyInputToDeviceAsync(stream);
//         std::cout << "b\n";
//         // Asynchronously enqueue the inference work
//         if (!context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr))
//             return false;

//         // Asynchronously copy data from device output buffers to host output buffers
//         buffers.copyOutputToHostAsync(stream);

//         // Wait for the work in the stream to complete
//         cudaStreamSynchronize(stream);

//         // Release stream
//         cudaStreamDestroy(stream);

//         // Check and print the output of the inference
//         // There should be just one output tensor
//         assert(mParams.outputTensorNames.size() == 1);
//         std::cout << "here\n";
//         // prepare output
//         const float *prob_out = static_cast<const float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

//         out_vec.insert(out_vec.end(),
//                        prob_out,
//                        prob_out + data_in.size() * outSize);
//     }
//     auto t_end = std::chrono::high_resolution_clock::now();
//     auto total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
//     std::cout << "totol time is " << total << "ms."
//               << "\n";
//     // std::cout << "m out size: " << mOutputDims.d[0] << '\n';
//     out_vec.resize(data_in.size() / (inputH * inputW) * outSize);
//     out = out_vec;
//     std::cout << "out is " << out.size() << "\n";
//     return 0;
// }

//!
//! \brief This function can be used to clean up any state created in the sample class
//!
bool uffModel::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    // nvcaffeparser1::shutdownProtobufLibrary();
    nvuffparser::shutdownProtobufLibrary();
    return true;
}
