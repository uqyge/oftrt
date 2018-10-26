//! \file sampleMNIST.cpp
//! \brief This file contains the implementation of the MNIST sample.
//!
//! It builds a TensorRT engine by importing a trained MNIST Caffe model. It uses the engine to run
//! inference on an input image of a digit.
//! It can be run with the following command line:
//! Command: ./sample_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]

// #include "common.h"
// #include "argsParser.h"
// #include "buffers.h"

// #include "NvCaffeParser.h"
// #include "NvInfer.h"

// #include <cuda_runtime_api.h>
// #include <algorithm>
// #include <cassert>
// #include <cmath>
// #include <fstream>
// #include <iostream>
// #include <sstream>

// static Logger gLogger;

// //!
// //! \brief The MNISTSampleParams structure groups the additional parameters required by
// //!         the MNIST sample.
// //!
// struct MNISTSampleParams : public samplesCommon::CaffeSampleParams
// {
//     std::string meanFileName;
// };

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
    const int digit = rand() % 10;

    // Read the input data into the managed buffers
    // There should be just 1 input tensor
    assert(mParams.inputTensorNames.size() == 1);
    std::cout << "input name: " << mParams.inputTensorNames[0] << '\n';
    if (!processInput(buffers, mParams.inputTensorNames[0], digit))
        return false;

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
    bool outputCorrect = verifyOutput(buffers, mParams.outputTensorNames[0], digit);
    const float *prob_1 = static_cast<const float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    int vec_num = 10;
    std::vector<float> vec_prob(prob_1, prob_1 + vec_num);
    out = vec_prob;
    return outputCorrect;
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