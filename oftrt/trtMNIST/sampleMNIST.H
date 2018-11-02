//! \file sampleMNIST.cpp
//! \brief This file contains the implementation of the MNIST sample.
//!
//! It builds a TensorRT engine by importing a trained MNIST Caffe model. It uses the engine to run
//! inference on an input image of a digit.
//! It can be run with the following command line:
//! Command: ./sample_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]

#include "common.h"
#include "argsParser.h"
#include "buffers.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

static Logger gLogger;

//!
//! \brief The MNISTSampleParams structure groups the additional parameters required by
//!         the MNIST sample.
//!
struct MNISTSampleParams : public samplesCommon::CaffeSampleParams
{
    std::string meanFileName;
};

//!
//! \brief This function initializes members of the params struct using the command line args
//!
MNISTSampleParams initializeSampleParams(const samplesCommon::Args &args)
{
    MNISTSampleParams params;
    if (args.dataDirs.size() != 0) //!< Use the data directory provided by the user
        params.dataDirs = args.dataDirs;
    else //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    params.prototxtFileName = "mnist.prototxt";
    params.weightsFileName = "mnist.caffemodel";
    params.meanFileName = "mnist_mean.binaryproto";
    params.inputTensorNames.push_back("data");
    params.batchSize = 1;
    params.outputTensorNames.push_back("prob");
    params.dlaID = args.useDLA;

    return params;
}

//!
//! \brief  The SampleMNIST class implements the MNIST sample
//!
//! \details It creates the network using a trained Caffe MNIST classification model
//!
class SampleMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

  public:
    SampleMNIST(const MNISTSampleParams &params)
        : mParams(params)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief This function runs the TensorRT inference engine for this sample
    //!
    bool infer(std::vector<float> &out);

    //!
    //! \brief This function can be used to clean up any state created in the sample class
    //!
    bool teardown();

  private:
    //!
    //! \brief This function uses a Caffe parser to create the MNIST Network and marks the
    //!        output layers
    //!
    void constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder, SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvcaffeparser1::ICaffeParser> &parser);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager &buffers, const std::string &inputTensorName, int inputFileIdx) const;

    //!
    //! \brief Verifies that the output is correct and prints it
    //!
    bool verifyOutput(const samplesCommon::BufferManager &buffers, const std::string &outputTensorName, int groundTruthDigit) const;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr; //!< The TensorRT engine used to run the network

    MNISTSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    // nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int outSize;

    SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob> mMeanBlob; //! the mean blob, which we need to keep around until build is done
};
