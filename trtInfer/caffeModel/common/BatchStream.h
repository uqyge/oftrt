#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include "NvInfer.h"
#include "common.h"
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <vector>

class BatchStream
{
public:
    BatchStream(int batchSize, int maxBatches, std::string prefix, std::vector<std::string> directories);

    void reset(int firstBatch);

    bool next();

    void skip(int skipCount);

    float* getBatch();

    int getBatchesRead() const;

    int getBatchSize() const;

    int getImageSize() const;

    nvinfer1::Dims getDims() const;

private:
    float* getFileBatch();

    bool update();

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mFileCount{0};
    int mFileBatchPos{0};
    int mImageSize{0};
    nvinfer1::Dims mDims;
    std::vector<float> mBatch;
    std::vector<float> mFileBatch;
    std::string mPrefix;
    std::vector<std::string> mDataDir;
};
#endif
