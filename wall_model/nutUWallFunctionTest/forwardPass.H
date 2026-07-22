#ifndef forwardPass_H
#define forwardPass_H

#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

#include "scalar.H"
#include "fileName.H"
#include "error.H"

namespace Foam
{

struct DenseLayer
{
    label inDim, outDim;
    std::vector<scalar> W;   // flattened, row-major: W[i*outDim + j]
    std::vector<scalar> b;
    scalar (*activation)(scalar);

    std::vector<scalar> forward(const std::vector<scalar>& x) const;
};

std::vector<scalar> loadBin(const fileName& path, label n);

class wallModelMLP
{
    std::vector<DenseLayer> layers_;
    //std::vector<scalar> inMean_, inStd_, outMean_, outStd_;

public:
    wallModelMLP(const fileName& dir);
    scalar predict(const std::vector<scalar>& rawInputs) const;
};

}

#endif