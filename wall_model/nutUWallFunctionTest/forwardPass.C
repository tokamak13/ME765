#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

#include "scalar.H"
#include "fileName.H"
#include "error.H"

#include "forwardPass.H"

namespace Foam
{

// --- Activation functions ---
inline scalar relu(scalar x) { return std::max<scalar>(0, x); }
inline scalar tanhAct(scalar x) { return std::tanh(x); }
inline scalar linear(scalar x) { return x; }


std::vector<scalar> Foam::DenseLayer::forward
(
    const std::vector<scalar>& x
) const
{
    if (label(x.size()) != inDim)
    {
        FatalErrorInFunction
            << "Input size " << x.size()
            << " does not match expected inDim " << inDim
            << exit(FatalError);
    }
    
    std::vector<scalar> y(outDim, 0.0);

    for (label j = 0; j < outDim; ++j)
    {
        scalar sum = b[j];

        for (label i = 0; i < inDim; ++i)
        {
            sum += x[i] * W[i*outDim + j];
        }

        y[j] = activation(sum);
    }

    return y;
}

// --- Load a flat binary blob into a vector ---
std::vector<scalar> loadBin(const fileName& path, label n)
{
    std::vector<scalar> data(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) FatalErrorInFunction << "Cannot open " << path << exit(FatalError);
    f.read(reinterpret_cast<char*>(data.data()), n * sizeof(scalar));
    return data;
}


wallModelMLP::wallModelMLP(const fileName& dir)
:
    layers_() //,
    //inMean_(),
    //inStd_(),
    //outMean_(),
    //outStd_()
{
    // hard-code architecture from your Step 3 printout, e.g. 3 -> 16 -> 16 -> 1
    // DenseLayer l0{3, 32, loadBin(dir/"layer0_W.bin", 3*32),
    //                    loadBin(dir/"layer0_b.bin", 32), relu};
    DenseLayer l0{3, 32, loadBin(dir/"layer0_W.bin", 3*32),
                             loadBin(dir/"layer0_b.bin", 32), relu};
    DenseLayer l1{32, 64, loadBin(dir/"layer1_W.bin", 32*64),
                             loadBin(dir/"layer1_b.bin", 64), relu};
    DenseLayer l2{64, 64, loadBin(dir/"layer2_W.bin", 64*64),
                             loadBin(dir/"layer2_b.bin", 64), relu};
    DenseLayer l3{64, 64, loadBin(dir/"layer3_W.bin", 64*64),
                             loadBin(dir/"layer3_b.bin", 64), relu};
    DenseLayer l4{64, 32, loadBin(dir/"layer4_W.bin", 64*32),
                             loadBin(dir/"layer4_b.bin", 32), relu};
    DenseLayer l5{32, 1, loadBin(dir/"layer5_W.bin", 32*1),
                            loadBin(dir/"layer5_b.bin", 1), linear};
    layers_ = {l0, l1, l2, l3, l4, l5};

    //inMean_ = loadBin(dir/"input_mean.bin", 3);
    //inStd_  = loadBin(dir/"input_std.bin", 3);
    //outMean_ = loadBin(dir/"output_mean.bin", 1);
    //outStd_  = loadBin(dir/"output_std.bin", 1);
}

scalar wallModelMLP::predict
(
    const std::vector<scalar>& rawInputs
) const
{
    std::vector<scalar> x(rawInputs.size());
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = (rawInputs[i]); // - inMean_[i]) / inStd_[i];

    for (const auto& layer : layers_)
        x = layer.forward(x);

    return x[0]; //* outStd_[0] + outMean_[0];   // denormalize
}


}