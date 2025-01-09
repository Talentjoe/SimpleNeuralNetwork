#include <cstdint>
#include <ctime>

#include "../lib/NN.h"
#include "../lib/readData.h"

using namespace std;


int main() {
    const int termsOfTrain = 5;
    double Srate = 0.1;

    srand(time(nullptr) * 3049);

    auto *nn = new NN::NNcore();
    nn->init(vector{28 * 28, 128, 64, 10}, Srate);

    vector<vector<double> > inData;
    vector<int> outData;
    inData = readData::readData::readImageData("../Data/train-images.idx3-ubyte");
    outData = readData::readData::readTagData("../Data/train-labels.idx1-ubyte");

    for (int j = 0; j < termsOfTrain; j++) {
        nn->train(inData, outData, true);
        Srate *= 0.05;
        nn->changeStudyRate(Srate);
    }

    vector<vector<double> > testInData;
    vector<int> testOutData;
    testInData = readData::readData::readImageData("../Data/t10k-images.idx3-ubyte");
    testOutData = readData::readData::readTagData("../Data/t10k-labels.idx1-ubyte");

    nn->test(testInData, testOutData);

    NN::NNcore::save(*nn, "test.mod");

    auto *nn2 = new NN::NNcore();
    nn2->init("test.mod", 0);

    nn2->test(testInData, testOutData);

    return 0;
}
