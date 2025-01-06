//
// Created by lenovo on 25-1-6.
//

#include "NN.h"

namespace NN {
    using namespace std;

    int size;
    double studyRate;

    std::vector<std::vector<double> > layers;
    std::vector<std::vector<double> > layersZ;
    std::vector<int> layerSize;

    std::vector<std::vector<double> > b;

    std::vector<std::vector<std::vector<double> > > w;


    double getRandomDoubleNumber(double max = 1, double min = -1) {
        return min + (double) (rand()) / ((double) (RAND_MAX / (max - min)));
    }

    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    double sigmoidP(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }


    std::vector<double> NNcore::forward(std::vector<double> inNums, bool printRes) {
        if (inNums.size() != layerSize[0]) {
            cout << "Size Not Mathch !! " << endl;
            return {};
        }
        layers[0] = inNums;

        for (auto &row: layersZ) // 将内容归零
        {
            fill(row.begin(), row.end(), 0.0);
        }

        for (int i = 1; i < size; i++) {
            for (int k = 0; k < layerSize[i]; k++) {
                for (int j = 0; j < layerSize[i - 1]; j++)
                    layersZ[i][k] += layers[i - 1][j] * w[i - 1][j][k];
                layersZ[i][k] += b[i][k];
                layers[i][k] = sigmoid(layersZ[i][k]);
            }
        }
        if (printRes) {
            for (auto i: layers[size - 1]) cout << setw(10) << i;
            cout << endl;
        }
        return layers[size - 1];
    }

    double NNcore::CalCost(std::vector<double> correctOut) {
        double cost = 0;
        if (correctOut.size() != layerSize[size - 1]) {
            cout << "Size Error" << endl;
            return 0;
        }
        for (int i = 0; i < layerSize[size - 1]; i++) {
            cost += (correctOut[i] * log(layers[size - 1][i]) + (1 - correctOut[i]) * log(1 - layers[size - 1][i])) /
                    2.0;
        }
        cost /= -layerSize[size - 1];
        return cost;
    }

    double NNcore::pushBack(std::vector<double> correctOut) {
        std::vector<std::vector<double> > delta(size);
        for (int i = 1; i < size; i++) {
            delta[i].resize(layerSize[i]);
        }

        for (int i = 0; i < layerSize[size - 1]; i++) {
            delta[size - 1][i] = (layers[size - 1][i] - correctOut[i]) * sigmoidP(layersZ[size - 1][i]); //BP1
        }
        for (int i = size - 2; i > 0; i--) {
            for (int j = 0; j < layerSize[i]; j++) {
                double value = 0;
                for (int k = 0; k < layerSize[i + 1]; k++) {
                    value += w[i][j][k] * delta[i + 1][k];
                }
                delta[i][j] = sigmoidP(layersZ[i][j]) * value; // BP2
            }
        }
        for (int i = 1; i < size; i++) {
            for (int j = 0; j < layerSize[i]; j++) {
                b[i][j] -= delta[i][j] * studyRate;

                for (int k = 0; k < layerSize[i - 1]; k++) {
                    //cout<<layers[i-1][k]*delta[i][j]*studyRate;
                    w[i - 1][k][j] -= layers[i - 1][k] * delta[i][j] * studyRate;
                }
            }
        }

        double pre = CalCost(correctOut);
        forward(layers[0]);
        double after = CalCost(correctOut);

        return after - pre;
    }

    /*
    double CalCost(std::vector<double> correctOut)
    {
        double cost=0;
        if(correctOut.size()!=layerSize[Size-1])
        {
            cout<<"Size Error"<<endl;
            return 0;
        }
        for(int i = 0;i < layerSize[Size-1];i++)
        {
            cost += (layers[Size-1][i]-correctOut[i])*(layers[Size-1][i]-correctOut[i])/2.0;
        }
        cost/=layerSize[Size-1];
        return cost;
    }*/


    void NNcore::printLayers() {
        for (int i = 0; i < size; i++) {
            cout << "Layer " << setw(2) << i << ": ";
            for (int j = 0; j < layerSize[i]; j++)
                cout << setw(10) << layers[i][j] << " ";

            cout << endl << endl;
        }
    }

    void NNcore::printW(int layerNumberToPrint) {
        cout << "W from layer " << layerNumberToPrint << " to " << layerNumberToPrint + 1 << ": " << endl;
        cout << "From↓   To-> ";
        for (int i = 0; i < layerSize[layerNumberToPrint + 1]; i++) {
            cout << setw(10) << i << " ";
        }
        cout << endl << endl;

        for (int i = 0; i < layerSize[layerNumberToPrint]; i++) {
            cout << "From " << setw(2) << i << " note:";
            for (int j = 0; j < layerSize[layerNumberToPrint + 1]; j++)
                cout << setw(10) << w[layerNumberToPrint][i][j] << " ";

            cout << endl << endl;
        }
    }

    int NNcore::choice() {
        double max = 0;
        int res;
        for (int i = 0; i < layerSize[size - 1]; i++) {
            if (layers[size - 1][i] > max) {
                max = layers[size - 1][i];
                res = i;
            }
        }
        return res;
    }

    void NNcore::init(std::vector<int> LayerS, double studyR) {
        size = LayerS.size();
        layerSize = LayerS;
        layers.resize(size);
        layersZ.resize(size);
        b.resize(size);
        w.resize(size - 1);

        studyRate = studyR;

        for (int i = 0; i < size; i++) {
            layers[i].resize(layerSize[i]);

            if (i != 0) {
                layersZ[i].resize(layerSize[i]);
                b[i].resize(layerSize[i]);
            }

            if (i < size - 1) {
                w[i].resize(layerSize[i]);
                for (int j = 0; j < layerSize[i]; j++) {
                    w[i][j].resize(layerSize[i + 1]);
                }
            }
        }

        cout << "RESIZED" << endl;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < layerSize[i]; j++) {
                layers[i][j] = getRandomDoubleNumber();
                if (i != 0) {
                    layersZ[i][j] = getRandomDoubleNumber();
                    b[i][j] = getRandomDoubleNumber();
                }
                if (i < size - 1) {
                    for (int k = 0; k < layerSize[i + 1]; k++) {
                        w[i][j][k] = getRandomDoubleNumber();
                    }
                }
            }
        }
    }

    void changeStudyRate(double rate) {
        studyRate = rate;
    }
} // NN
