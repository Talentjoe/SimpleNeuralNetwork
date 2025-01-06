//
// Created by lenovo on 25-1-6.
//

#ifndef NN_H
#define NN_H

#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip>

namespace NN {
    class NNcore {
    public:
        std::vector<double> forward(std::vector<double> inNums, bool printRes = false);

        double pushBack(std::vector<double> correctOut);

        double CalCost(std::vector<double> correctOut);

        void changeStudyRate(double rate);

        void init(std::vector<int> LayerS, double studyR);

        int choice();

        void printLayers();

        void printW(int layerNumberToPrint);
    };
}

#endif //NN_H
