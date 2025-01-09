//
// Created by lenovo on 25-1-6.
//

#ifndef NN_H
#define NN_H

#include <vector>
#include <string>

namespace NN {
    class NNcore {
        int size;
        double studyRate;
        double dropRate;

        std::vector<std::vector<double> > layers;
        std::vector<std::vector<double> > layersZ;
        std::vector<int> layerSize;

        std::vector<std::vector<double> > b;

        std::vector<std::vector<std::vector<double> > > w;

    public:
        double train(std::vector<std::vector<double> > inNums, std::vector<int> correctOut, bool getAcc = false);

        double test(std::vector<std::vector<double> > inNums, std::vector<int> correctOut);

        std::vector<double> forward(std::vector<double> inNums, bool printRes = false);

        double backpropagation(std::vector<double> correctOut);

        double CalCost(std::vector<double> correctOut);

        void changeStudyRate(double rate);

        void init(const std::vector<int> &LayerS, double studyR);

        int choice();

        void printLayers();

        static void printLayers(const NNcore &nn);

        void printW(int layerNumberToPrint);

        static void printW(const NNcore &nn, int layerNumberToPrint);

        static void save(const NNcore &nn, std::string path);

        void init(const std::string &path, double studyRate);
    };
}

#endif //NN_H
