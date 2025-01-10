//
// Created by lenovo on 25-1-6.
//

#ifndef NN_H
#define NN_H

#include <vector>
#include <string>

namespace NN {
    class NNcore {
        int size; // size of layers
        double studyRate; // study rate
        bool addDropout; // add dropout or not
        double dropOutRate; // dropout rate

        std::vector<std::vector<double> > layers; //value of each layer
        std::vector<std::vector<double> > layersZ; //value of each layer before sigmoid
        std::vector<int> layerSize; // size of each layer

        std::vector<std::vector<double> > b; // bias
        std::vector<std::vector<std::vector<double> > > w; // weight

    public:
        /**
         * Start training process, modify current NN function
         * @param inNums input data, each element in the outer vector is a piece of data,
         * and each element in the inner vector is a feature, the size of the feature
         * should be equal to the size of the first layer
         * @param correctOut correct output data, each element in the vector is a piece of
         * data, and the value should be less than the size of the last layer
         * @param getAcc a bool which indicate whether to get the accuracy of the training
         * @return if getAcc is true, return the accuracy of the training, otherwise return -1
         */
        double train(std::vector<std::vector<double> > inNums, std::vector<int> correctOut, bool getAcc = false);

        /**
         * Test the NN with the given data
         * @param inNums input data, each element in the outer vector is a piece of data,
         * and each element in the inner vector is a feature, the size of the feature
         * should be equal to the size of the first layer
         * @param correctOut correct output data, each element in the vector is a piece of
         * data, and the value should be less than the size of the last layer
         * @return the accuracy of the test
         */
        double test(std::vector<std::vector<double> > inNums, std::vector<int> correctOut);

        /**
         * Forward propagation
         * @param inNums input data, each element in the vector is a feature, the size of the feature
         * @param printRes if true, print the running process
         * @return the output of the last layer
         */
        std::vector<double> forward(std::vector<double> inNums, bool printRes = false);

        /**
         * Back propagation, modify the current NN function. Should first do forward propagation which
         * gives the output of current framework and modify based on that
         * @param correctOut Expect output of the last layer, the size should be equal to the size of the last layer
         * @return the cost after modify framework
         */
        double backpropagation(std::vector<double> correctOut);

        /**
         * Calculate the cost of the current framework, with current data.
         * @param correctOut Expected output of the last layer, the size should be equal to the size of the last layer
         * @return the cost.
         */
        double CalCost(std::vector<double> correctOut);

        /**
         * Change the study rate of the NN
         * @param rate the new study rate
         */
        void changeStudyRate(double rate);

        /**
         * Change the dropout rate of the NN
         * @param rate the new dropout rate
         */
        void changeDropOutRate(double rate);

        /**
         * process the dropout, modify the current NN function
         */
        void dropSome();

        /**
         * Get the choice of the NN, which is the index of the output layer with the largest value
         * @return the choice number
         */
        int choice();

        /**
         * Print the weight of the given layer
         * @param layerNumberToPrint the layer number to print
         */
        void printLayers();

        /**
         * Print the weight of the given layer
         * @param nn the NN to print
         * @param layerNumberToPrint the layer number to print
         */
        static void printLayers(const NNcore &nn);

        /**
         * Print the weight of the given layer
         * @param layerNumberToPrint the layer number to print
         */
        void printW(int layerNumberToPrint);

        /**
         * Print the weight of the given layer
         * @param nn the NN to print
         * @param layerNumberToPrint the layer number to print
         */
        static void printW(const NNcore &nn, int layerNumberToPrint);

        /**
         * Save the nn framework to the given path
         * @param nn the framework to save
         * @param path the path to save
         */
        static void save(const NNcore &nn, std::string path);

        /**
         * Initialize the NN with the given path, load the framework from the path
         * @param path the target framework path
         * @param studyRate the study rate
         * @param drRate the dropout rate
         */
        void init(const std::string &path, double studyRate, double drRate = -1);

        /**
         * Initialize the NN with the given layer size, study rate and dropout rate
         * @param LayerS the size of each layer, each number indicates the size of the layer
         * @param studyR the study rate
         * @param drRate the dropout rate
         */
        void init(const std::vector<int> &LayerS, double studyR, double drRate = -1);
    };
}

#endif //NN_H
