//
// Created by 孙桢波 on 2017/4/2.
//

#ifndef MNIST_NETWORK_H
#define MNIST_NETWORK_H

#include "layer.h"
#include "Data.h"
#include <vector>

class Network
{
private:
    std::vector<Layer *> _layers;
    int sizes;
public:
    Network(std::vector<size_t> &size_of_layer);
    Eigen::VectorXd feedforward(const Eigen::VectorXd &input);
    void train(const std::vector<Data> &train_data, int epochs, int mini_batch_size, double eta, const std::vector<Data> &test_data);
    void update_mini_batch(const std::vector<Data> &mini_batch, double &eta);
    void evaluate(const std::vector<Data> &test_data);
    
};


#endif //MNIST_NETWORK_H
