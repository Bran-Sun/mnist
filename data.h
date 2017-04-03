//
// Created by 孙桢波 on 2017/4/2.
//

#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Core>

class Data
{
private:
    int label;
    Eigen::VectorXd data;
public:
    Data() = default;
    Data(const Eigen::VectorXd &_data, int _label): label(_label), data(_data)
    {}
    const Eigen::VectorXd & get_data() const {
        return data;
    }
    int getlabel() const {
        return label;
    }
};


#endif //MNIST_DATA_H
