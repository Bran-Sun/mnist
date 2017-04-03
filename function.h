//
// Created by 孙桢波 on 2017/4/2.
//

#ifndef MNIST_FUNCTION_H
#define MNIST_FUNCTION_H

#include <eigen3/Eigen/Core>

class Function
{
public:
    Eigen::VectorXd sigmoid(const Eigen::VectorXd &input);
};


#endif //MNIST_FUNCTION_H
