//
// Created by 孙桢波 on 2017/4/2.
//

#include "function.h"
#include <cmath>

Eigen::VectorXd Function::sigmoid(const Eigen::VectorXd &input)
{
    Eigen::VectorXd answer = input;
    for (int i = 0; i < input.size(); i++)
    {
        answer[i] = 1.0 / (1+exp(-input[i]));
    }
    return answer;
}