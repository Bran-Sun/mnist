//
// Created by 孙桢波 on 2017/4/2.
//

#ifndef MNIST_LAYER_H
#define MNIST_LAYER_H

#include <iostream>
#include <eigen3/Eigen/Core>

class Layer
{
private:
    Eigen::MatrixXd weight_matrix;  //weight of the layer
    Eigen::VectorXd bias_vector;   //biases of the layer
    
    Eigen::VectorXd activation;
    Eigen::VectorXd z;
    Eigen::VectorXd derive_z;
    Eigen::VectorXd error;
    
    Eigen::MatrixXd delta_weight;
    Eigen::VectorXd delta_b;
    
public:
    Layer(size_t input_dim, size_t output_dim);
    
    void init_delta();
    void feedforward(const Eigen::VectorXd &input_data);
    void back_propogate(const Eigen::VectorXd &input, const Eigen::VectorXd &former_activation);
    void change_weight_and_biases(double eta, int mini_number);
    
    Eigen::VectorXd sigmoid(const Eigen::VectorXd &input_data);
    Eigen::VectorXd derivesigmoid(const Eigen::VectorXd &input_data);
    Eigen::VectorXd hadamard(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    
    Eigen::VectorXd get_activation()
    {
        return activation;
    }
    
    Eigen::VectorXd getback()
    {
        return weight_matrix.transpose() * error;
    }
    
    void print();
    
    friend class Network;
    
    
};


#endif //MNIST_LAYER_H
