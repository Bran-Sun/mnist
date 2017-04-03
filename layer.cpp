//
// Created by 孙桢波 on 2017/4/2.
//

#include "layer.h"

Layer::Layer(size_t input_dim, size_t output_dim)
{
    weight_matrix.setRandom(output_dim, input_dim);
    bias_vector.setRandom(output_dim);
    activation.setZero(output_dim);
    z.setZero(output_dim);
    error.setZero(output_dim);
    
    delta_b.setZero(output_dim);
    delta_weight.setZero(output_dim, input_dim);
    derive_z.setZero(output_dim);
    
}

void Layer::init_delta()
{
    delta_b.setZero();
    delta_weight.setZero();
}

void Layer::feedforward(const Eigen::VectorXd &input_data)
{
    z = weight_matrix * input_data + bias_vector;
    activation = sigmoid(z);
    derive_z = derivesigmoid(z);
}

void Layer::back_propogate(const Eigen::VectorXd &input, const Eigen::VectorXd &former_activation)
{
    error = hadamard(input, derive_z);
    delta_b += error;
    delta_weight += error * former_activation.transpose();
    
}

void Layer::change_weight_and_biases(double eta, int mini_number)
{
    weight_matrix -= (eta/mini_number) * delta_weight;
    bias_vector -= (eta/mini_number) * delta_b;
}


Eigen::VectorXd Layer::sigmoid(const Eigen::VectorXd &input_data)
{
    Eigen::VectorXd answer = input_data;
    for (int i = 0; i < input_data.size(); i++)
    {
        answer[i] = 1.0 / (1+exp(-input_data[i]));
    }
    return answer;
}

Eigen::VectorXd Layer::derivesigmoid(const Eigen::VectorXd &input_data)
{
    Eigen::VectorXd answer = input_data;
    for (int i = 0; i < input_data.size(); i++)
    {
        answer[i] = exp(-input_data[i]) / ((1 + exp(-input_data[i])) * (1 + exp(-input_data[i])));
    }
    return answer;
}

Eigen::VectorXd Layer::hadamard(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
{
    Eigen::VectorXd answer = x1;
    for (int i = 0; i < x1.size(); i++)
    {
        answer[i] = x1[i] * x2[i];
    }
    return answer;
}

void Layer::print()
{
    std::cout << "weight_matrix: " << std::endl;
    std::cout << weight_matrix << std::endl;
    std::cout << "bias_vector:" << std::endl;
    std::cout << bias_vector << std::endl;

}
