//
// Created by 孙桢波 on 2017/4/2.
//

#include "network.h"


Network::Network(std::vector<size_t> &size_of_layer)
{
    sizes = (int) size_of_layer.size();
    sizes--;
    for ( size_t i = 1; i < size_of_layer.size(); i++ )
    {
        _layers.push_back(new Layer(size_of_layer[i-1], size_of_layer[i]));
    }
}

Eigen::VectorXd Network::feedforward(const Eigen::VectorXd &input)
{
    Eigen::VectorXd output = input;
    for (auto layer : _layers)
    {
        output = layer->sigmoid(output);
    }
    return output;
}

void Network::train(const std::vector<Data> &train_data, int epochs, int mini_batch_size, double eta,
                    const std::vector<Data> &test_data)
{
    std::vector<Data> data = train_data;
    
    for (int i = 0; i < epochs; i++)
    {
        std::random_shuffle(data.begin(), data.end());
        
        for (int j = 0; j < data.size(); j += mini_batch_size)
        {
            std::vector<Data> mini_batch;
            for ( int k = 0; k < mini_batch_size; k++ )
            {
                mini_batch.push_back(data[j + k]);
            }
            update_mini_batch(mini_batch, eta);
        
        }
        evaluate(test_data);
    }
    
}

void Network::update_mini_batch(const std::vector<Data> &mini_batch, double &eta)
{
    for (int i = 0 ; i < _layers.size(); i++)
    {
        _layers[i]->init_delta();
    }
    
    for (Data batch : mini_batch)
    {
        _layers[0]->feedforward(batch.get_data());
        for (int i = 1 ; i < _layers.size(); i++)
        {
            _layers[i]->feedforward(_layers[i-1]->get_activation());
        }
        Eigen::VectorXd derive_c(10);
        for (int i = 0; i < 10; i++)
        {
            int tem;
            if (batch.getlabel() == i) tem = 1;
            else tem = 0;
            derive_c[i] = _layers[_layers.size()-1]->get_activation()[i] - tem;
        }
        for (int i =int(_layers.size() - 1); i > 0; i--)
        {
            _layers[i]->back_propogate(derive_c, _layers[i-1]->get_activation());
            derive_c.setZero(_layers[i]->getback().size());
            derive_c = _layers[i]->getback();
        }
        _layers[0]->back_propogate(derive_c, batch.get_data());
    }
    for (int i = 0 ; i < _layers.size(); i++)
    {
        _layers[i]->change_weight_and_biases(eta,(int)mini_batch.size());
    }
}

void Network::evaluate(const std::vector<Data> &test_data)
{
    int cnt = 0;
    int label = 0;
    double active;
    for ( int i = 0; i < test_data.size(); i++ )
    {
         label = -1;
        active = 0.0;
        _layers[0]->feedforward(test_data[i].get_data());
        for (int j = 1 ; j < _layers.size(); j++)
        {
            _layers[j]->feedforward(_layers[j-1]->get_activation());
        }
        Eigen::VectorXd answer = _layers[_layers.size() - 1]->get_activation();
        for (int j = 0; j < 10; j++)
        {
            if (active < answer[j])
            {
                label = j;
                active = answer[j];
            }
        }
        if (label == test_data[i].getlabel()) cnt++;
    }
    std::cout << "evaluate result: " << double(cnt)/test_data.size() << std::endl;
}