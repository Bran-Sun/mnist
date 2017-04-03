#include <iostream>
#include "layer.h"
#include "network.h"
#include "mnist_loader.h"
#include "data.h"

int main()
{
    std::vector<size_t> sizes;
    sizes.push_back(784);
    sizes.push_back(30);
    sizes.push_back(10);
    
    Network network(sizes);
    std::vector<Data> train_data;
    std::vector<Data> test_data;
    load_train_data(train_data);
    load_test_data(test_data);
    
    network.train(train_data,30,20,1.0,test_data);
    
    
    return 0;
}