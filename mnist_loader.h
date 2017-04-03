//
// Created by 孙桢波 on 2017/4/2.
//

#ifndef MNIST_MNIST_LOADER_H
#define MNIST_MNIST_LOADER_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include "data.h"

void load_images(const std::string &filename, std::vector<std::vector<double>> &images);
void load_labels(const std::string &filename, std::vector<int> &labels);
void load_train_data(std::vector<Data> &train_data);
void load_test_data(std::vector<Data> &test_data);


#endif //MNIST_MNIST_LOADER_H
