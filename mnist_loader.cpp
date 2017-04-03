//
// Created by 孙桢波 on 2017/4/2.
//

#include "mnist_loader.h"

using namespace std;

int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = (unsigned char)( i & 255);
    ch2 = (unsigned char)((i >> 8) & 255);
    ch3 = (unsigned char)((i >> 16) & 255);
    ch4 = (unsigned char)((i >> 24) & 255);
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void load_images(const std::string &filename, std::vector<std::vector<double>> &images)
{
    ifstream fin(filename, std::ios::binary);
    if ( fin.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        unsigned char label;
        fin.read((char *) &magic_number, sizeof(magic_number));
        fin.read((char *) &number_of_images, sizeof(number_of_images));
        fin.read((char *) &n_rows, sizeof(n_rows));
        fin.read((char *) &n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);
        
        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;
        cout << "rows = " << n_rows << endl;
        cout << "cols = " << n_cols << endl;
        
        for ( int i = 0; i < number_of_images; i++ )
        {
            vector<double> tp;
            
            for ( int r = 0; r < n_rows; r++ )
            {
                for ( int c = 0; c < n_cols; c++ )
                {
                    unsigned char image = 0;
                    
                    fin.read((char *) &image, sizeof(image));
                    tp.push_back((double) image / 255.0);
                }
            }
            images.push_back(tp);
        }
    }
    fin.close();
}

void load_labels(const string &filename, std::vector<int> &labels)
{
    std::ifstream fin(filename, std::ios::binary);
    if (fin.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        fin.read((char*)&magic_number, sizeof(magic_number));
        fin.read((char*)&number_of_images, sizeof(number_of_images));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;
    
        for (int i = 0; i < number_of_images; i++)
        {
            unsigned char label = 0;
            fin.read((char*)&label, sizeof(label));
            labels.push_back(label + 0);
        }
    }
    fin.close();
}
    
    

void load_train_data(std::vector<Data> &train_data)
{
    std::vector<std::vector<double>> training_images;
    std::vector<int> training_labels;
    
    load_images("../data/train-images.idx3-ubyte", training_images);
    load_labels("../data/train-labels.idx1-ubyte", training_labels);
    
    for (size_t i = 0; i < training_images.size(); i++)
    {
        Eigen::VectorXd image(training_images[i].size());
        
        for (int j = 0; j < training_images[i].size(); j++)
        {
            image[j] = training_images[i][j];
        }
        train_data.push_back(Data(image, training_labels[i]));
    }
}

void load_test_data(std::vector<Data> &test_data)
{
    std::vector<std::vector<double>> testing_images;
    std::vector<int> testing_labels;
    
    load_images("../data/t10k-images.idx3-ubyte", testing_images);
    load_labels("../data/t10k-images.idx3-ubyte", testing_labels);
    
    for (size_t i = 0; i < testing_images.size(); i++)
    {
        Eigen::VectorXd image(testing_images[i].size());
        
        for (int j = 0; j < testing_images[i].size(); j++)
        {
            image[j] = testing_images[i][j];
        }
        test_data.push_back(Data(image, testing_labels[i]));
        
    }
}