//
// Created by ovpju on 25/03/2026.
//

#ifndef CPP_STUDY_LINEAR_REGRESSION_H
#define CPP_STUDY_LINEAR_REGRESSION_H


#include <vector>

class linear_regression {
    std::vector<double> weights;
    double bias;
    double learning_rate;
    int num_features;

public:
    linear_regression(int features, double lr = 0.01);

    void train(const std::vector<std::vector<double>> &X,
        const std::vector<double> &y,
        int epochs= 1000);

    double predict(const std::vector<double> &X);
    std::vector<double> predict_batch(const std::vector<std::vector<double>> &X);

    double mean_squared_error(const std::vector<std::vector<double>> &X,
        const std::vector<double> &y);

    void print_weights();

    void gradient_descent_step(const std::vector<std::vector<double>> &X,
        const std::vector<double> &y);


};

#endif //CPP_STUDY_LINEAR_REGRESSION_H