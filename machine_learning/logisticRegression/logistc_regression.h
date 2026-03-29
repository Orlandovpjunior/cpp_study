//
// Created by ovpju on 25/03/2026.
//

#ifndef CPP_STUDY_LOGISTC_REGRESSION_H
#define CPP_STUDY_LOGISTC_REGRESSION_H
#include <vector>



class logistc_regression {

    std::vector<double> weights;
    double bias;
    double learning_rate;
    int num_features;

public:
    logistc_regression(int features, double lr = 0.01);

    static double sigmoid(double z);
    static double binary_cross_entropy(double predicted, double actual);

    void train(const std::vector<std::vector<double>> &X,
        const std::vector<int> &y,
        int epochs= 1000);

    double predict_proba(const std::vector<double> &X);

    int predict(const std::vector<double> &X);

    std::vector<int> predict_batch(const std::vector<std::vector<double>> &X);

    std::vector<double> predict_proba_batch(const std::vector<std::vector<double>> &X);

    double accuracy (const std::vector<std::vector<double>> &X,
        const std::vector<int> &y);

    double compute_loss(const std::vector<std::vector<double>> &X,
        const std::vector<int> &y);

    void print_weights();

    void gradient_descent_step(const std::vector<std::vector<double>> &X,
        const std::vector<int> &y);

};

#endif //CPP_STUDY_LOGISTC_REGRESSION_H