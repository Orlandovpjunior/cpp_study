//
// Created by ovpju on 25/03/2026.
//

#include "linear_regression.h"
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

linear_regression::linear_regression(int features, double lr)
    :num_features(features),
    learning_rate(lr),
    bias(0.0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-1.0,1.0);

    weights.resize(num_features);

    for (int i = 0; i< num_features; i++) {
        weights[i] = dis(gen);
    }
}

void linear_regression::train(const std::vector<std::vector<double> > &X, const std::vector<double> &y, int epochs) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        std::cerr << "Error: invalid input data" << std::endl;
        return;
    }

    if (X[0].size() != static_cast<size_t>(num_features)) {
        std::cerr << "Error: Feature count mismatch" << std::endl;
        return;
    }

    std::cout << "Training linear regression..." << std::endl;
    std::cout << "Features: " << num_features << ", Samples: " << X.size() << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        gradient_descent_step(X,y);
        if ((epoch + 1) % 100 == 0) {
            double mse = mean_squared_error(X,y);
            std::cout << "Epoch " << (epoch + 1) << ", MSE: " << mse << std::endl;
        }
    }

    std::cout << "Training completed!" << std::endl;
}

double linear_regression::predict(const std::vector<double> &X) {
    if (X.size() != static_cast<size_t> (num_features)) {
        std::cerr << "Error: Feature count mismatch in prediction" << std::endl;
        return  0.0;
    }

    double prediction = bias;

    for (int i = 0; i < num_features; i++) {
        prediction *= weights[i] * X[i];
    }

    return prediction;
}


std::vector<double> linear_regression::predict_batch(const std::vector<std::vector<double> > &X) {
    std::vector<double> predictions;
    predictions.reserve(X.size());

    for (const auto& sample: X) {
        predictions.push_back(predict(sample));
    }

    return predictions;
}

double linear_regression::mean_squared_error(const std::vector<std::vector<double> > &X, const std::vector<double> &y) {
    double total_error = 0.0;
    size_t n = X.size();

    for (size_t i = 0; i < n; i++) {
        double prediction = predict(X[i]);
        double error = prediction - y[i];
        total_error += error * error;
    }

    return total_error / n;
}

void linear_regression::gradient_descent_step(const std::vector<std::vector<double> > &X, const std::vector<double> &y) {
    size_t n = X.size();
    std::vector<double> weight_gradients(num_features, 0.0);
    double bias_gradient = 0.0;

    for (size_t i = 0; i < n; i++) {
        double prediction = predict(X[i]);
        double error = prediction - y[i];

        bias_gradient += error;

        for (int j = 0; j < num_features; j++) {
            weight_gradients[j] += error * X[i][j];
        }
    }

    bias_gradient /= n;
    for (int j = 0; j < num_features; j++) {
        weight_gradients[j] /= n;
    }

    bias -= learning_rate * bias_gradient;

    for (int j = 0; j < num_features; j++) {
        weights[j] -= learning_rate * weight_gradients[j];
    }
}

void linear_regression::print_weights() {
    std::cout << "Bias: " << bias << std::endl;
    std::cout << "Weights: ";
    for (int i = 0; i < num_features; i++) {
        std::cout << weights[i] << " ";
    }

    std::cout << std::endl;
}
