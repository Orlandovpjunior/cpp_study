#include "logistc_regression.h"
#include "cmath"
#include <random>
#include <iostream>

logistc_regression::logistc_regression(int features, double lr)
    : num_features(features),
    learning_rate(lr),
    bias(0.0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-1.0,1.0);

    weights.resize(num_features);

    for (int i = 0; i <num_features; i++) {
        weights[i] = dis(gen);
    }
}

double logistc_regression::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

double logistc_regression::binary_cross_entropy(double predicted, double actual) {
    const double epsilon = 1e-15;

    predicted = std::max(epsilon, std::min(1.0 - epsilon, predicted));

    return - (actual * std::log(predicted) + (1.0 - actual) * std::log(1.0 - predicted));
}

void logistc_regression::train(const std::vector<std::vector<double> > &X, const std::vector<int> &y, int epochs) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        std::cerr << "Error: Invalid input data" << std::endl;
        return;
    }

    if (X[0].size() != static_cast<size_t> (num_features)) {
        std::cerr << "Error: Feature count mismatch" << std::endl;
        return;
    }

    for (int label : y) {
        if (label != 0 && label != 1) {
            std::cerr << "Error: Labels must be 0 or 1 for binary classification" << std::endl;
            return;
        }
    }

    std::cout << "Training logistc regression..." << std::endl;
    std::cout << "Features :" << num_features << ", Samples: " << X.size() << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        gradient_descent_step(X, y);

        if ((epoch +1) % 100 == 0) {
            double loss = compute_loss(X,y);
            double acc = accuracy(X, y);
            std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss << ", Accuracy: " << acc << std::endl;
        }
    }
    std::cout << "Training completed" << std::endl;
}

double logistc_regression::predict_proba(const std::vector<double> &X) {
    if (X.size() != static_cast<size_t>(num_features)) {
        std::cerr << "Error: Feature count mismatch in prediction" << std::endl;
        return 0.5;
    }

    double z = bias;

    for (int i = 0; i < num_features; i++) {
        z += weights[i] * X[i];
    }

    return sigmoid(z);
}

int logistc_regression::predict(const std::vector<double> &X) {
    return predict_proba(X) >= 0.5 ? 1 : 0;
}

std::vector<int> logistc_regression::predict_batch(const std::vector<std::vector<double> > &X) {
    std::vector<int> predictions;
    predictions.reserve(X.size());

    for (const auto& sample: X) {
        predictions.push_back(predict(sample));
    }
    return predictions;
}

std::vector<double> logistc_regression::predict_proba_batch(const std::vector<std::vector<double> > &X) {
    std::vector<double> predictions;
    predictions.reserve(X.size());

    for (const auto& sample: X) {
        predictions.push_back(predict_proba(sample));
    }

    return predictions;
}

double logistc_regression::accuracy(const std::vector<std::vector<double> > &X, const std::vector<int> &y) {
    size_t correct = 0;
    size_t n = X.size();

    for (size_t i = 0; i < n; i++) {
        int prediction = predict(X[i]);

        if (prediction == predict(X[i])) {
            correct++;
        }
    }

    return static_cast<double> (correct)/n;
}

double logistc_regression::compute_loss(const std::vector<std::vector<double> > &X, const std::vector<int> &y) {
    double total_loss = 0.0;
    size_t n = X.size();

    for (size_t i = 0; i < n; i++) {
        double prediction = predict_proba(X[i]);
        total_loss += binary_cross_entropy(prediction, y[i]);
    }

    return total_loss / n;
}

void logistc_regression::gradient_descent_step(const std::vector<std::vector<double>> &X, const std::vector<int> &y) {
    size_t n = X.size();

    std::vector<double> weight_gradients(num_features, 0.0);
    double bias_gradient = 0.0;
    for (size_t i = 0; i < n; i++) {
        double prediction = predict_proba(X[i]);
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

void logistc_regression::print_weights() {
    std::cout << "Bias: " << bias << std::endl;
    std::cout << "Weights: ";

    for (int i = 0; i < num_features; i++) {
        std::cout << weights[i] << " ";
    }

    std::cout << std::endl;
}
