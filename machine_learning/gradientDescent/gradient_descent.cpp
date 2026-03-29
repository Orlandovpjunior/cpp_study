#include "gradient_descent.h"

#include <iostream>
#include <random>

gradient_descent::gradient_descent(GradientDescentType gd_type, double lr, double tol, int max_iter)
    : type(gd_type),
    learning_rate(lr),
    tolerance(tol),
    max_iterations(max_iter),
    batch_size(32),
    use_momentum(false),
    momentum_factor(0.9),
    use_decay(false),
    decay_rate(0.9),
    decay_step(100)
{}

void gradient_descent::set_momentum(double factor) {
    if (factor <= 0 || factor >= 1.0) {
        std::cerr << "Warning: Momentum factior should be between 0 and 1" << std::endl;
        return;
    }

    use_momentum = true;
    momentum_factor = factor;
}

void gradient_descent::set_learning_rate_decay(double rate, int steps) {
    if (rate <= 0.0 || rate >= 1.0) {
        std::cerr << "Warning: Decay rate should be between 0 and 1" << std::endl;
        return;
    }

    if (steps <= 0) {
        std::cerr << "Warning: Decay steps should be positive" << std::endl;
        return;
    }

    use_decay=true;
    decay_rate =rate;
    decay_step = steps;
}

void gradient_descent::set_batch_size(int size) {
    if (size <= 0) {
        std::cerr << "Warning: Batch size should be positive" << std::endl;
        return;
    }

    batch_size=size;
}

std::vector<double> gradient_descent::optimize(const std::vector<double> &initial_params, ObjectiveFunction objective_function, GradientFunction gradient_function) {
    switch (type) {
        case GradientDescentType::BATCH:
            return batch_gradient_descent(initial_params, objective_function,gradient_function);
        case GradientDescentType::STOCHASTIC:
            return stochastic_gradient_descent(initial_params,objective_function,gradient_function);
        case GradientDescentType::MINI_BATCH:
            return mini_batch_gradient_descent(initial_params,objective_function,gradient_function);
        default:
            return initial_params;

    }
}

std::vector<double> gradient_descent::batch_gradient_descent(const std::vector<double> &initial_params, ObjectiveFunction objective_function, GradientFunction gradient_function) {
    std::vector<double>params = initial_params;
    previous_update = std::vector<double>(params.size(), 0.0);

    std::cout << "Starting Batch Gradient Descent..." << std::endl;
    print_params(params);

    double current_lr = learning_rate;

    for (int iter = 0; iter < max_iterations; iter++) {
        std::vector<double> gradient = gradient_function(params);

        if (use_decay && iter > 0 && iter % decay_step == 0) {
            current_lr *= decay_step;
            std::cout << "Learning rate decayed to: " << current_lr << std::endl;
        }
        if (use_momentum) {
            for (size_t i =0; i < gradient.size(); i++) {
                gradient[i] = momentum_factor * previous_update[i] + current_lr + gradient[i];
                previous_update[i] = gradient[i];
            }
        }else {
            for (size_t i =0; i < gradient.size(); i++) {
                gradient[i] *= current_lr;
            }
        }

        double max_change = 0.0;

        for (size_t i = 0; i< params.size(); i++) {
            double change = gradient[i];
            params[i] -= change;
            max_change = std::max(max_change, std::abs(change));
        }

        if (max_change < tolerance) {
            std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
        }

        if ((iter + 1) % 100 == 0) {
            double objective_value = objective_function(params);
            std::cout << "Iteration " << iter + 1 << ", Objective: " << objective_value
                << ", Max change: " << max_change << std::endl;
        }
    }

    std::cout << "Final parameters: ";
    print_params(params);
    return  params;
}

std::vector<double> gradient_descent::mini_batch_gradient_descent(const std::vector<double> &initial_params, ObjectiveFunction objective_function, GradientFunction gradient_function) {
    std::vector<double> params = initial_params;
    previous_update = std::vector<double>(params.size(), 0.0);

    std::cout << "Starting Mini-Batch Gradient Descent..." << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Initial parameters: ";
    print_params(params);

    double current_lr = learning_rate;

    for (int iter = 0; iter < max_iterations; iter++) {
        std::vector<double> gradient = gradient_function(params);

        if (use_decay && iter > 0 && iter % decay_step == 0) {
            current_lr *= decay_rate;
            std::cout << "Learning rate decayed to: " << current_lr << std::endl;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> noise(0.0,0.5); // mean, stddev
        for (size_t i = 0; i < gradient.size(); i++) {
            gradient[i] += noise(gen) / std::sqrt(batch_size);
        }

        if (use_momentum) {
            for (size_t i = 0; i < gradient.size(); i++) {
                gradient[i] = momentum_factor * previous_update[i] + current_lr * gradient[i];
                previous_update[i] = gradient[i];
            }
        }else {
            for (size_t i = 0; i < gradient.size(); i++) {
                gradient[i] *= current_lr;
            }
        }

        double max_change = 0.0;
        for (size_t i = 0; i< params.size(); i++) {
            double change = gradient[i];
            params[i] -= change;
            max_change = std::max(max_change, std::abs(change));
        }
        if (max_change < tolerance) {
            std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
            break;
        }

        if ((iter + 1) % 200 == 0) {
            double objective_value = objective_function(params);
            std::cout << "Iteration " << iter + 1 << ", Objective " << objective_value
            << ", max change: " << max_change << std::endl;
        }
    }

    std::cout << "Final Parameters: ";
    print_params(params);
    return  params;

}

std::vector<double> gradient_descent::stochastic_gradient_descent(const std::vector<double> &initial_params, ObjectiveFunction objective_function, GradientFunction gradient_function) {
    std::vector<double>params = initial_params;
    previous_update = std::vector<double>(params.size(), 0.0);

    std::cout << "Starting Stochastic Gradient Descent..." << std::endl;
    std::cout << "Initial parameters: ";
    print_params(params);

    double current_lr = learning_rate;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int iter = 0; iter < max_iterations; iter++) {
        std::vector<double> gradient = gradient_function(params);

        if (use_decay && iter > 0 && iter % decay_step == 0) {
            current_lr *= decay_rate;
            std::cout << "Learning rate decayed to: " << current_lr << std::endl;
        }

        std::normal_distribution<> noise(0.0,0.1); // mean, stddev
        for (size_t i = 0; i < gradient.size(); i++) {
            gradient[i] += noise(gen);
        }

        if (use_momentum) {
            for (size_t i = 0; i < gradient.size(); i++) {
                gradient[i] = momentum_factor * previous_update[i] + current_lr * gradient[i];
                previous_update[i] = gradient[i];
            }
        }else {
            for (size_t i = 0; i < gradient.size(); i++) {
                gradient[i] *= current_lr;
            }
        }

        double max_change = 0.0;
        for (size_t i = 0; i< params.size(); i++) {
            double change = gradient[i];
            params[i] -= change;
            max_change = std::max(max_change, std::abs(change));
        }
        if (max_change < tolerance) {
            std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
            break;
        }

        if ((iter + 1) % 500 == 0) {
            double objective_value = objective_function(params);
            std::cout << "Iteration " << iter + 1 << ", Objective " << objective_value
            << ", max change: " << max_change << std::endl;
        }
    }

    std::cout << "Final Parameters: ";
    print_params(params);
    return  params;
}

gradient_descent::OptimizationResult gradient_descent::optimize_with_history(const std::vector<double> &initial_params, ObjectiveFunction objective_function, GradientFunction gradient_function) {
    OptimizationResult result;
    result.final_params = initial_params;
    result.converged = false;
    result.iteration_performed = 0;

    std::vector<double> params = initial_params;
    previous_update = std::vector<double>(params.size(), 0.0);

    double current_lr = learning_rate;

    result.objective_history.push_back(objective_function(params));

    for (int iter = 0; iter < max_iterations; iter++) {
        std::vector<double> gradient = gradient_function(params);

        if (use_decay && iter % decay_step == 0) {
            current_lr *= decay_rate;
        }

        if (use_momentum) {
            for (size_t i = 0; i < gradient.size(); i++) {
                gradient[i] = momentum_factor * previous_update[i] + current_lr * gradient[i];
                previous_update[i] = gradient[i];
            }
        }else {
            for (size_t i = 0; i < params.size(); i++) {
                gradient[i] *= current_lr;
            }
        }

        double max_change = 0.0;
        for (size_t i = 0; i < gradient.size(); i++) {
            double change = gradient[i];
            params[i] -= change;
            max_change = std::max(max_change, std::abs(change));
        }

        result.iteration_performed = iter + 1;
        result.objective_history.push_back(objective_function(params));

        if (max_change < tolerance) {
            result.converged = true;
            break;
        }
    }

    result.final_params = params;
    result.final_objective_value = objective_function(params);
    return result;
}

void gradient_descent::print_params(const std::vector<double> &params) {
    std::cout << "[";

    for (size_t i = 0; i < params.size(); i++) {
        std::cout << params[i];
        if (i < params.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void gradient_descent::print_history(const std::vector<double> &history) {
    std::cout << "Objective function history: " << std::endl;
    for (size_t i = 0; i < history.size(); i++) {
        if (i % 10 == 0) {
            std::cout << "Step " << i << ": " << history[i] << std::endl;
        }
    }
}
