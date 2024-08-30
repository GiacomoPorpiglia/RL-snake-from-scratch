#ifndef RMSPROP_H
#define RMS_PROP_H
#include <vector>
#include "types.h"

class RMSprop {

public: 
    int n_inputs;
    int n_outputs;
    double beta;
    double epsilon;
    std::vector<std::vector<double>> s;

    RMSprop() : n_inputs(0), n_outputs(0), beta(0.9), epsilon(1e-6) {}

    RMSprop(int n_in, int n_out, double b, double e) {
        n_inputs = n_in;
        n_outputs = n_out;
        beta = b;
        epsilon = e;
        s.resize(n_inputs);
        for(auto& x : s) {
            x.resize(n_outputs);
            std::fill(x.begin(), x.end(), 0);
        }
    }


    void optimize(std::vector<std::vector<double>>& gradients) {
        for(int i = 0; i < n_inputs; i++) {
            for(int j = 0; j < n_outputs; j++) {
                s[i][j] = s[i][j] * beta + (1 - beta) * gradients[i][j] * gradients[i][j];
                gradients[i][j] = gradients[i][j] / (sqrt(s[i][j]) + epsilon);
            }
        }
    }
};

#endif