#ifndef AMSGRAD_H
#define AMSGRAD_H
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "types.h"

class Adam {

    public:
        int n_inputs;
        int n_outputs;
        double beta1;
        double beta2;
        double epsilon;

        double beta1_to_t;
        double beta2_to_t;
        int t;
        int layerIdx;
        std::vector<std::vector<double>> m_w;
        std::vector<std::vector<double>> v_w;

        
        std::vector<double> m_b;
        std::vector<double> v_b;

        Adam() : n_inputs(0), n_outputs(0), beta1(0.9), beta2(0.999), epsilon(1e-8) {}

        Adam(int n_in, int n_out, double b1, double b2, double e, int time, int idx) {
            n_inputs = n_in;
            n_outputs = n_out;
            beta1 = b1;
            beta2 = b2;
            epsilon = e;
            t = time;

            layerIdx = idx;

            beta1_to_t = beta1;
            beta2_to_t = beta2;

            for(int i = 0; i < time; i++) {
                beta1_to_t*=beta1;
                beta2_to_t*=beta2;
            }
            
            t = 0;

            m_w.resize(n_inputs);
            v_w.resize(n_inputs);


            for (auto &x : m_w) {
                x.resize(n_outputs);
                std::fill(x.begin(), x.end(), 0);
            }
            for (auto &x : v_w) {
                x.resize(n_outputs);
                std::fill(x.begin(), x.end(), 0);
            }

            m_b.resize(n_outputs);
            v_b.resize(n_outputs);
            std::fill(m_b.begin(), m_b.end(), 0);
            std::fill(v_b.begin(), v_b.end(), 0);
        }

        void optimize(std::vector<std::vector<double>> &gradientsW, std::vector<double>& gradientsB) {
            t++;
            for (int i = 0; i < n_inputs; i++) {
                for (int j = 0; j < n_outputs; j++) {
                    m_w[i][j] = m_w[i][j]*beta1 + (1-beta1)*gradientsW[i][j];
                    v_w[i][j] = v_w[i][j]*beta2 + (1-beta2)*gradientsW[i][j]*gradientsW[i][j];
                    double m_hat = m_w[i][j]/(1-beta1_to_t);
                    double v_hat = v_w[i][j]/(1-beta2_to_t);
                    gradientsW[i][j] = m_hat / (sqrt(v_hat) + epsilon);
                }
            }

            for(int i = 0; i < n_outputs; i++) {
                m_b[i] = m_b[i] * beta1 + (1 - beta1) * gradientsB[i];
                v_b[i] = v_b[i] * beta2 + (1 - beta2) * gradientsB[i] * gradientsB[i];
                double m_hat = m_b[i] / (1 - beta1_to_t);
                double v_hat = v_b[i] / (1 - beta2_to_t);
                gradientsB[i] = m_hat / (sqrt(v_hat) + epsilon);
            }

            if(beta1_to_t < 1e-20) beta1_to_t = 0;
            if(beta2_to_t < 1e-20) beta2_to_t = 0;

            beta1_to_t *= beta1;
            beta2_to_t *= beta2;
        }

        void load(std::string path) {
            std::ifstream inFile(path + "/layer" + std::to_string(layerIdx) + "_adam.txt");
            if (!inFile) {
                std::cerr << "Error: Could not open file " << path << "/layer" << layerIdx << "_adam.txt for reading. Initializing new Adam configuration..." << std::endl;
                return;
            }

            std::string line;

            // Read the first line for beta1, beta2, beta1_to_t, beta2_to_t, and t
            if (std::getline(inFile, line)) {
                std::istringstream iss(line);
                if (!(iss >> beta1 >> beta2 >> beta1_to_t >> beta2_to_t >> t)) {
                    std::cerr << "Error: Failed to read the initial Adam configuration values." << std::endl;
                    return;
                }
            }
            else {
                std::cerr << "Error: The file is empty or malformed." << std::endl;
                return;
            }

            std::cout << "Values: " << beta1 << " " << beta2 << " " << beta1_to_t << " " << beta2_to_t << " " << t << "\n";

            // Read m values
            for (int i = 0; i < n_inputs; i++) {
                if (!std::getline(inFile, line)) {
                    std::cerr << "Error: Not enough lines to read m values." << std::endl;
                    return;
                }
                std::istringstream iss(line);
                for (int j = 0; j < n_outputs; ++j) {
                    if (!(iss >> m_w[i][j])) {
                        std::cerr << "Error: Failed to read m value at position [" << i << "][" << j << "]" << std::endl;
                        return;
                    }
                }
            }

            // Read v values
            for (int i = 0; i < n_inputs; i++) {
                if (!std::getline(inFile, line)) {
                    std::cerr << "Error: Not enough lines to read v values." << std::endl;
                    return;
                }
                std::istringstream iss(line);
                for (int j = 0; j < n_outputs; ++j) {
                    if (!(iss >> v_w[i][j])) {
                        std::cerr << "Error: Failed to read v value at position [" << i << "][" << j << "]" << std::endl;
                        return;
                    }
                }
            }

            if (std::getline(inFile, line)) {
                std::istringstream iss(line);
                for(int i = 0; i < n_outputs; i++) {
                    if(!(iss >> m_b[i])) {
                        std::cerr << "Error: Failed to read m value at position [" << i << "]" << std::endl;
                    }
                }
            } else {
                std::cerr << "Error: Failed to read m values for the biases." << std::endl;
            }

            if (std::getline(inFile, line)) {
                std::istringstream iss(line);
                for(int i = 0; i < n_outputs; i++) {
                    if(!(iss >> v_b[i])) {
                        std::cerr << "Error: Failed to read v value at position [" << i << "]" << std::endl;
                    }
                }
            } else {
                std::cerr << "Error: Failed to read v values for the biases." << std::endl;
            }


            inFile.close();
            std::cout << "Successfully loaded Adam for layer " << layerIdx << " from file." << std::endl;

        }

        void save(std::string path) {
            std::ofstream outFile(path + "/layer" + std::to_string(layerIdx) + "_adam.txt");

            if (!outFile) {
                std::cerr << "Error: Could not open file " << path << "/layer" << layerIdx << "_adam.txt for writing." << std::endl;
                return;
            }

            outFile << beta1 << " " << beta2 << " " << beta1_to_t << " " << beta2_to_t << " " << t << std::endl;

            for (int i = 0; i < n_inputs; i++) {
                for (int j = 0; j < n_outputs; j++) {
                    outFile << m_w[i][j] << " ";
                }
                outFile << std::endl;
            }

            for (int i = 0; i < n_inputs; i++) {
                for (int j = 0; j < n_outputs; j++) {
                    outFile << v_w[i][j] << " ";
                }
                outFile << std::endl;
            }

            for(int i = 0; i < n_outputs; i++) {
                outFile << m_b[i] << " ";
            }
            outFile << std::endl;

            for (int i = 0; i < n_outputs; i++) {
                outFile << v_b[i] << " ";
            }
            outFile << std::endl;

            outFile.close();
            std::cout << "Adam parameters saved successfully\n";
        }
};

#endif