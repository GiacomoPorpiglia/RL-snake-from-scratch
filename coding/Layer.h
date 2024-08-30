#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include "types.h"
#include "RMSprop.h"
#include "Adam.h"


class Layer {


    public:
        int n_inputs;
        int n_outputs;
        int layerIdx;
        LayerType layerType;
        std::vector<std::vector<double>> weights;
        
        std::vector<double> outputs;
        std::vector<double> inputs;
        std::vector<double> nodeValues;

        std::vector<std::vector<double>> gradients;

        Adam optimizer;

        Layer(int n_in, int n_out, LayerType type, int idx) {
            n_inputs  = n_in;
            n_outputs = n_out;
            layerType = type;
            layerIdx = idx;
            outputs.resize(n_outputs);
            inputs.resize(n_inputs);

            nodeValues.resize(n_outputs);

            optimizer = Adam(n_inputs, n_outputs, 0.9, 0.999, 1e-8, 0, layerIdx);

            std::random_device rd;
            std::default_random_engine generator(rd());
            std::normal_distribution<double> distribution(0, sqrt(2/(double)n_inputs));
            weights.resize(n_inputs);
            gradients.resize(n_inputs);
            for(int i = 0; i < n_inputs; i++) {
                weights[i].resize(n_outputs);
                gradients[i].resize(n_outputs);
                for(int j = 0; j < n_outputs; j++) {
                    //Initialize random weights
                    weights[i][j] = distribution(generator);
                }
            }
        }

        void resetGradients() {
            for(int i = 0; i < n_inputs; i++) {
                std::fill(gradients[i].begin(), gradients[i].end(), 0);
            }
        }


        std::vector<double> forward(std::vector<double>& in) {
            
            inputs = in;

            std::fill(outputs.begin(), outputs.end(), 0);

            for(int i = 0; i < inputs.size(); i++) {
                for(int j = 0; j < outputs.size(); j++) {
                    outputs[j] += inputs[i] * weights[i][j];   
                }
            }

            if(layerType == HIDDEN) {
                for(double& n : outputs)  {
                    if (n < 0) n = 0; // relu
                }
            }
            else if(layerType == OUTPUT) {
                //nothing
            }
            return outputs;
        }

        std::vector<double>& outputLayerNodeValues(double lossDerivative, Action action) {
            std::fill(nodeValues.begin(), nodeValues.end(), 0);
            nodeValues[action] = lossDerivative;
            return nodeValues;
        }

        std::vector<double>& hiddenLayerNodeValues(Layer& oldLayer, std::vector<double>& oldNodeValues) {

            std::vector<double> activationDerivatives = outputs;
            for(double& x : activationDerivatives) {
                if (x > 0) x = 1; // relu derivative
                else x = 0;
            }

            for(int i = 0; i < oldLayer.n_inputs; i++) {
                double nodeValue = 0;
                for(int j = 0; j < oldLayer.n_outputs; j++) {
                    nodeValue += oldLayer.weights[i][j] * oldNodeValues[j];
                }

                nodeValue *= activationDerivatives[i];
                nodeValues[i] = nodeValue;
            }

            return nodeValues;
        }


        void updateGradients() {
            for(int i = 0; i < n_inputs; i++) {
                for(int j = 0; j < n_outputs; j++) {
                    gradients[i][j] += inputs[i] * nodeValues[j];
                }
            }
        }


        void applyGradients(double learnRate) {
      
            optimizer.optimize(gradients);

            for(int i = 0; i < n_inputs; i++) {
                for(int j = 0; j < n_outputs; j++) {
                    weights[i][j] -= gradients[i][j] * learnRate;
                }
            }
        }

        void save(int idx, std::string path) {
            std::ofstream outFile(path + "/layer" + std::to_string(idx) + ".txt");

            if (!outFile) {
                // If file opening fails, output an error message
                std::cerr << "Error: Could not open file" << path << "/layer" << idx << ".txt for writing." << std::endl;
                return;
            }

            for(int i = 0; i < n_inputs; i++) {
                for(int j = 0; j < n_outputs; j++) {
                    if (outFile.is_open()) {
                        outFile << weights[i][j] << " ";
                    }
                }
                outFile << std::endl;
            }
            std::cout << "Layer saved successfully\n";
            outFile.close();
        }


        void load(int idx, std::string path) {
            std::ifstream inFile(path + "/layer" + std::to_string(idx) + ".txt");
            if (!inFile) {
                std::cerr << "Error: Could not open file" << path << "/layer" << idx << ".txt for reading. initializing random layer..." << std::endl;
                return;
            }

            std::string line;
            int i = 0; // Row index for weights

            // Read the file line by line
            while (std::getline(inFile, line) && i < n_inputs) {
                std::istringstream iss(line);
                for (int j = 0; j < n_outputs; ++j) {
                    if (!(iss >> weights[i][j])) {
                        std::cerr << "Error: Failed to read weight at position [" << i << "][" << j << "]" << std::endl;
                        return;
                    }
                }
                ++i;
            }

            inFile.close();
            std::cout << "Successfully loaded layer " << idx << " from file." << std::endl;
        }

};

#endif