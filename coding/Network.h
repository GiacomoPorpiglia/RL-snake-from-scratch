#ifndef NETWORK_H
#define NETWORK_H
#include <vector>
#include "Layer.h"
#include "Adam.h"
#include "types.h"
#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <typeinfo>

class Network {

    public:
        std::vector<Layer> layers;
        double learnRate;
        double epsilon;
        std::string path;

        Network(std::vector<int> layerSizes, double eps, double lr, std::string p) {

            learnRate = lr;
            epsilon = eps;
            path = p;

            for(int i = 1; i < layerSizes.size(); i++) {
                LayerType type = HIDDEN;
                if(i == layerSizes.size()-1) type = OUTPUT;
                layers.push_back(Layer(layerSizes[i-1], layerSizes[i], type, i-1));
            }

            load();

        }

        std::vector<double> forward(std::vector<double> input) {
            for(int i = 0; i < layers.size(); i++) {
                input = layers[i].forward(input);
            }
            return input; //it's called input but it's actually the final output
        }

        void learn(std::vector<std::tuple<ReplayRecord, double>> batch) {

            for(auto& layer : layers)
                layer.resetGradients();

            std::ofstream outFile("output.txt", std::ios::app);

            double totalLoss = 0;

            for(auto btc : batch) {
                auto b      = std::get<0>(btc);
                auto reward = std::get<1>(btc);

                auto input    = std::get<0>(b).state;
                Action action = std::get<1>(b);

                auto networkOutput = forward(input);

                double lossDerivative = networkOutput[action]-reward;

                totalLoss += lossDerivative * lossDerivative / 2;

                auto nodeValues = layers[layers.size()-1].outputLayerNodeValues(lossDerivative, action);
                layers[layers.size()-1].updateGradients();

                for (int layerIdx = layers.size()-2; layerIdx >= 0; layerIdx--) {
                    nodeValues = layers[layerIdx].hiddenLayerNodeValues(layers[layerIdx+1], nodeValues);
                    
                    layers[layerIdx].updateGradients();
                }
            }

            for(auto& layer : layers) {
                layer.applyGradients(learnRate);
            }
            if (outFile.is_open()) {
                outFile << totalLoss / batch.size() << std::endl;
            }

            std::cout << "\nAvg loss: " << totalLoss / batch.size() << "\n";
        }



        void save() {
            int i = 0;
            for(auto& layer : layers) {
                layer.save(i, path);
                if(typeid(layer.optimizer) == typeid(Adam)) {
                    layer.optimizer.save(path);
                }
                i++;
            }
        }


        void load() {
            int i = 0;
            for(auto& layer : layers) {
                layer.load(i, path);
                if(typeid(layer.optimizer) == typeid(Adam)) {
                    layer.optimizer.load(path);
                }
                i++;
            }
        }
};

#endif