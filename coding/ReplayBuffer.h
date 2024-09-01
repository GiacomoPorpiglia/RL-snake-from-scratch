#ifndef REPLAY_H
#define REPLAY_H
#include <vector>
#include "types.h"
#include <tuple>
#include <random>
#include <algorithm>

bool compareByFirst(const ReplayRecord &a, const ReplayRecord &b) {
    return std::get<2>(a) < std::get<2>(b);
}

class ReplayBuffer {

    public:
        std::vector<ReplayRecord> buffer;
        std::vector<double> rewards;
        double gamma;
        double reward;
        int maxSize;
        int batchSize;

        ReplayBuffer(int bs) {
            buffer.clear();
            gamma = 0.95;
            maxSize = 2000;
            batchSize = bs;
        }

        void empty() {
            buffer.clear();
        }

        void addRecord(State curr, Action a, double r, State next, bool final, Network& network) {
            buffer.push_back(std::make_tuple(curr, a, r, next, final));

            if(final) {
                //if final, there is no cumulative reward, so the q-value is just the static reward
                rewards.push_back(r);
            } else {
                //if not final, the q-value = is static_reward + gamma * best_next_state_q-value
                auto nextStateOutputs = network.forward(next.state);
                rewards.push_back(r + gamma * *std::max_element(nextStateOutputs.begin(), nextStateOutputs.end()));
            }


            //acts like a queue with maxSize (FIFO)
            if(buffer.size() > maxSize) {
                buffer.erase(buffer.begin());
                rewards.erase(rewards.begin());
            }
        }

        std::vector<std::tuple<ReplayRecord, double>> sampleBatch() {

            if(batchSize > buffer.size()) {
                return {};
            }

            std::random_device rd;
            std::mt19937 generator(rd());
            
            std::vector<std::tuple<ReplayRecord, double>> batch;
 
            while(batch.size() < batchSize) {

                std::uniform_int_distribution<> dist(0, buffer.size() - 1);
                int index = dist(generator);
                batch.push_back(std::make_tuple(buffer[index], rewards[index]));
                buffer.erase(buffer.begin()+index);
                rewards.erase(rewards.begin()+index);
            }

            return batch;
        }
};

#endif