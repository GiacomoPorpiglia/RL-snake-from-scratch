#include <SFML/Graphics.hpp>
#include "types.h"
#include "snake.h"
#include <iostream>
#include <cstdlib>
#include "Network.h"
#include <algorithm>
#include "ReplayBuffer.h"
#include <random>
#include <fstream>

//0.055 epsilon
int counter = 0;

void drawFrame(sf::RenderWindow& window, Snake& s) {
    sf::RectangleShape head(sf::Vector2f(CELL_SIZE-1, CELL_SIZE-1)); // head
    head.setFillColor(sf::Color::Green);
    head.setPosition(s.xpos * CELL_SIZE, s.ypos * CELL_SIZE);
    window.draw(head);

    sf::RectangleShape apple(sf::Vector2f(CELL_SIZE-1, CELL_SIZE-1)); // apple
    apple.setFillColor(sf::Color::Red);
    apple.setPosition(s.appleX * CELL_SIZE, s.appleY * CELL_SIZE);
    window.draw(apple);

    for(int i = 0; i < s.len-1; i++) {
        sf::RectangleShape tail(sf::Vector2f(CELL_SIZE-1, CELL_SIZE-1)); // head
        sf::Color Violet(128, 0, 200, 255);
        tail.setFillColor(Violet);
        tail.setPosition(s.tailX[i] * CELL_SIZE, s.tailY[i] * CELL_SIZE);
        window.draw(tail);
    }
}




bool isDead(Snake& s) {
    if(s.xpos < 0 || s.xpos >= GRID_SIZE || s.ypos < 0 || s.ypos >= GRID_SIZE)
        return true;

    for(int i = 0; i < s.len-1; i++) {
        if(s.xpos == s.tailX[i] && s.ypos == s.tailY[i])
            return true;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
        return true;

    return false;
}



State getState(Snake& s) {
    //get distances from walls/tail in every direction
    int w, e, n, st;
    for(w = 1; s.xpos-w >= 0; w++) {
        bool found = false;
        for(int i = 0; i < s.len-1; i++) {
            if(s.tailX[i] == (s.xpos-w) && s.tailY[i] == s.ypos) {
                found = true;
                break;
            }
        }
        if(found) break;
    }

    for(e = 1; s.xpos+e < GRID_SIZE; e++) {
        bool found = false;
        for(int i = 0; i < s.len-1; i++) {
            if(s.tailX[i] == (s.xpos+e) && s.tailY[i] == s.ypos) {
                found = true;
                break;
            }
        }
        if(found) break;
    }

    for(n = 1; s.ypos-n >= 0; n++) {
        bool found = false;
        for(int i = 0; i < s.len-1; i++) {
            if(s.tailX[i] == s.xpos && s.tailY[i] == (s.ypos-n)) {
                found = true;
                break;
            }
        }
        if(found) break;
    }

    for(st = 1; s.ypos+st < GRID_SIZE; st++) {
        bool found = false;
        for(int i = 0; i < s.len-1; i++) {
            if(s.tailX[i] == s.xpos && s.tailY[i] == (s.ypos+st)) {
                found = true;
                break;
            }
        }
        if(found) break;
    }

    int leftDist, rightDist, forwardDist;

    if(s.dir == NORTH) {
        leftDist = w; rightDist = e; forwardDist = n;
    }
    else if(s.dir == SOUTH) {
        leftDist = e; rightDist = w; forwardDist = st;
    }
    else if(s.dir == WEST) {
        leftDist = st; rightDist = n; forwardDist = w;
    }
    else if(s.dir == EAST) {
        leftDist = n; rightDist = st; forwardDist = e;
    }
    
    //create input vector
    std::vector<double> inputs = {
        (double) s.xpos      / GRID_SIZE,
        (double) s.ypos      / GRID_SIZE,
        (double) s.dir       / 4,
        (double) leftDist    / GRID_SIZE,
        (double) rightDist   / GRID_SIZE,
        (double) forwardDist / GRID_SIZE,
        (double) (s.xpos-s.appleX) / GRID_SIZE,
        (double) (s.ypos-s.appleY) / GRID_SIZE,
        (double) (s.len) / (GRID_SIZE*GRID_SIZE)
    };

    State currentState;
    currentState.state = inputs;
    return currentState;
}



void updateSnake(Snake& s, Network& network, ReplayBuffer& buffer, bool train) {


    State currentState = getState(s);
    //compute network output

    std::vector<double> outputs = network.forward(currentState.state);


    //---------------------epsilon-greedy policy-----------------

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine seeded with rd()

    //random value between 0 and 1
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double random_value = dis(gen);

    int actionIndex;

    if(random_value < network.epsilon) {
        //play random action with probability epsilon
        std::uniform_int_distribution<> dis(0, 2);
        actionIndex = dis(gen);
    }
    else {
        //play best action
        auto maxElementIter = std::max_element(outputs.begin(), outputs.end());
        actionIndex = std::distance(outputs.begin(), maxElementIter);
    }


    //-----------------------------------------

    if(actionIndex == ON) {
    }
    else if (actionIndex == SX) {
        s.dir += 1;
        s.dir = s.dir % 4;
    }
    else if (actionIndex == DX) {
        s.dir += 3;
        s.dir = s.dir % 4;
    }
    move(s);
    s.stepsAlive++;

    State nextState = getState(s);
    double reward = 0;
    bool final = false;
    if (s.xpos == s.appleX && s.ypos == s.appleY) {
        eat(s);
        s.stepsAlive = 0;
        reward = EAT_REWARD;
    }
    else if (isDead(s)) {
        reward = DIE_REWARD;
        final = true;
    } else {
        reward = MOVE_REWARD;
    }
    if (s.stepsAlive > 80) {
        final = true;
    }

    buffer.addRecord(currentState, (Action) actionIndex, reward, nextState, final, network);

    if (final) {
        counter++;

        //print infos
        std::cout << "Length: " << s.len << "   Counter: " <<  counter << "\n";
        std::cout << "Epsilon: " << network.epsilon << "\n";
        std::cout << "Learn rate: " << network.learnRate << "\n";

        if(counter % 1000 == 0 && train) {
            network.save(); // save the network every 1000 games
        }

        // network.epsilon -= network.epsilon / 250000; //gradually decrease epsilon
        if(train)
            network.epsilon = std::max(network.epsilon, 0.0);

        if(train) {
            auto batch = buffer.sampleBatch();
            s.batches++;
            if(batch.size() != 0) {
                network.learn(batch);
            }

            //print on file the len of snake
            std::ofstream outFile("length.txt", std::ios::app);
            if (outFile.is_open()) {
                outFile << s.len << std::endl;
            }
        }

        s = initSnake(s);
    }

}

int main(int argc, char* argv[])
{

    std::string mode;
    bool train = false;

    double epsilon = 0.2;
    double learnRate = 0.001;

    int batchSize = 64;
    std::string path = "";

    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if(arg == "--mode" && i+1 < argc) {
            mode = argv[i+1];
            i++;
            if (mode == "train")
                train = true;
            else if (mode == "play")
                train = false;
            else {
                std::cerr << "Unknown value for flag --mode. Accepted values are 'train' and 'play'." << std::endl;
                return 1;
            }
        }
        else if(arg=="--epsilon" && i+1 < argc) {
            epsilon = std::stod(argv[i+1]);
            if(epsilon < 0 || epsilon > 1) {
                std::cerr << "Possible values for epsilon are between 0 and 1." << std::endl;
                return 1;
            }
            i++;
        }

        else if(arg == "--learnrate" && i+1 < argc) {
            learnRate = std::stod(argv[i+1]);
            if(learnRate <= 0) {
                std::cerr << "Possible values for the learnRate have to be strictly positive" << std::endl;
                return 1;
            }
            i++;
        }
        else if(arg == "--batchsize" && i+1 < argc) {
            batchSize = std::stoi(argv[i+1]);
            if(batchSize <= 0) {
                std::cerr << "Possible values for the batch size have to be strictly positive integers." << std::endl;
                return 1;
            }
            i++;
        }
        else if(arg == "--path" && i+1 < argc) {
            path = argv[i+1];
            i++;
        }
    }

    if(path == "") {
        std::cerr << "You have to specify a path to load an existing network, or to save a new one" << std::endl;
        return 1;
    }
    

    std::srand(static_cast<unsigned int>(std::time(nullptr))); // init random seed

    Snake snake;
    snake = initSnake(snake);
    int frameRate = 10;



    std::vector<int> networkSizes = {9, 100, 50, 3}; //net size
    Network network = Network(networkSizes, epsilon, learnRate, path);      // init network
    ReplayBuffer buffer = ReplayBuffer(batchSize);         // init buffer

    sf::RenderWindow window;
    if(!train) {
        window.create(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "snake");
        window.setFramerateLimit(frameRate);
    }
    

    //print infos
    std::cout << "\n-------------------------------------\n";
    std::cout << "Network has the following parameters: \n";
    std::cout << "Epsilon: -----------------  " << network.epsilon << "\n";
    std::cout << "Learn Rate: --------------  " << network.learnRate << "\n";
    std::cout << "Batch size: --------------  " << batchSize << "\n";
    std::cout << "Path: --------------------  " << path << "\n";
    std::cout << "-------------------------------------\n";

    if(!train) {
        std::cout << "The mode is 'play', so the network won't change.\n";
        std::cout << "If you wish to train your network, set --mode to train.\n";
        network.epsilon = 0;
        std::cout << "The epsilon is automatically set to 0 (since it's mode play)\n";
    }

    //game loop
    while (true) {

        if(!train) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed)
                    window.close();

                if (event.type == sf::Event::KeyPressed) {
                    if (event.key.code == sf::Keyboard::F) {
                        // Toggle the frame rate
                        frameRate = (frameRate == 60) ? 10 : 60;
                        window.setFramerateLimit(frameRate);
                    }
                    //if S is pressed, save the network
                    if (event.key.code == sf::Keyboard::S) {
                        // network.save();
                    }
                }
            }
        }

        
        
        if(!train)
            window.clear();

        updateSnake(snake, network, buffer, train);

        if(!train) {
            drawFrame(window, snake);
            window.display();
        }

    }

    return 0;
}