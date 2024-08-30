#ifndef TYPES_H
#define TYPES_H
#include <vector>

#define GRID_SIZE 12
#define WINDOW_SIZE 600
#define CELL_SIZE WINDOW_SIZE/GRID_SIZE

#define NORTH 0
#define WEST 1
#define SOUTH 2
#define EAST 3

#define NONE -1

#define MOVE_REWARD -0.1
#define EAT_REWARD 1
#define DIE_REWARD -10

#define ReplayRecord std::tuple<State, Action, double, State, bool>


struct State {
    std::vector<double> state;
};

typedef enum {
    SX,
    ON,
    DX
} Action;


typedef enum {
    HIDDEN,
    OUTPUT
} LayerType;

#endif