#ifndef SNAKE_H
#define SNAKE_H
#include "types.h"

typedef struct {
    int xpos;
    int ypos;
    int tailX[GRID_SIZE * GRID_SIZE - 1];
    int tailY[GRID_SIZE * GRID_SIZE - 1];
    int dir;

    int prevTailEndX;
    int prevTailEndY;
    int len;

    int appleX;
    int appleY;

    int stepsAlive;

    int batches;

} Snake;

void move(Snake &s);
void eat(Snake& s);
int getDistFromFood(Snake& s);

Snake& initSnake(Snake& s);

#endif