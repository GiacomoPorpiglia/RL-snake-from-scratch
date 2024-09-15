#include "snake.h"
#include <cstdlib>
#include <iostream>
#include <random>

Snake& initSnake(Snake &s) {


    s.len = 1;

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister RNG
    std::uniform_int_distribution<> dis(0, GRID_SIZE - 1);

    s.xpos = dis(gen);
    s.ypos = dis(gen);

    s.stepsAlive = 0;


    for(int i = 0; i < GRID_SIZE*GRID_SIZE; i++) {
        s.tailX[i] = NONE;
        s.tailY[i] = NONE;
    }
    s.prevTailEndX = NONE;
    s.prevTailEndY = NONE;
    s.dir = WEST;

    s.appleX = dis(gen);
    s.appleY = dis(gen);
    bool valid = false;
    while (!valid)
    {
        valid = true;
        for (int i = 0; i < s.len - 1; i++) {
            if (s.appleX == s.tailX[i] && s.appleY == s.tailY[i]) {
                valid = false;
                break;
            }
        }
        if (s.appleX == s.xpos && s.appleY == s.ypos)
            valid = false;
        if(s.appleX >= GRID_SIZE || s.appleY >= GRID_SIZE)
            valid = false;

        if(!valid) {
            s.appleX = dis(gen);
            s.appleY = dis(gen);
        }
    }

    return s;
}

void move(Snake& s) {
    //update tail

    //if there is a tail, update the start of the tail to the prev head pos
    if (s.len >= 2) {

        s.prevTailEndX = s.tailX[s.len - 2];
        s.prevTailEndY = s.tailY[s.len - 2];

    } 
    //if there isn't a tail, we don't have a prev tail end
    else {
        s.prevTailEndX = NONE;
        s.prevTailEndY = NONE;
    }

    for (int i = s.len - 2; i > 0; i--) {
        s.tailX[i] = s.tailX[i - 1];
        s.tailY[i] = s.tailY[i - 1];
    }

    s.tailX[0] = s.xpos;
    s.tailY[0] = s.ypos;

    if (s.dir == NORTH) {
        s.xpos = s.xpos;
        s.ypos = s.ypos - 1;
    } 
    else if (s.dir == SOUTH) {
        s.xpos = s.xpos;
        s.ypos = s.ypos + 1;
    }
    else if (s.dir == WEST) {
        s.xpos = s.xpos - 1;
        s.ypos = s.ypos;
    }
    else if (s.dir == EAST)
    {
        s.xpos = s.xpos + 1;
        s.ypos = s.ypos;
    }
}


void eat(Snake& s) {
    
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister RNG
    std::uniform_int_distribution<> dis(0, GRID_SIZE - 1);

    // if there is no tail
    if (s.len==1) {
        if(s.dir == WEST) {
            s.tailX[0] = s.xpos + 1;
            s.tailY[0] = s.ypos;
        } 
        else if(s.dir == EAST) {
            s.tailX[0] = s.xpos - 1;
            s.tailY[0] = s.ypos;
        }
        else if (s.dir == NORTH) {
            s.tailX[0] = s.xpos;
            s.tailY[0] = s.ypos + 1;
        } 
        else if (s.dir == SOUTH) {
            s.tailX[0] = s.xpos;
            s.tailY[0] = s.ypos - 1;
        }
    }
    else {
        s.tailX[s.len-1] = s.prevTailEndX;
        s.tailY[s.len-1] = s.prevTailEndY;
    }

    s.len++;


    s.appleX = dis(gen);
    s.appleY = dis(gen);
    bool valid = false;
    while(!valid) {
        valid = true;
        for(int i = 0; i < s.len-1; i++) {
            if(s.appleX == s.tailX[i] && s.appleY == s.tailY[i]) {
                valid = false;
                break;
            }
        }
        if(s.appleX == s.xpos && s.appleY == s.ypos) 
            valid = false;

        if(!valid) {
            s.appleX = dis(gen);
            s.appleY = dis(gen);
        }

    }   
}

int getDistFromFood(Snake &s) {
    return abs(s.xpos-s.appleX) + abs(s.ypos-s.appleY);
}
