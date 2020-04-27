#ifndef BLACKWHITECHESS_H
#define BLACKWHITECHESS_H
#include "node.h"
#include <time.h>
#include <algorithm>
#include <iostream>

enum GameState
{
    PLAYING,
    BLACKWIN,
    WHITEWIN,
    DOGFALL,
    END
};

struct weightstruct {
    vector<double> subweight;
    int fitness;
};


class BlackWhiteChess
{
private:
    int player;
    int limit;
    bool personValidPos;
    bool botValidPos;
    vector<double> bestweight;
    GameState gs;
    bool dialogCheck(int chess, int posx, int posy);
    bool verticalCheck(int chess, int posx, int posy);
    bool horizontalCheck(int chess, int posx, int posy);
    bool setChess(int chess, int posx, int posy);
    double alphabetapruning(Node &root, int mode, int depth, double alpha, double beta);

    void init_board(int board[8][8]);
    bool is_on_board(int row, int col);
    vector<int> find_filp_dirs(int board[8][8], pair<int, int> pos, int color);
    vector<pair<int, int>> find_places(int board[8][8], int turn);
    void flip(int board[8][8], pair<int, int> action, int color);
    void move(int board[8][8], pair<int, int> pos, int color);
    double calPosEval(int board[8][8], int color);
    double calRateEval(int board[8][8], int color);
    double calCornerEval(int board[8][8], int color);
    double calNearCornerEval(int board[8][8], int color);
    double calMoveEval(int board[8][8], int color);
    bool is_stable(int board[8][8], int x, int y);
    double calStableEval(int board[8][8], int color);
    double calSideEval(int board[8][8], int color);
    double calEval(vector<double> weight, vector<double> subeval);
    double evaluate(int board[8][8], int color, vector<double> weight);
    void random_play(int board[8][8], int color);
    vector<double> get_average_all_weights(vector<weightstruct> weights);
    void setBestWeights();
    weightstruct get_weight_by_GA(int board[8][8], int color);
    weightstruct ga(vector<double> subeval);
    void copy(vector<weightstruct>& weightvec, vector<weightstruct>& new_weightvec);
    void cross(vector<weightstruct>& weightvec, vector<weightstruct>& new_weightvec);
    void mutate(vector<weightstruct>& weightvec, vector<weightstruct>& new_weightvec);

    int square_weights[8][8] = {
        {  1000, -200,  30,   20,   20,  30, -200,  1000},
        { -200, -250,  20,   0,   0,  20, -250, -200},
        {  30,  20,   30,   2,   2,   30,  20,  30},
        {   20,   0,   2,  -3,  -3,   2,   0,   20},
        {   20,   0,   2,  -3,  -3,   2,   0,   20},
        {  30,  20,   30,   2,   2,   30,  20,  30},
        { -200,-250,  20,   0,   0,  20, -250, -200},
        {  1000, -200,  30,   20,   20,  30, -200,  1000},
    };

    int direction[8][2] = {
        {0, 1}, {0, -1}, {1, 1}, {1, 0}, {1, -1}, {-1, 0}, {-1, 1}, {-1, -1}
    };



public:
    int person, bot;
    int chessBoard[8][8];
    BlackWhiteChess(int firstPlyer = 1, int depth = 6);
    ~BlackWhiteChess() {}
    void play(int posx, int posy);
    void checkGameState();
    GameState getGameState() const;
    int getPlayer() const;
    void setGameEnd();
    void botPlay();
};

#endif // BLACKWHITECHESS_H
