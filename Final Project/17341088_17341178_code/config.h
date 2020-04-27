#ifndef CONFIG_H
#define CONFIG_H
#include <vector>
using namespace std;


struct weightstruct {
    vector<double> subweight;
    int fitness;
};

struct Node {
    int mode;
    char board[8][8];
    double score;
    pair<int, int> action;
    vector<Node> children;
    Node(char raw_board[8][8], int cur_mode) {
        mode = cur_mode;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                board[i][j] = raw_board[i][j];
    }
};

#endif // CONFIG_H
