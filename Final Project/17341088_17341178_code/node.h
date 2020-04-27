#ifndef NODE_H
#define NODE_H
#include <vector>
using namespace std;

struct Node {
    int mode;
    int board[8][8];
    double score;
    pair<int, int> action;
    vector<Node> children;
    Node(int raw_board[8][8], int cur_mode) {
        mode = cur_mode;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                board[i][j] = raw_board[i][j];
    }
};

#endif // NODE_H
