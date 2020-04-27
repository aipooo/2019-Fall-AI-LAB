#ifndef GA_H
#define GA_H
#include "config.h"
#include <iostream>
#include <time.h>
#include <vector>
#include <cstring>
#include <algorithm>
using namespace std;

vector <double> bestweight;//最佳权重
char board_copy[8][8];

class Reversi {
    private:
        int square_weights[8][8] = {
            {400, -30,  11,   8,   8,  11, -30, 400},
            {-30, -70,  -4,   1,   1,  -4, -70, -30},
            { 11,  -4,   2,   2,   2,   2,  -4,  11},
            {  8,   1,   2,  -3,  -3,   2,   1,   8},
            {  8,   1,   2,  -3,  -3,   2,   1,   8},
            { 11,  -4,   2,   2,   2,   2,  -4,  11},
            {-30, -70,  -4,   1,   1,  -4, -70, -30},
            {400, -30,  11,   8,   8,  11, -30, 400},
        };
        int direction[8][2] = {
            {0, 1}, {0, -1}, {1, 1}, {1, 0}, {1, -1}, {-1, 0}, {-1, 1}, {-1, -1}
        };
        int limit;

    public:
        Reversi(int depth) {
            limit = depth;
        }

        void init_board(char board[8][8]) {
            // 对棋盘初始化
            for (int i = 0; i < 8; ++i)
                memset(board[i], ' ', 8);
            board[3][4] = board[4][3] = 'X';
            board[3][3] = board[4][4] = 'O';
        }

        void print_board(char board[8][8], vector<pair<int, int>> valid_pos) {
            // 打印棋盘，valid_pos表示可落子位置
            char col = 'A';
            cout << endl << ' ';
            for (char i = 0; i < 8; i++)
                cout << ' ' << char(col + i);
            cout << endl;
            for (int i = 0; i < 8; i++) {
                cout << i + 1 << ' ';
                for (int j = 0; j < 8; ++j)
                    if (find(valid_pos.begin(), valid_pos.end(), make_pair(i, j)) == valid_pos.end())
                        cout << board[i][j] << ' ';
                    else
                        cout << "* ";
                cout << endl;
            }
            pair<int, int> nums = count_number(board);
            cout << "黑棋 : 白棋 = " << nums.first << " : " << nums.second << endl;
        }

        bool is_on_board(int row, int col) {
            // 判断某个位置是否在棋盘上
            if (row < 8 && row >= 0 && col < 8 && col >= 0)
                return true;
            return false;
        }

        vector<pair<int, int>> find_places(char board[8][8], char turn) {
            // 寻找可下子位置，turn标记是黑棋落子还是白棋落子
            vector<pair<int, int>> valid_pos;
            char item = turn == 'X' ? 'X' : 'O';
            char opp = turn == 'X' ? 'O' : 'X';
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    if (board[i][j] == ' ' && !find_filp_dirs(board, make_pair(i, j), item).empty())
                        valid_pos.push_back(make_pair(i, j));
            return valid_pos;
        }

        vector<int> find_filp_dirs(char board[8][8], pair<int, int> pos, char color) {
            // 获取某个位置能够翻转对方棋子的方向
            int x = pos.first, y = pos.second;
            vector<int> dirs;
            char opp = color == 'X' ? 'O' : 'X';
            for (int i = 0; i < 8; ++i) {
                x += direction[i][0];
                y += direction[i][1];
                if (is_on_board(x, y) && board[x][y] == opp) {
                    while (is_on_board(x, y) && board[x][y] == opp) {
                        x += direction[i][0];
                        y += direction[i][1];
                    }
                    if (is_on_board(x, y) && board[x][y] == color)
                        dirs.push_back(i);
                }
                x = pos.first;
                y = pos.second;
            }
            return dirs;
        }

        void flip(char board[8][8], pair<int, int> action, char color) {
            // 根据落子位置，执行翻转操作
            vector<int> dirs = find_filp_dirs(board, action, color);
            char opp = color == 'X' ? 'O' : 'X';
            int x = action.first, y = action.second;
            for (int i = 0; i < dirs.size(); ++i) {
                x += direction[dirs[i]][0];
                y += direction[dirs[i]][1];
                while (is_on_board(x, y) && board[x][y] == opp) {
                    board[x][y] = color;
                    x += direction[dirs[i]][0];
                    y += direction[dirs[i]][1];
                }
                x = action.first;
                y = action.second;
            }
        }

        bool is_over(char board[8][8]) {
            // 判断比赛是否结束
            vector<pair<int, int>> v1 = find_places(board, 'X');
            vector<pair<int, int>> v2 = find_places(board, 'O');
            if (v1.empty() && v2.empty())
                return true;
            return false;
        }

        void player_play(char board[8][8], vector<pair<int, int>> valid_pos, char turn, int row, int col) {
            // 玩家下棋
            cout << "请输入落子位置（例如：A1）：";
            while (true) {
                if (col >= 'A' && col <= 'H' && row >= '1' && row <= '8') {
                    pair<int, int> action = make_pair(row - '1', col - 'A');
                    if (find(valid_pos.begin(), valid_pos.end(), action) != valid_pos.end()) {
                        move(board, action, turn);
                        break;
                    }
                }
                cout << "该位置不允许落子,请重新输入落子位置（例如：A1）：";
            }
            cout << endl;
        }

        vector<weightstruct> init_weightvec(int num){
            // 随机生成num个权重向量
            vector<weightstruct> weight_vec;
            for (int i = 0; i < num; i++) {
                weightstruct weightstructi;
                vector<double> subweighti;
                double sum = 0;
                for (int k = 0; k < 7; k++) {
                    double tmp = rand() / double(RAND_MAX);
                    subweighti.push_back(tmp);
                    sum += tmp;
                }
                for (int k = 0; k < 7; k++) {
                    subweighti[k] /= sum;
                    weightstructi.subweight.push_back(subweighti[k]);
                }
                weightstructi.fitness = 0;
                weight_vec.push_back(weightstructi);
            }
            return weight_vec;
        }

        double calEval(weightstruct weight, vector<double> subeval){
            // 计算总的评估值
            double eval = 0;
            for (int i = 0; i < 7; i++)
                eval += weight.subweight[i] * subeval[i];
            return eval;
        }

        double calEval(vector<double> weight, vector<double> subeval){
            // 计算总的评估值
            double eval = 0;
            for (int i = 0; i < 7; i++)
                eval += weight[i] * subeval[i];
            return eval;
        }

        void copy(vector<weightstruct>& weightvec, vector<weightstruct>& new_weightvec){
            // 复制部分，有90%的概率选取适应度最高的3个进行复制
            if (rand() % 10) {
                new_weightvec.push_back(weightvec[0]);
                new_weightvec.back().fitness = 0;
                new_weightvec.push_back(weightvec[1]);
                new_weightvec.back().fitness = 0;
                new_weightvec.push_back(weightvec[2]);
                new_weightvec.back().fitness = 0;
            }
            //有10%的概率生成随机向量
            else {
                vector<weightstruct> rand_weight = init_weightvec(3);
                for (int i = 0; i < 3; i++)
                    new_weightvec.push_back(rand_weight[i]);
            }
        }

        void cross(vector<weightstruct>& weightvec, vector<weightstruct>& new_weightvec){
            // 交叉，选取适应度最高的4个两两进行单点交叉
            for (int i = 0; i < 4; i++) {
                for (int j = i + 1; j < 4; j++) {
                    weightstruct new_weight1, new_weight2;
                    new_weight1.fitness = 0;
                    new_weight2.fitness = 0;
                    double sum1 = 0, sum2 = 0;
                    int cut = rand() % 7;	//cut为随机选择的单点交叉点
                    for (int k = 0; k < 7; k++) {
                        //如果位置<=cut，保持不变，直接从父母复制
                        if (k <= cut) {
                            new_weight1.subweight.push_back(weightvec[i].subweight[k]);
                            sum1 += weightvec[i].subweight[k];
                            new_weight2.subweight.push_back(weightvec[j].subweight[k]);
                            sum2 += weightvec[j].subweight[k];
                        }
                        //如果位置>cut，父母向量对应位置进行交换
                        else {
                            new_weight1.subweight.push_back(weightvec[j].subweight[k]);
                            sum1 += weightvec[j].subweight[k];
                            new_weight2.subweight.push_back(weightvec[i].subweight[k]);
                            sum2 += weightvec[i].subweight[k];
                        }
                    }
                    //归一化
                    for (int k = 0; k < 7; k++) {
                        new_weight1.subweight[k] = new_weight1.subweight[k] / sum1;
                        new_weight2.subweight[k] = new_weight2.subweight[k] / sum1;
                    }
                    new_weightvec.push_back(new_weight1);
                    new_weightvec.push_back(new_weight2);
                }
            }
        }

        void mutate(vector<weightstruct>& weightvec, vector<weightstruct>& new_weightvec){
            // 变异部分，从适应度最高的3个向量中随机选择一个向量，随机选择一个向量元素赋予新的随机值
            int index = rand() % 3;
            weightstruct new_weight = weightvec[index];
            new_weight.fitness = 0;
            double sum = 0;
            // 随机选择一个位置赋予新的随机值
            int replace_pos = rand() % 7;
            for (int k = 0; k < 7; k++) {
                if (k == replace_pos){
                    double replace_ele = (rand() / double(RAND_MAX));
                    new_weight.subweight.push_back(replace_ele);
                    sum += replace_ele;
                }
                else
                    sum += new_weight.subweight[k];
            }
            // 归一化
            for (int k = 0; k < 7; k++)
                new_weight.subweight[k] = new_weight.subweight[k] / sum;
            new_weightvec.push_back(new_weight);
        }

        weightstruct ga(vector<double> subeval) {
            // 遗传算法计算子评估值的权重向量
            srand(time(0));
            vector<weightstruct> weightvec = init_weightvec(10);
            int iter = 20;	// 进行20次迭代
            while (iter--) {
                // 对当前10个向量两两进行比较
                for (int i = 0; i < 10; i++) {
                    for (int j = i + 1; j < 10; j++) {
                        if (calEval(weightvec[i], subeval) > calEval(weightvec[j], subeval))
                            weightvec[i].fitness++;
                        else
                            weightvec[j].fitness++;
                    }
                }
                //排序
                for (int i = 0; i < 10; i++)
                    for (int j = i+1; j < 10; j++)
                        if (weightvec[i].fitness < weightvec[j].fitness)
                            swap(weightvec[i], weightvec[j]);

                vector<weightstruct> new_weightvec;
                //复制
                copy(weightvec, new_weightvec);
                //交叉
                cross(weightvec, new_weightvec);
                //变异
                mutate(weightvec, new_weightvec);
                weightvec = new_weightvec;
            }
            return weightvec[0];
        }

        double calPosEval(char color){
            // 计算位置权重评估值
            char opp = color == 'X' ? 'O' : 'X';
            double poseval = 0;
            for (int i = 0; i < 8; ++i){
                for (int j = 0; j < 8; ++j){
                    if (board_copy[i][j] == color)
                        poseval += square_weights[i][j];
                    else if (board_copy[i][j] == opp)
                        poseval -= square_weights[i][j];
                }
            }
            return poseval;
        }

        double calRateEval(char color){
            // 计算黑白子比例评估值
            char opp = color == 'X' ? 'O' : 'X';
            int mycount = 0, opcount = 0;
            for (int i = 0; i < 8; ++i){
                for (int j = 0; j < 8; ++j){
                    if (board_copy[i][j] == color)
                        mycount++;
                    else if (board_copy[i][j] == opp)
                        opcount++;
                }
            }
            if (mycount > opcount)
                return 100.0 * mycount / (mycount + opcount);
            else if (mycount < opcount)
                return -100.0 * opcount / (mycount + opcount);
            else
                return 0;
        }

        double calCornerEval(char color){
            // 计算占角评估值
            char opp = color == 'X' ? 'O' : 'X';
            int corner_pos[4][2] = { { 0, 0 },{ 0, 7 },{ 7, 0 },{ 7, 7 } };
            int mycorner = 0, opcorner = 0;
            for (int i = 0; i < 4; ++i){
                if (board_copy[corner_pos[i][0]][corner_pos[i][1]] == color)
                    mycorner++;
                else if (board_copy[corner_pos[i][0]][corner_pos[i][1]] == opp)
                    opcorner++;
            }
            return 25 * (mycorner - opcorner);
        }

        double calNearCornerEval(char color){
            // 计算近角评估值
            char opp = color == 'X' ? 'O' : 'X';
            int corner_pos[4][2] = { { 0, 0 },{ 0, 7 },{ 7, 0 },{ 7, 7 } };
            int mynear = 0, oppnear = 0;
            for (int i = 0; i < 4; i++) {
                int x = corner_pos[i][0], y = corner_pos[i][1];
                    if (board_copy[x][y] == ' ')
                        for (int j = 0; j < 8; j++) {
                            int nx = x + direction[j][0], ny = y + direction[j][1];
                            if (is_on_board(nx, ny))
                                if (board_copy[nx][ny] == color)
                                    mynear++;
                                else if (board_copy[nx][ny] == opp)
                                    oppnear++;
                        }
            }
            return -25 * (mynear - oppnear);
        }

        double calMoveEval(char color){
            // 计算行动力评估值
            char opp = color == 'X' ? 'O' : 'X';
            int mymove = find_places(board_copy, color).size();
            int opmove = find_places(board_copy, opp).size();
            //如果本方没有地方落子，则设定很低的行动力评估值
            if (mymove == 0)
                return -450;
            //如果对方没有地方落子，则设定很高的行动力评估值
            else if (opmove == 0)
                return 150;
            else if (mymove > opmove)
                return (100 * mymove) / (mymove + opmove);
            else if (mymove < opmove)
                return -(100 * opmove) / (mymove + opmove);
            else
                return 0;
        }

        bool is_stable(char board[8][8], int x, int y) {
            // 判断某个位置是否为稳定点
            for (int i = 0; i < 8; ++i)
                for (int nx = x + direction[i][0], ny = y + direction[i][1]; is_on_board(nx, ny); nx += direction[i][0], ny += direction[i][1])
                    if (board[nx][ny] == ' ')
                        return false;
            return true;
        }

        double calStableEval(char color){
            // 计算稳定子评估值
            char opp = color == 'X' ? 'O' : 'X';
            int mystable = 0, oppstable = 0;
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    if (board_copy[i][j] != ' ' && is_stable(board_copy, i, j))
                        if (board_copy[i][j] == color)
                            mystable++;
                        else
                            oppstable++;
            return 12.5 * (mystable - oppstable);
        }

        double calSideEval(char color){
            //计算边界棋子评估值
            char opp = color == 'X' ? 'O' : 'X';
            int index[2] = {0, 7};
            int myside = 0, oppside = 0;
            for (int i = 0; i < 2; i++){
                for (int j = 0; j < 8; j++){
                    if (board_copy[index[i]][j] == color)
                        myside++;
                    else if (board_copy[index[i]][j] == opp)
                        oppside++;
                }
            }
            for (int i = 0; i < 2; i++){
                for (int j = 1; j < 7; j++){
                    if (board_copy[j][index[i]] == color)
                        myside++;
                    else if (board_copy[index[i]][j] == opp)
                        oppside++;
                }
            }
            return 2.5 * (myside - oppside);
        }

        weightstruct get_weight_by_GA(char color) {
            // 利用遗传算法得到权重向量
            vector<double> eval;
            eval.push_back(calPosEval(color));
            eval.push_back(calMoveEval(color));
            eval.push_back(calSideEval(color));
            eval.push_back(calCornerEval(color));
            eval.push_back(calRateEval(color));
            eval.push_back(calStableEval(color));
            eval.push_back(calNearCornerEval(color));
            return ga(eval);
        }



        double evaluate(char board[8][8], char color) {
            //计算整个棋局评估值
            vector<double> eval;
            eval.push_back(calPosEval(color));
            eval.push_back(calMoveEval(color));
            eval.push_back(calSideEval(color));
            eval.push_back(calCornerEval(color));
            eval.push_back(calRateEval(color));
            eval.push_back(calStableEval(color));
            eval.push_back(calNearCornerEval(color));
            return calEval(bestweight, eval);
        }

        void move(char board[8][8], pair<int, int> pos, char color) {
            // 在pos处落子
            board[pos.first][pos.second] = color;
            flip(board, pos, color);
        }

        pair<int, int> count_number(char board[8][8]) {
            // 计算棋面双方棋子的数量
            int b_num = 0, w_num = 0;
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    if (board[i][j] == 'X')
                        b_num++;
                    else if (board[i][j] == 'O')
                        w_num++;
            return make_pair(b_num, w_num);
        }

        void random_play(char board[8][8], char color) {
            vector<pair<int, int>> valid_places = find_places(board, color);
            int index = rand() % valid_places.size();
            move(board, valid_places[index], color);
        }

        void ai_play(char board[8][8], char color, int depth) {
            char opp = color == 'X' ? 'O' : 'X';
            Node root = Node(board, 2);
            cout << "\nIDIOT正在思考...\n";
            double starttime = clock();
            int index = alphabetapruning(root, color, opp, 1, depth, -100000.0, 100000.0);
            cout << "IDIOT的下子位置为：" << char(root.children[index].action.second + 'A') << root.children[index].action.first + 1 << endl;
            double endtime = clock();
            cout << "IDIOT思考用时：" << (endtime - starttime) / 1000 << 's' << endl;
            move(board, root.children[index].action, color);
           // botplay(root.children[index].action.first, root.children[index].action.second);
            root.children.clear();
            cout << endl;
        }

        double alphabetapruning(Node &root, char ai, char player, int mode, int depth, double alpha, double beta) {
            //mode=0时：MAX层节点，mode=1：MIN层节点
            char color = mode == 1 ? ai : player;
            char opp = color == 'X' ? 'O' : 'X';
            auto avaiplaces = find_places(root.board, color);	//得到可下子的位置
            double v;
            if (depth == limit) {
                for (int i = 0; i < avaiplaces.size(); i++) {
                    Node newnode = Node(root.board, mode);		//新建子节点
                    newnode.action = avaiplaces[i];				//记录该节点下子的位置
                    move(newnode.board, avaiplaces[i], color);	//下子后棋盘发生变化
                    int oppmode = mode == 1 ? 0 : 1;			//进入下一种mode
                    auto places = find_places(newnode.board, opp);	//得到对方可下子的位置
                    if (places.size() != 0)
                        //如果对方还有可下子的位置，则递归搜索
                        newnode.score = alphabetapruning(newnode, ai, player, oppmode, depth - 1, alpha, beta);
                    else
                        //如果对方没有地方下子，则评估当前棋局
                        newnode.score = evaluate(newnode.board, color);
                    root.children.push_back(newnode);			//加入新的子节点
                }
                int index;
                double max = -100000.0;
                //得到估价值最高的走法
                for (int i = 0; i < root.children.size(); i++)
                    if (root.children[i].score > max) {
                        index = i;
                        max = root.children[i].score;
                    }
                cout << "IDIOT的走法及对应的估计值：";
                for (int i = 0; i < root.children.size(); i++) {
                    cout << char(root.children[i].action.second + 'A') << root.children[i].action.first + 1 << ':' << root.children[i].score << "  ";
                }
                cout << endl;
                //返回估价值最高的走法
                return index;
            }
            if (mode == 0) {
                //MAX层节点
                v = -100000.0;
                for (int i = 0; i < avaiplaces.size(); ++i) {
                    Node newnode = Node(root.board, mode);		//新建子节点
                    newnode.action = avaiplaces[i];				//记录该节点下子的位置
                    move(newnode.board, avaiplaces[i], color);	//下子后棋盘发生变化
                    int oppmode = mode == 1 ? 0 : 1;			//进入下一种mode
                    auto places = find_places(newnode.board, opp);	//得到对方可下子的位置
                    if (depth != 1 && places.size() != 0) {
                        //未到达深度限制且对方有地方下子，则递归搜索并更新v和alpha值
                        v = max(v, alphabetapruning(newnode, ai, player, oppmode, depth - 1, alpha, beta));
                        alpha = max(alpha, v);
                        if (beta <= alpha)	//alpha剪枝
                            break;
                    }
                    else {
                        //到达深度限制或对方无子可下，则评估当前棋局并更新v值
                        newnode.score = evaluate(newnode.board, ai);
                        v = max(v, newnode.score);
                    }
                }
            }
            else {
                //MIN层节点
                v = 100000.0;
                for (int i = 0; i < avaiplaces.size(); ++i) {
                    Node newnode = Node(root.board, mode);		//新建子节点
                    newnode.action = avaiplaces[i];;			//记录该节点下子的位置
                    move(newnode.board, avaiplaces[i], color);	//下子后棋盘发生变化
                    int oppmode = mode == 1 ? 0 : 1;			//进入下一种mode
                    auto places = find_places(newnode.board, opp);	//得到对方可下子的位置
                    if (depth != 1 && places.size() != 0) {
                        //未到达深度限制且对方有地方下子，则递归搜索并更新v和beta值
                        v = min(v, alphabetapruning(newnode, ai, player, oppmode, depth - 1, alpha, beta));
                        beta = min(beta, v);
                        if (beta <= alpha)	//beta剪枝
                            break;
                    }
                    else {
                        //到达深度限制或对方无子可下，则评估当前棋局并更新v值
                        newnode.score = evaluate(newnode.board, ai);
                        v = min(v, newnode.score);
                    }
                }
            }
            //回溯时清空占用的内存
            root.children.clear();
            return v;
        }

        vector <double> get_average_all_weights(vector<weightstruct> weights) {
            // 计算n个权重向量的均值
            /*for (int i = 0; i < weights.size(); ++i) {
                for (int j = 0; j < 7; ++j)
                    cout << weights[i].subweight[j] << ' ';
                cout << endl;
            }*/
            vector <double> average_weight;
            //均值
            for (int i = 0; i < 7; i++) {
                average_weight.push_back(0);
                for (int j = 0; j < weights.size(); ++j)
                    average_weight[i] += weights[j].subweight[i];
                average_weight[i] /= weights.size();
            }
            //归一化
            double sum = 0;
            for (int i = 0; i < 7; ++i)
                sum += average_weight[i];
            for (int i = 0; i < 7; ++i)
                average_weight[i] /= sum;
            return average_weight;
        }

        int game_run() {
            cout << "请选择（1为黑棋，2为白棋）：";
            int choose = 1;
            cin >> choose;
            char player = choose == 1 ? 'X' : 'O';
            char ai = choose == 1 ? 'O' : 'X';
            cout << "\n游戏开始！\nX为黑棋，O为白棋，*为可落子位置\n\n";
            char current_turn = 'X';
            char board[8][8];
            vector<pair<int, int>> valid_pos;
            init_board(board);
            vector<weightstruct> weights;
            int n_game = 500;
            cout << "初始化中……请稍等……"<< endl;
            while (n_game--) {
                int n_step = 60;	//双方下60步
                while (n_step--) {
                    for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 8; j++)
                            board_copy[i][j] = board[i][j];
                    }
                    weights.push_back(get_weight_by_GA(current_turn));

                    valid_pos = find_places(board, current_turn);
                    if (valid_pos.empty()) {
                        current_turn = current_turn == 'X' ? 'O' : 'X';
                        valid_pos.clear();
                        continue;
                    }
                    if (current_turn == player)
                        random_play(board, player);
                    else
                        random_play(board, ai);
                    valid_pos.clear();
                    current_turn = current_turn == 'X' ? 'O' : 'X';
                }
                init_board(board);
                valid_pos.clear();
            }

            bestweight = get_average_all_weights(weights);
            init_board(board);
            valid_pos.clear();
            limit = 4;

            while (!is_over(board)) {
                valid_pos = find_places(board, current_turn);
                if (valid_pos.empty()) {
                    current_turn = current_turn == 'X' ? 'O' : 'X';
                    valid_pos.clear();
                    continue;
                }
                if (current_turn == player) {
                    print_board(board, valid_pos);
                    cout << "\n轮到玩家！";
                    player_play(board, valid_pos, current_turn);
                    //random_play(board, player);
                }
                else {
                    print_board(board, valid_pos);
                    ai_play(board, ai, limit);
                }
                valid_pos.clear();
                current_turn = current_turn == 'X' ? 'O' : 'X';
            }
            auto nums = count_number(board);
            print_board(board, valid_pos);
            cout << "游戏结束！" << endl;
            if (nums.first > nums.second) {
                cout << "黑棋胜利！" << endl;
                return 0;
            }
            else if (nums.first < nums.second) {
                cout << "白棋胜利！" << endl;
                return 1;
            }
            else {
                cout << "平局！" << endl;
                return 0;
            }
        }
};



#endif // GA_H
