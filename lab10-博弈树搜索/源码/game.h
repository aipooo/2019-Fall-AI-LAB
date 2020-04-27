//ai

#ifndef GAME_H
#define GAME_H
#include "config.h" 
#include<iostream>
#include<cstring>
using namespace std;

class Game :CONFIG {
public:
    ChessBoard curState; 	// 当前棋盘
    bool isStart; 			// 是否进行中
    int curUser; 			// 当前行棋人
    int MAX_DEPTH; 			// 最大搜索层数

    void startGame(int nd = 2) {
        MAX_DEPTH = nd;
        isStart = true;
        curUser = USER_1;
    }

    //转换行棋人 
    void changeUser() {
        curUser = curUser == USER_1 ? USER_2 : USER_1;
    }

    /*
    根据给定type
    A:待判断棋子的类型
    type:我方棋子的类型
    返回A是待判断棋子 无棋子 对方棋子
    */
    int getPieceType(int A, int type) {
        return A == type ? AI_MY : (A == EMPTY ? AI_EMPTY : AI_OP);
    }

    int getPieceType(const ChessBoard &board, int x, int y, int type) {
        if (x < 0 || y < 0 || x >= BOARD_SIZE || y >= BOARD_SIZE)// 超出边界按对方棋子算
            return AI_OP;
        else
            return getPieceType(board.chessBoard[x][y].type, type);
    }

    /*
    当前行棋人放置棋子
    放置失败返回失败
    放置成功
    检察游戏是否结束
    转换游戏角色后返回成功
    */
    bool placePiece(int x, int y) {
        if (curState.placePiece(x, y, curUser)) {
            // 检察行棋人是否胜利
            if (isWin(x, y)) {
                isStart = false; // 游戏结束
                                 //  return true;
            }
            changeUser(); // 转换游戏角色
            return true;
        }
        return false;
    }

    bool isWin(int x, int y) {
        if (evaluatePiece(curState, x, y, curUser) >= AI_FIVE)
            return true;
        return false;
    }

    //以center作为评估位置进行评价一个方向的棋子
    int evaluateLine(int line[], bool ALL) {
        int value = 0; 		//估值
        int cnt = 0; 		//连子数
        int blk = 0; 		//封闭数
        for (int i = 0; i < BOARD_SIZE; ++i) {
            if (line[i] == AI_MY) { 	//找到第一个己方的棋子,还原计数
                cnt = 1;
                blk = 0;
                //看左侧是否封闭
                if (line[i - 1] == AI_OP)
                    ++blk;
                //计算连子数
                for (i = i + 1; i < BOARD_SIZE && line[i] == AI_MY; ++i, ++cnt);
                //看右侧是否封闭
                if (line[i] == AI_OP)
                    ++blk;
                //计算评估值
                value += getValue(cnt, blk);
            }
        }
        return value;
    }

    //以center作为评估位置进行评价一个方向的棋子（前后4格范围内）
    int evaluateLine(int line[]) {
        int cnt = 1; // 连子数
        int blk = 0; // 封闭数
                     // 向左右扫
        for (int i = 3; i >= 0; --i) {
            if (line[i] == AI_MY) ++cnt;
            else if (line[i] == AI_OP) {
                ++blk;
                break;
            }
            else
                break;
        }
        for (int i = 5; i < 9; ++i) {
            if (line[i] == AI_MY) ++cnt;
            else if (line[i] == AI_OP) {
                ++blk;
                break;
            }
            else
                break;
        }
        return getValue(cnt, blk);
    }

    //根据连字数和封堵数给出一个评价值
    int getValue(int cnt, int blk) {
        if (blk == 0) {			//活棋
            switch (cnt) {
            case 1:
                return AI_ONE;
            case 2:
                return AI_TWO;
            case 3:
                return AI_THREE;
            case 4:
                return AI_FOUR;
            default:
                return AI_FIVE;
            }
        }
        else if (blk == 1) {	//单向封死
            switch (cnt) {
            case 1:
                return AI_ONE_S;
            case 2:
                return AI_TWO_S;
            case 3:
                return AI_THREE_S;
            case 4:
                return AI_FOUR_S;
            default:
                return AI_FIVE;
            }
        }
        else {					//双向堵死
            if (cnt >= 5)
                return AI_FIVE;
            else
                return AI_ZERO;
        }
    }

    //对一个状态的一个位置放置一种类型的棋子的优劣进行估价
    int evaluatePiece(ChessBoard state, int x, int y, int type) {
        int value = 0; 			//估价值
        int line[17];  			//线状态
        bool flagX[8];			//横向边界标志
        flagX[0] = x - 4 < 0;
        flagX[1] = x - 3 < 0;
        flagX[2] = x - 2 < 0;
        flagX[3] = x - 1 < 0;
        flagX[4] = x + 1 > 14;
        flagX[5] = x + 2 > 14;
        flagX[6] = x + 3 > 14;
        flagX[7] = x + 4 > 14;
        bool flagY[8];			// 纵向边界标志
        flagY[0] = y - 4 < 0;
        flagY[1] = y - 3 < 0;
        flagY[2] = y - 2 < 0;
        flagY[3] = y - 1 < 0;
        flagY[4] = y + 1 > 14;
        flagY[5] = y + 2 > 14;
        flagY[6] = y + 3 > 14;
        flagY[7] = y + 4 > 14;

        line[4] = AI_MY; 		// 中心棋子
        
		// 横
        line[0] = flagX[0] ? AI_OP : (getPieceType(state.chessBoard[x - 4][y].type, type));
        line[1] = flagX[1] ? AI_OP : (getPieceType(state.chessBoard[x - 3][y].type, type));
        line[2] = flagX[2] ? AI_OP : (getPieceType(state.chessBoard[x - 2][y].type, type));
        line[3] = flagX[3] ? AI_OP : (getPieceType(state.chessBoard[x - 1][y].type, type));

        line[5] = flagX[4] ? AI_OP : (getPieceType(state.chessBoard[x + 1][y].type, type));
        line[6] = flagX[5] ? AI_OP : (getPieceType(state.chessBoard[x + 2][y].type, type));
        line[7] = flagX[6] ? AI_OP : (getPieceType(state.chessBoard[x + 3][y].type, type));
        line[8] = flagX[7] ? AI_OP : (getPieceType(state.chessBoard[x + 4][y].type, type));

        value += evaluateLine(line);

        // 纵
        line[0] = flagY[0] ? AI_OP : getPieceType(state.chessBoard[x][y - 4].type, type);
        line[1] = flagY[1] ? AI_OP : getPieceType(state.chessBoard[x][y - 3].type, type);
        line[2] = flagY[2] ? AI_OP : getPieceType(state.chessBoard[x][y - 2].type, type);
        line[3] = flagY[3] ? AI_OP : getPieceType(state.chessBoard[x][y - 1].type, type);

        line[5] = flagY[4] ? AI_OP : getPieceType(state.chessBoard[x][y + 1].type, type);
        line[6] = flagY[5] ? AI_OP : getPieceType(state.chessBoard[x][y + 2].type, type);
        line[7] = flagY[6] ? AI_OP : getPieceType(state.chessBoard[x][y + 3].type, type);
        line[8] = flagY[7] ? AI_OP : getPieceType(state.chessBoard[x][y + 4].type, type);

        value += evaluateLine(line);

        // 左上-右下
        line[0] = flagX[0] || flagY[0] ? AI_OP : getPieceType(state.chessBoard[x - 4][y - 4].type, type);
        line[1] = flagX[1] || flagY[1] ? AI_OP : getPieceType(state.chessBoard[x - 3][y - 3].type, type);
        line[2] = flagX[2] || flagY[2] ? AI_OP : getPieceType(state.chessBoard[x - 2][y - 2].type, type);
        line[3] = flagX[3] || flagY[3] ? AI_OP : getPieceType(state.chessBoard[x - 1][y - 1].type, type);

        line[5] = flagX[4] || flagY[4] ? AI_OP : getPieceType(state.chessBoard[x + 1][y + 1].type, type);
        line[6] = flagX[5] || flagY[5] ? AI_OP : getPieceType(state.chessBoard[x + 2][y + 2].type, type);
        line[7] = flagX[6] || flagY[6] ? AI_OP : getPieceType(state.chessBoard[x + 3][y + 3].type, type);
        line[8] = flagX[7] || flagY[7] ? AI_OP : getPieceType(state.chessBoard[x + 4][y + 4].type, type);

        value += evaluateLine(line);

        // 右上-左下
        line[0] = flagX[7] || flagY[0] ? AI_OP : getPieceType(state.chessBoard[x + 4][y - 4].type, type);
        line[1] = flagX[6] || flagY[1] ? AI_OP : getPieceType(state.chessBoard[x + 3][y - 3].type, type);
        line[2] = flagX[5] || flagY[2] ? AI_OP : getPieceType(state.chessBoard[x + 2][y - 2].type, type);
        line[3] = flagX[4] || flagY[3] ? AI_OP : getPieceType(state.chessBoard[x + 1][y - 1].type, type);

        line[5] = flagX[3] || flagY[4] ? AI_OP : getPieceType(state.chessBoard[x - 1][y + 1].type, type);
        line[6] = flagX[2] || flagY[5] ? AI_OP : getPieceType(state.chessBoard[x - 2][y + 2].type, type);
        line[7] = flagX[1] || flagY[6] ? AI_OP : getPieceType(state.chessBoard[x - 3][y + 3].type, type);
        line[8] = flagX[0] || flagY[7] ? AI_OP : getPieceType(state.chessBoard[x - 4][y + 4].type, type);

        value += evaluateLine(line);

        return value;
    }

    //评价一个棋面上的一方
    int evaluateState(ChessBoard state, int type) {
        int value = 0;
        // 分解成线状态
        int line[6][17];
        int lineP;

        for (int p = 0; p < 6; ++p)
            line[p][0] = line[p][16] = AI_OP;

        //从四个方向产生
        for (int i = 0; i < BOARD_SIZE; ++i) {
            //产生线状态
            lineP = 1;
            for (int j = 0; j < BOARD_SIZE; ++j) {
                line[0][lineP] = getPieceType(state, i, j, type); 						/* | */
                line[1][lineP] = getPieceType(state, j, i, type); 						/* - */
                line[2][lineP] = getPieceType(state, i + j, j, type); 					/* \ */
                line[3][lineP] = getPieceType(state, i - j, j, type); 					/* / */
                line[4][lineP] = getPieceType(state, j, i + j, type); 					/* \ */
                line[5][lineP] = getPieceType(state, BOARD_SIZE - j - 1, i + j, type); /* / */
                ++lineP;
            }
            //估计
            int special = i == 0 ? 4 : 6;
            for (int p = 0; p < special; ++p) {
                value += evaluateLine(line[p], true);
            }
        }
        return value;
    }

    //若x, y位置周围1格内有棋子则搜索
    bool canSearch(ChessBoard state, int x, int y) {

        int tmpx = x - 1;
        int tmpy = y - 1;
        for (int i = 0; tmpx < BOARD_SIZE && i < 3; ++tmpx, ++i) {
            int ty = tmpy;
            for (int j = 0; ty < BOARD_SIZE && j < 3; ++ty, ++j) {
                if (tmpx >= 0 && ty >= 0 && state.chessBoard[tmpx][ty].type != EMPTY)
                    return true;
                else
                    continue;
            }
        }
        return false;
    }

    //给出后继节点的类型
    int nextType(int type) {
        return type == MAX_NODE ? MIN_NODE : MAX_NODE;
    }

    /*************************************************
    参数说明：
	state 待转换的状态
    type  当前层的标记：MAX MIN
    depth 当前层深
    alpha 父层alpha值
    beta  父层beta值
    *************************************************/
    int minMax(ChessBoard state, int x, int y, int type, int depth, int alpha, int beta) {
        ChessBoard newState(state);
        newState.placePiece(x, y, nextType(type));
        int weight = 0;
        int max = -INF; 	//下层权值上界
        int min = INF; 		//下层权值下界

        if (depth < MAX_DEPTH) {
            //已输或已胜则不继续搜索
            if (evaluatePiece(newState, x, y, nextType(type)) >= AI_FIVE) {
                if (type == MIN_NODE)
                    return AI_FIVE; 	//我方胜
                else
                    return -AI_FIVE;	//我方负
            }

            int i, j;
            for (i = 0; i < BOARD_SIZE; ++i) {
                for (j = 0; j < BOARD_SIZE; ++j) {
                    if (newState.chessBoard[i][j].type == EMPTY && canSearch(newState, i, j)) {
                        weight = minMax(newState, i, j, nextType(type), depth + 1, min, max);
                        if (weight > max)
                            max = weight; 	//更新下层上界
                        if (weight < min)
                            min = weight; 	//更新下层下界
                        // alpha-beta
                        if (type == MAX_NODE) {
                            if (max >= alpha)
                                return max;
                        }
                        else {
                            if (min <= beta)
                                return min;
                        }
                    }
                    else
                        continue;
                }
            }
            if (type == MAX_NODE)
                return max; //最大层给出最大值
            else
                return min; //最小层给出最小值
        }
        else {
			//评估我方局面
            weight = evaluateState(newState, MAX_NODE);
			// 评估对方局面
            weight -= type == MIN_NODE ? evaluateState(newState, MIN_NODE)*10 : evaluateState(newState, MIN_NODE); 
            return weight; 		//搜索到限定层后给出权值
        }
    }


    int cnt[BOARD_SIZE][BOARD_SIZE];
    
    //AI行棋
    bool placePieceAI() {
        int weight;
        int max = -INF; // 本层的权值上界
        int x = 0, y = 0;
        memset(cnt, 0, sizeof(cnt));
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                if (curState.chessBoard[i][j].type == EMPTY && canSearch(curState, i, j)) {
                    weight = minMax(curState, i, j, nextType(MAX_NODE), 1, -INF, max);
                    cnt[i][j] = weight;
                    if (weight > max) {
                        max = weight; // 更新下层上界
                        x = i;
                        y = j;
                    }
                }
                else
                    continue;
            }
        }
        return placePiece(x, y); // AI最优点
    }

    //控制台打印
    void show() {
		curState.print_board();
    }

};

#endif
