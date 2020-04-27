//参数

#ifndef CONFIG_H
#define CONFIG_H

class CONFIG {
	public:
	    static const int BOARD_SIZE = 11;
	    static const int EMPTY = 0;
	    static const int USER_1 = 1;
	    static const int USER_2 = 2;
	    static const int AI_EMPTY = 0; // 无子
	    static const int AI_MY = 1; // 待评价子
	    static const int AI_OP = 2; // 对方子或不能下子
	    static const int MAX_NODE = 2;
	    static const int MIN_NODE = 1;
	    static const int INF = 106666666;
	    static const int ERROR_INDEX = -1;
	    //估价值
	    static const int AI_ZERO = 0;
	    static const int AI_ONE = 10;
	    static const int AI_ONE_S = 1;
	    static const int AI_TWO = 100;
	    static const int AI_TWO_S = 10;
	    static const int AI_THREE = 1000;
	    static const int AI_THREE_S = 100;
	    static const int AI_FOUR = 10000;
	    static const int AI_FOUR_S = 1000;
	    static const int AI_FIVE = 100000;
};

#endif
