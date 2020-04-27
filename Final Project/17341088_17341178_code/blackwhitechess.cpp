#include "blackwhitechess.h"


BlackWhiteChess::BlackWhiteChess(int firstPlyer, int depth)
{
    setBestWeights();
    limit = depth;
    personValidPos = true;
    botValidPos = true;
    gs = PLAYING;
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            chessBoard[i][j] = 0;
        }
    }
    chessBoard[3][3] = chessBoard[4][4] = 2;
    chessBoard[3][4] = chessBoard[4][3] = 1;
    player = 1;
    if (firstPlyer == 1)
    {
        person = 1;
        bot = 2;
    }
    else
    {
        bot = 1;
        person = 2;
        botPlay();
    }


}

void BlackWhiteChess::random_play(int board[8][8], int color) {
    vector<pair<int, int>> valid_places = find_places(board, color);
    if (valid_places.size() == 0)
        return;
    int index = rand() % valid_places.size();
    move(board, valid_places[index], color);
}

void BlackWhiteChess::setBestWeights()
{
    //得到最优权重
    for (int i = 0; i < 7; i++)
        bestweight.push_back(1.0);

    int board[8][8];
    vector<pair<int, int>> valid_pos;
    init_board(board);
    vector<weightstruct> weights;
    int n_game = 10;
    int current_turn = 1;
    int player = 1;
    int ai = 2;
    while (n_game--) {
        int n_step = 60;	//双方下60步
        while (n_step--) {
            weights.push_back(get_weight_by_GA(board, current_turn));

            valid_pos = find_places(board, current_turn);
            if (valid_pos.empty()) {
                current_turn = current_turn == 1 ? 2 : 1;
                valid_pos.clear();
                continue;
            }
            if (current_turn == player)
                random_play(board, player);
            else
                random_play(board, ai);
            valid_pos.clear();
            current_turn = current_turn == 1 ? 2 : 1;
        }
        init_board(board);
        valid_pos.clear();
    }

    bestweight = get_average_all_weights(weights);
}

vector<double> BlackWhiteChess::get_average_all_weights(vector<weightstruct> weights) {
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


void BlackWhiteChess::init_board(int board[8][8])
{
    // 对棋盘初始化
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            chessBoard[i][j] = 0;
        }
    }
    board[3][3] = board[4][4] = 2;
    board[3][4] = board[4][3] = 1;
}

bool BlackWhiteChess::setChess(int chess, int posx, int posy)
{
    //judge no chess
    if (chessBoard[posx][posy] != 0) return false;
    chessBoard[posx][posy] = chess;
    //judge blank
    bool existBlank = true;
    auto inChessBoard = [] (int posx, int posy)->bool
    {
        return posx >= 0 && posx < 8 && posy >= 0 && posy < 8;
    };
    for (int i = -1; i <= 1 && existBlank; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            if (i != 0 || j != 0)
            {
                if (inChessBoard(posx+i, posy+j) && chessBoard[posx+i][posy+j] != 0)
                {
                    existBlank = false;
                    break;
                }
            }
        }
    }
    if (existBlank) return false;
    //judge change
    bool isChange = false;
    bool vCheck, hCheck, dCheck;
    vCheck = verticalCheck(chess, posx, posy);
    hCheck = horizontalCheck(chess, posx, posy);
    dCheck = dialogCheck(chess, posx, posy);
    isChange = isChange || vCheck || hCheck || dCheck;
    return isChange;
}

bool BlackWhiteChess::dialogCheck(int chess, int posx, int posy)
{
    //leftup-rightdown
    bool leftUpChange, rightDownChange;
    leftUpChange = rightDownChange = false;
    int leftUpLimit, rightDownLimit;
    leftUpLimit = rightDownLimit = -1;
    int x,y;
    for (int i = 1; i < 8; ++i)
    {
        x = posx - i;
        y = posy - i;
        if (x >= 0 && y >= 0)
        {
            if (chessBoard[x][y] == 0) break;
            if (chessBoard[x][y] == chess)
            {
                leftUpLimit = i;
                break;
            }
        }
    }
    for (int i = 1; i < 8; ++i)
    {
        x = posx + i;
        y = posy + i;
        if (x < 8 && y < 8)
        {
            if (chessBoard[x][y] == 0) break;
            if (chessBoard[x][y] == chess)
            {
                rightDownLimit = i;
                break;
            }
        }
    }
    if (leftUpLimit > 1)
    {
        for (int i = 1; i <= leftUpLimit; ++i) chessBoard[posx-i][posy-i] = chess;
        leftUpChange = true;
    }
    if (rightDownLimit > 1)
    {
        for (int i = 1; i <= rightDownLimit; ++i) chessBoard[posx+i][posy+i] = chess;
        leftUpChange = true;
    }
    //leftdown-rightup
    bool leftDownChange, rightUpChange;
    leftDownChange = rightUpChange = false;
    int leftDownLimit, rightUpLimit;
    leftDownLimit = rightUpLimit = -1;
    for (int i = 1; i < 8; ++i)
    {
        x = posx + i;
        y = posy - i;
        if (x < 8 && y >= 0)
        {
            if (chessBoard[x][y] == 0) break;
            if (chessBoard[x][y] == chess)
            {
                leftDownLimit = i;
                break;
            }
        }
    }
    for (int i = 1; i < 8; ++i)
    {
        x = posx - i;
        y = posy + i;
        if (x >= 0 && y < 8)
        {
            if (chessBoard[x][y] == 0) break;
            if (chessBoard[x][y] == chess)
            {
                rightUpLimit = i;
                break;
            }
        }
    }
    if (leftDownLimit > 1)
    {
        for (int i = 1; i <= leftDownLimit; ++i) chessBoard[posx+i][posy-i] = chess;
        leftDownChange = true;
    }
    if (rightUpLimit > 1)
    {
        for (int i = 1; i <= rightUpLimit; ++i) chessBoard[posx-i][posy+i] = chess;
        rightUpChange = true;
    }
    return leftUpChange || rightDownChange || leftDownChange || rightUpChange;
}

bool BlackWhiteChess::verticalCheck(int chess, int posx, int posy)
{
    bool upChange, downChange;
    upChange = downChange = false;
    int topLimit, bottomLimit;
    topLimit = bottomLimit = -1;
    for (int x = posx-1; x >= 0; x--)
    {
        if (chessBoard[x][posy] == 0) break;
        if (chessBoard[x][posy] == chess)
        {
            topLimit = x;
            break;
        }
    }
    for (int x = posx+1; x < 8; x++)
    {
        if (chessBoard[x][posy] == 0) break;
        if (chessBoard[x][posy] == chess)
        {
            bottomLimit = x;
            break;
        }
    }
    if (topLimit != -1 && posx - topLimit > 1)
    {
        for (int x = posx-1; x >= topLimit; x--) chessBoard[x][posy] = chess;
        upChange = true;
    }
    if (bottomLimit != -1 && bottomLimit - posx > 1)
    {
        for (int x = posx+1; x <= bottomLimit; x++) chessBoard[x][posy] = chess;
        downChange = true;
    }
    return upChange || downChange;
}

bool BlackWhiteChess::horizontalCheck(int chess, int posx, int posy)
{
   bool leftChange, rightChange;
   leftChange = rightChange = false;
   int leftLimit, rightLimit;
   leftLimit = rightLimit = -1;
   for (int y = posy-1; y >= 0; y--)
   {
       if (chessBoard[posx][y] == 0) break;
       if (chessBoard[posx][y] == chess)
       {
           leftLimit = y;
           break;
       }
   }
   for (int y = posy+1; y < 8; y++)
   {
       if (chessBoard[posx][y] == 0) break;
       if (chessBoard[posx][y] == chess)
       {
           rightLimit = y;
           break;
       }
   }
   if (leftLimit != -1 && posy - leftLimit > 1)
   {
       for (int y = posy-1; y >= leftLimit; y--) chessBoard[posx][y] = chess;
       leftChange = true;
   }
   if (rightLimit != -1 && rightLimit - posy > 1)
   {
       for (int y = posy+1; y <= rightLimit; y++) chessBoard[posx][y] = chess;
       rightChange = true;
   }
   return leftChange || rightChange;
}

void BlackWhiteChess::play(int posx, int posy)
{
    std::cout << posx << " : " << posy <<std::endl;

    bool isset = setChess(player, posx, posy);
    if (!isset) chessBoard[posx][posy] = 0;
    int b_num = 0, w_num = 0;
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; ++j)
        {
            std::cout << chessBoard[i][j] << " ";
            if (chessBoard[i][j] == 1)
                b_num++;
            else if (chessBoard[i][j] == 2)
                w_num++;
        }
        std::cout << std::endl;
    }
    cout << "Black : White = " << b_num << " : " << w_num << endl << endl;
    if (b_num == 0)
        gs = WHITEWIN;
    else if(w_num == 0)
        gs = BLACKWIN;
    bool personValid, botValid;
    personValid = botValid = false;
    int record[8][8];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            record[i][j] = chessBoard[i][j];
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            if (setChess(person, i, j))
            {
                personValid = true;
                break;
            }
        }
    }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; ++j)
            chessBoard[i][j] = record[i][j];
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; ++j)
        {
            if (setChess(bot, i, j))
            {
                botValid = true;
                break;
            }
        }
    }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; ++j)
            chessBoard[i][j] = record[i][j];
    if (!personValid && !botValid)
    {
        checkGameState();
    }
    else
    {
        if (player == person)
        {
            if ((!isset && personValid) || !botValid )
                player = person;
            else
                player = bot;
        }
        else if (player == bot)
        {
            if ((!isset && botValid) || !personValid )
                player = bot;
            else
                player = person;
        }
        //std::cout << player << std::endl << std::endl;
    }
}

void BlackWhiteChess::checkGameState()
{
    int blackNum, whiteNum;
    blackNum = whiteNum = 0;
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            if (chessBoard[i][j] == 1) blackNum++;
            else if (chessBoard[i][j] == 2) whiteNum++;
        }
    }
    if (blackNum > whiteNum) gs = BLACKWIN;
    else if (blackNum < whiteNum) gs = WHITEWIN;
    else gs = DOGFALL;

}
GameState BlackWhiteChess::getGameState() const
{
    return gs;
}
int BlackWhiteChess::getPlayer() const
{
    return player;
}
void BlackWhiteChess::setGameEnd()
{
    gs = END;
}
void BlackWhiteChess::botPlay()
{
    //play(4, 2);
    //play(算法输出坐标)


    Node root = Node(chessBoard, bot);
    int index = alphabetapruning(root, 1, limit, -100000.0, 100000.0);
    if (index == -1) {
        player = person;
        return;
    }
    int x = root.children[index].action.first;
    int y = root.children[index].action.second;
    play(x, y);
    root.children.clear();
}




bool BlackWhiteChess::is_on_board(int row, int col) {
    // 判断某个位置是否在棋盘上
    if (row < 8 && row >= 0 && col < 8 && col >= 0)
        return true;
    return false;
}

vector<int> BlackWhiteChess::find_filp_dirs(int board[8][8], pair<int, int> pos, int color) {
    // 获取某个位置能够翻转对方棋子的方向
    int x = pos.first, y = pos.second;
    vector<int> dirs;
    int opp = color == 1 ? 2 : 1;
    for (int i = 0; i < 8; i++) {
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

vector<pair<int, int>> BlackWhiteChess::find_places(int board[8][8], int turn) {
    // 寻找可下子位置，turn标记是黑棋落子还是白棋落子
    vector<pair<int, int>> valid_pos;
    int item = turn == 1 ? 1 : 2;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            if (board[i][j] == 0 && !find_filp_dirs(board, make_pair(i, j), item).empty())
                valid_pos.push_back(make_pair(i, j));
    return valid_pos;
}

void BlackWhiteChess::flip(int board[8][8], pair<int, int> action, int color) {
    // 根据落子位置，执行翻转操作
    vector<int> dirs = find_filp_dirs(board, action, color);
    int opp = color == 1 ? 2 : 1;
    int x = action.first, y = action.second;
    for (int i = 0; i < dirs.size(); i++) {
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

void BlackWhiteChess::move(int board[8][8], pair<int, int> pos, int color) {
    // 在pos处落子
    board[pos.first][pos.second] = color;
    flip(board, pos, color);
}

double BlackWhiteChess::calPosEval(int board[8][8], int color){
    // 计算位置权重评估值
    int opp = color == 1 ? 2 : 1;
    double poseval = 0;
    for (int i = 0; i < 8; ++i){
        for (int j = 0; j < 8; ++j){
            if (board[i][j] == color)
                poseval += square_weights[i][j];
            else if (board[i][j] == opp)
                poseval -= square_weights[i][j];
        }
    }
    return poseval;
}

double BlackWhiteChess::calRateEval(int board[8][8], int color){
    // 计算黑白子比例评估值
    int opp = color == 1 ? 2 : 1;
    int mycount = 0, opcount = 0;
    for (int i = 0; i < 8; ++i){
        for (int j = 0; j < 8; ++j){
            if (board[i][j] == color)
                mycount++;
            else if (board[i][j] == opp)
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

double BlackWhiteChess::calCornerEval(int board[8][8], int color){
    // 计算占角评估值
    int opp = color == 1 ? 2 : 1;
    int corner_pos[4][2] = { { 0, 0 },{ 0, 7 },{ 7, 0 },{ 7, 7 } };
    int mycorner = 0, opcorner = 0;
    for (int i = 0; i < 4; ++i){
        if (board[corner_pos[i][0]][corner_pos[i][1]] == color)
            mycorner++;
        else if (board[corner_pos[i][0]][corner_pos[i][1]] == opp)
            opcorner++;
    }
    return 100 * (mycorner - opcorner);
}

double BlackWhiteChess::calNearCornerEval(int board[8][8], int color){
    // 计算近角评估值
    int opp = color == 1 ? 2 : 1;
    int corner_pos[4][2] = { { 0, 0 },{ 0, 7 },{ 7, 0 },{ 7, 7 } };
    int mynear = 0, oppnear = 0;
    for (int i = 0; i < 4; i++) {
        int x = corner_pos[i][0], y = corner_pos[i][1];
            if (board[x][y] == 0)
                for (int j = 0; j < 8; j++) {
                    int nx = x + direction[j][0], ny = y + direction[j][1];
                    if (is_on_board(nx, ny))
                        if (board[nx][ny] == color)
                            mynear++;
                        else if (board[nx][ny] == opp)
                            oppnear++;
                }
    }
    return -100 * (mynear - oppnear);
}

double BlackWhiteChess::calMoveEval(int board[8][8], int color){
    // 计算行动力评估值
    int opp = color == 1 ? 2 : 1;
    int mymove = find_places(board, color).size();
    int opmove = find_places(board, opp).size();
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

bool BlackWhiteChess::is_stable(int board[8][8], int x, int y) {
    // 判断某个位置是否为稳定点
    for (int i = 0; i < 8; ++i)
        for (int nx = x + direction[i][0], ny = y + direction[i][1]; is_on_board(nx, ny); nx += direction[i][0], ny += direction[i][1])
            if (board[nx][ny] == 0)
                return false;
    return true;
}

double BlackWhiteChess::calStableEval(int board[8][8], int color){
    // 计算稳定子评估值
    int mystable = 0, oppstable = 0;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            if (board[i][j] != 0 && is_stable(board, i, j))
                if (board[i][j] == color)
                    mystable++;
                else
                    oppstable++;
    return 12.5 * (mystable - oppstable);
}

double BlackWhiteChess::calSideEval(int board[8][8], int color){
    //计算边界棋子评估值
    int opp = color == 1 ? 2 : 1;
    int index[2] = {0, 7};
    int myside = 0, oppside = 0;
    for (int i = 0; i < 2; i++){
        for (int j = 2; j < 6; j++){
            if (board[index[i]][j] == color)
                myside++;
            else if (board[index[i]][j] == opp)
                oppside++;
        }
    }
    for (int i = 0; i < 2; i++){
        for (int j = 2; j < 6; j++){
            if (board[j][index[i]] == color)
                myside++;
            else if (board[index[i]][j] == opp)
                oppside++;
        }
    }
    return 10 * (myside - oppside);
}

double BlackWhiteChess::calEval(vector<double> weight, vector<double> subeval){
    // 计算总的评估值
    double eval = 0;
    for (int i = 0; i < 7; i++)
        eval += weight[i] * subeval[i];
    return eval;
}

double BlackWhiteChess::evaluate(int board[8][8], int color, vector<double> weight) {
    //计算整个棋局评估值
    vector<double> eval;
    eval.push_back(calPosEval(board, color));
    eval.push_back(calMoveEval(board, color));
    eval.push_back(calSideEval(board, color));
    eval.push_back(calCornerEval(board, color));
    eval.push_back(calRateEval(board, color));
    eval.push_back(calStableEval(board, color));
    eval.push_back(calNearCornerEval(board, color));
    return calEval(weight, eval);
}


double BlackWhiteChess::alphabetapruning(Node &root, int mode, int depth, double alpha, double beta) {
    //mode=0时：MAX层节点，mode=1：MIN层节点
    int color = mode == 1 ? bot : player;
    int opp = color == 1 ? 2 : 1;
    vector<pair<int, int>> avaiplaces = find_places(root.board, color);	//得到可下子的位置
    if (avaiplaces.size() == 0)
        return -1;
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
                newnode.score = alphabetapruning(newnode, oppmode, depth - 1, alpha, beta);
            else{
                //如果对方没有地方下子，则评估当前棋局
                newnode.score = evaluate(newnode.board, color, bestweight);
            }
            root.children.push_back(newnode);			//加入新的子节点
        }
        int index = -1;
        double max = -100000.0;
        //得到估价值最高的走法
        cout << "children size : " << root.children.size() << endl;
        for (int i = 0; i < root.children.size(); i++)
            if (root.children[i].score >= max) {
                index = i;
                max = root.children[i].score;
            }
        //返回估价值最高的走法
        cout << "max child index : " << index << endl;
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
                v = max(v, alphabetapruning(newnode, oppmode, depth - 1, alpha, beta));
                alpha = max(alpha, v);
                if (beta <= alpha)	//alpha剪枝
                    break;
            }
            else {
                //到达深度限制或对方无子可下，则评估当前棋局并更新v值
                newnode.score = evaluate(newnode.board, bot, bestweight);
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
                v = min(v, alphabetapruning(newnode, oppmode, depth - 1, alpha, beta));
                beta = min(beta, v);
                if (beta <= alpha)	//beta剪枝
                    break;
            }
            else {
                //到达深度限制或对方无子可下，则评估当前棋局并更新v值
                newnode.score = evaluate(newnode.board, bot, bestweight);
                v = min(v, newnode.score);
            }
        }
    }
    //回溯时清空占用的内存
    root.children.clear();
    return v;
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

void BlackWhiteChess::copy(vector<weightstruct>& weightvec, vector<weightstruct>& new_weightvec){
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

void BlackWhiteChess::cross(vector<weightstruct>& weightvec, vector<weightstruct>& new_weightvec){
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

void BlackWhiteChess::mutate(vector<weightstruct>& weightvec, vector<weightstruct>& new_weightvec){
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

weightstruct BlackWhiteChess::ga(vector<double> subeval) {
    // 遗传算法计算子评估值的权重向量
    srand((unsigned int)time(0));
    vector<weightstruct> weightvec = init_weightvec(10);
    int iter = 20;	// 进行20次迭代
    while (iter--) {
        // 对当前10个向量两两进行比较
        for (int i = 0; i < 10; i++) {
            for (int j = i + 1; j < 10; j++) {
                if (calEval(weightvec[i].subweight, subeval) > calEval(weightvec[j].subweight, subeval))
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

weightstruct BlackWhiteChess::get_weight_by_GA(int board[8][8], int color) {
    // 利用遗传算法得到权重向量
    vector<double> eval;
    eval.push_back(calPosEval(board, color));
    eval.push_back(calMoveEval(board, color));
    eval.push_back(calSideEval(board, color));
    eval.push_back(calCornerEval(board, color));
    eval.push_back(calRateEval(board, color));
    eval.push_back(calStableEval(board, color));
    eval.push_back(calNearCornerEval(board, color));
    return ga(eval);
}

