import numpy as np
import time


def load_data(filename):
    with open(filename, 'r') as file:
        data = []
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            data.append(list(line))
    data = np.array(data)
    return data


def get_start_and_end(data):
    start = None
    end = None
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 'S':
                start = (i, j)
            elif data[i][j] == 'E':
                end = (i, j)
    return (start, end)


class IDS:
    def __init__(self, maze):
        '''
        参数说明：
        maze：用于存储迷宫图
        is_visited：用于记录迷宫的位置是否被访问过
        pre：用于记录每个位置上一步的位置信息
        limit：当前的深度限制，初始化为0
        reached：标记是否到达终点
        '''
        self.maze = maze
        self.is_visited = np.array([[False for i in range(len(maze[0]))] for j in range(len(maze))])
        self.pre = np.array([[None for i in range(len(maze[0]))] for j in range(len(maze))])
        self.limit = 0
        self.reached = False
        
    def search(self, pos, pre_pos, depth):
        '''
        对位置pos进行搜索，pre_pos为该位置的上一个位置，depth为当前深度
        '''
        x = pos[0]
        y = pos[1]
        #如果当前的深度超过的最大深度限制，结束搜索
        if depth > self.limit:
            return
        #如果当前位置是墙壁，则结束搜索
        if maze[x][y] == '1':
            return
        #如果当前位置是终点，则更新对应参数，结束搜索
        if self.maze[x][y] == 'E':
            self.reached = True
            self.pre[x][y] = pre_pos
            return
        #如果当前位置已经被访问过，结束搜索
        if self.is_visited[x][y] == True:
            return
        #将当前位置标记为已被访问，同时更新当前位置的上一步位置信息
        self.is_visited[x][y] = True
        self.pre[x][y] = pre_pos
        #分别沿左、上、右、下进行递归搜索
        if x-1>=0 and self.reached==False:
            self.search((x-1, y), pos, depth+1)
        if y-1>=0 and self.reached==False:
            self.search((x, y-1), pos, depth+1)
        if x+1<len(self.maze) and self.reached==False:
            self.search((x+1, y), pos, depth+1)
        if y+1<len(self.maze[0]) and self.reached==False:
            self.search((x, y+1), pos, depth+1)
        #回溯时将当前位置重新标记为未被访问
        self.is_visited[x][y] = False
                 
    def get_path(self, start, end):
        '''
        查找从起点到终点的路径，并将路径输出到txt文件
        '''
        while not self.reached:
            # 未到达目的地，则增加搜索深度
            self.limit += 1
            # 清除上次搜索的记录
            self.pre = np.array([[None for i in range(len(maze[0]))] for j in range(len(maze))])
            # 进行深度受限搜索
            self.search(start, None, 0)
        #获取路径上各个位置的坐标
        pos_list = []
        pos = end
        while self.pre[pos[0]][pos[1]] != None:
            pos_list.append(self.pre[pos[0]][pos[1]])
            pos = self.pre[pos[0]][pos[1]]
        #路径输出到txt文件
        with open('IDS_result.txt', 'w', encoding='utf-8') as file:
            for i in range(len(maze)):
                for j in range(len(maze[0])):
                    if (i, j) in pos_list and maze[i][j]!='S':
                        file.write('@')
                    elif self.maze[i][j] == '1':
                        file.write('#')
                    elif self.maze[i][j]=='S' or self.maze[i][j]=='E':
                        file.write(self.maze[i][j])
                    else:
                        file.write(' ')
                file.write('\n')
                


if __name__ == "__main__":
    maze = load_data('MazeData.txt')
    start, end = get_start_and_end(maze)

    start_time = time.time()
    ids = IDS(maze)
    ids.get_path(start, end)
    end_time = time.time()
    print('IDS time:', end_time-start_time)

