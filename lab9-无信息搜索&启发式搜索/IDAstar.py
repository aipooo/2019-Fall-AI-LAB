import numpy as np
import math
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

class Node:
    def __init__(self, pos, end):
        self.pos = pos
        self.father = None
        self.g = 0
        #启发式函数采用该位置到终点的曼哈顿距离
        self.h = abs(self.pos[0]-end[0]) + abs(self.pos[1]-end[1])
        #启发式函数采用该位置到终点的对角线距离
        #self.h = math.sqrt((self.pos[0]-end[0])**2+(self.pos[1]-end[1])**2)

class IDAstar:
    def __init__(self, maze, start, end):
        '''
        参数说明：
        maze：用于存储迷宫图
        start_node：起点
        end_node：终点
        visited：存储已访问过的所有节点
        f：存储所有下一步可访问的，超过限定的节点的f值
        reached：标记是否到达终点
        last_node：标记最后一个访问的节点（若有解则为终点)
        '''
        self.maze = maze
        self.start_node = Node(start, end)
        self.end_node = Node(end, end)
        self.visited = []
        self.f = []
        self.reached = False
        self.last_node = None
    
    def search_node(self, pos, depth):
        # 更新新节点的g和h值
        node = Node(pos, self.end_node.pos)
        node.g = depth
        return node

    def subsearch(self, node, pre_node, depth, limit):
        #节点的f值大于阈值，结束搜索，将该f值加入f列表
        if node.g + node.h > limit:
            self.f.append(node.g + node.h)
            return
        #如果当前位置是墙壁，则结束搜索
        if self.maze[node.pos[0]][node.pos[1]] == '1':
            return
        #如果当前位置是终点，则更新对应参数，结束搜索
        if node.pos == self.end_node.pos:
            node.father = pre_node
            self.reached = True
            self.last_node = node
            return
        #记录父节点
        node.father = pre_node
        #将节点加入已访问的列表
        self.visited.append(node.pos)
        x = node.pos[0]
        y = node.pos[1]
        #递归搜索所有相邻节点
        if x+1<len(self.maze) and not self.reached:
            next_pos = (x+1, y)
            #如果节点不在已访问的列表中，更新新节点的g和h值
            if next_pos not in self.visited:
                next_node = self.search_node(next_pos, depth)
                self.subsearch(next_node, node, depth + 1, limit)
        if x-1>=0 and not self.reached:
            next_pos = (x-1, y)
            #如果节点不在已访问的列表中，更新新节点的g和h值
            if next_pos not in self.visited:
                next_node = self.search_node(next_pos, depth)
                self.subsearch(next_node, node, depth + 1, limit)
        if y+1<len(self.maze[0]) and not self.reached:
            next_pos = (x, y+1)
            #如果节点不在已访问的列表中，更新新节点的g和h值
            if next_pos not in self.visited:
                next_node = self.search_node(next_pos, depth)
                self.subsearch(next_node, node, depth + 1, limit)
        if y - 1 >= 0 and not self.reached:
            next_pos = (x, y-1)
            #如果节点不在已访问的列表中，更新新节点的g和h值
            if next_pos not in self.visited:
                next_node = self.search_node(next_pos, depth)
                self.subsearch(next_node, node, depth + 1, limit)
        #回溯时将节点从已访问列表中删除
        self.visited.remove(node.pos)

    def search(self):
        # 阈值设为初始节点的h值
        limit = self.start_node.h
        while not self.reached:
            #如果没有找到目标节点，将阈值设为f列表中最小值
            self.subsearch(self.start_node, None, 1, limit)
            limit = min(self.f)
            self.f = []
        #获取路径上各个位置的坐标
        pos_list = []
        pos_node = self.last_node
        while pos_node.father != None:
            pos_list.append(pos_node.pos)
            pos_node = pos_node.father
        #路径输出到txt文件
        with open('IDAstar_result.txt', 'w', encoding='utf-8') as file:
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
    idastar = IDAstar(maze, start, end)
    idastar.search()
    end_time = time.time()
    print('IDA* time:', end_time-start_time)
