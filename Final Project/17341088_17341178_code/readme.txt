【文件用途】
-ga.h：黑白棋实现部分，包括遗传算法产生权重

-node.h：棋盘状态节点的定义

-blackwhitechess.h：UI界面的逻辑判断，设置接口调用ga.h 
 blackwhitechess.cpp

-gamewindow.h：UI界面实现，与用户交互
 gamewindow.cpp
 mainwindow.h
 mainwindow.cpp
 dialog.h

-main.cpp：主程序

-mainwindow.ui：UI界面的具体设计
 gamewindow.ui
 dialog.ui

【工程如何运行】
使用QT打开BlackWhiteChess.pro，点击运行（注意路径不能有中文名称）