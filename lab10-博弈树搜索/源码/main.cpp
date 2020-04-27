#include<iostream>
#include "config.h"
#include "grid.h"
#include "chessboard.h"
#include "game.h"

using namespace std;

int main()
{
    Game G;
    G.startGame(4);
    cout<<"请选择游戏模式："<<endl;
	cout<<"1.先手    2.后手"<<endl;
	int option;
	cin>>option;
	while(option!=1 && option!=2){
		cout<<"输入不合法，请重新输入！"<<endl;
		cin>>option; 
	} 
	system("cls");
	G.show();
	int x, y;
	if(option==1){
		while(1){
			cout<<"请输入落子位置：";
			cin>>x>>y;
			while(1){
				if(G.placePiece(x, y))
					break;
				cout<<"输入的落子位置不合法，请重新输入：";
				cin>>x>>y;
			}
			G.show();
			cout<<"当前局面得分："<<endl;
			cout<<"AI  : "<<G.evaluateState(G.curState, 2)<<endl;
			cout<<"USER: "<<G.evaluateState(G.curState, 1)<<endl<<endl<<endl;
			if(G.isStart==false){
				cout<<"你赢了！"<<endl;
				break;
			}
			G.placePieceAI();
	        G.show();
			cout<<"当前局面得分："<<endl;
			cout<<"AI  : "<<G.evaluateState(G.curState, 2)<<endl;
			cout<<"USER: "<<G.evaluateState(G.curState, 1)<<endl<<endl<<endl;	
			if(G.isStart==false){
				cout<<"你输了！"<<endl;
				break;
			}	
		}
	}
	else if(option==2){
		while(1){
			G.placePieceAI();
	        G.show();
	        cout<<"当前局面得分："<<endl;
			cout<<"AI  : "<<G.evaluateState(G.curState, 1)<<endl;
			cout<<"USER: "<<G.evaluateState(G.curState, 2)<<endl<<endl<<endl;
	        if(G.isStart==false){
				cout<<"你输了！"<<endl;
				break;
			}
			cout<<"请输入落子位置："; 
			cin>>x>>y;
			while(1){
				if(G.placePiece(x, y))
					break;
				cout<<"输入的落子位置不合法，请重新输入：";
				cin>>x>>y;
			}
			G.show();
	        cout<<"当前局面得分："<<endl;
			cout<<"AI  : "<<G.evaluateState(G.curState, 1)<<endl;
			cout<<"USER: "<<G.evaluateState(G.curState, 2)<<endl<<endl<<endl;
			if(G.isStart==false){
				cout<<"你赢了！"<<endl;
				break;
			}			
		}	
	}
	getchar(); 
}
