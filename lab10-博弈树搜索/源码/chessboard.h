//∆Â≈Ã

#ifndef CHESSBOARD_H
#define CHESSBOARD_H
#include "config.h" 
#include<iostream>
#include<stdlib.h>
#include<iomanip>
using namespace std; 

class ChessBoard :CONFIG {
public:
    Grid chessBoard[BOARD_SIZE][BOARD_SIZE];

    ChessBoard() {
        for (int i = 0; i < BOARD_SIZE; ++i)
            for (int j = 0; j < BOARD_SIZE; ++j)
                chessBoard[i][j].grid();
        chessBoard[5][5]=1;
        chessBoard[5][6]=1;
        chessBoard[6][5]=2;
        chessBoard[5][4]=2;
    }

    ChessBoard(const ChessBoard &othr) {
        for (int i = 0; i < BOARD_SIZE; ++i)
            for (int j = 0; j < BOARD_SIZE; ++j)
                chessBoard[i][j].grid(othr.chessBoard[i][j].type);
    }

    //∑≈÷√∆Â◊” 
    bool placePiece(int x, int y, int type) {
        if (chessBoard[x][y].isEmpty()) {
            chessBoard[x][y].type = type;
            return true;
        }
        return false;
    }
    
    void print_board(){
    	//system("cls");
        for(int i=0; i<BOARD_SIZE; i++)
        	cout<<"----";
        cout<<"--"<<endl;
        for(int i=0; i<BOARD_SIZE; i++){
        	for(int j=0; j<BOARD_SIZE; j++){
        		if(chessBoard[i][j].type==0)
        			cout<<"| "<<" "<<" ";  
				else if(chessBoard[i][j].type==1)
				    cout<<"| "<<"x"<<" "; 
				else if(chessBoard[i][j].type==2)
				    cout<<"| "<<"o"<<" "; 
			}
			cout<<" |  "<<i<<endl;
			for(int j=0; j<BOARD_SIZE; j++)
				cout<<"----";
			cout<<"--"<<endl;
		}
		for(int i=0; i<BOARD_SIZE; i++){
			cout.setf(ios::right);      
    		cout.fill(' ');             
    		cout.width(3);              
			cout<<i<<" ";			
		}
		cout<<endl<<endl;
	}
};

#endif
