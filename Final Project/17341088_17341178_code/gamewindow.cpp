#include "gamewindow.h"
#include "ui_gamewindow.h"
#include <iostream>
GameWindow::GameWindow(QWidget *parent, int firstPlayer) :
    QMainWindow(parent),
    ui(new Ui::GameWindow)
{
    ui->setupUi(this);
    game = new BlackWhiteChess(firstPlayer);
}

GameWindow::~GameWindow()
{
    delete ui;
}

void GameWindow::init()
{
    currentX = currentY = 7;
    this->setFixedSize(400, 400);
    this->setAutoFillBackground(true);
    QPalette palette;
    palette.setColor(QPalette::Background, QColor("#B1723C"));
    this->setPalette(palette);
    mouseflag = false;
    centralWidget()->setMouseTracking(true);
    setMouseTracking(true);
}
void GameWindow::paintEvent(QPaintEvent *event)
{
    GameState gs = game->getGameState();
    if (gs != PLAYING && gs != END)
    {
        dialog = new Dialog;
        if(gs == BLACKWIN)
        {
            dialog->setMyText("黑棋胜利");
        }
        else if(gs == WHITEWIN)
        {
            dialog->setMyText("白棋胜利");
        }
        else if(gs == DOGFALL)
        {
            dialog->setMyText("平局");
        }
        game->setGameEnd();

        dialog->setModal((false ));
        dialog->setWindowModality((Qt::ApplicationModal));
        dialog->show();
    }
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QPen pen = painter.pen();
    pen.setColor(QColor("#8D5882"));
    pen.setWidth(7);
    painter.setPen(pen);

    QBrush brush;
    brush.setColor(QColor("#EEC085"));
    brush.setStyle(Qt::SolidPattern);
    painter.setBrush(brush);
    painter.drawRect(20, 40, 370, 370);

    pen.setColor(Qt::black);
    pen.setWidth(1);
    painter.setPen(pen);
    for(int i = 0; i < 9; i++)
    {
        painter.drawLine(40+i*40, 60, 40+i*40, 380);
        painter.drawLine(40, 60+i*40, 360, 60+i*40);
    }

    for(int i = 0; i < 8; i++)
    {
        for(int j = 0; j < 8; j++)
        {
            if(game->chessBoard[i][j] == 1)
            {
                brush.setColor(Qt::black);
                painter.setBrush(brush);
                painter.drawEllipse(QPoint((j+1)*40+20, (i+1)*40+20+20), 18, 18);
            }
            else if (game->chessBoard[i][j] == 2)
            {
                brush.setColor(Qt::white);
                painter.setBrush(brush);
                painter.drawEllipse(QPoint((j+1)*40+20, (i+1)*40+20+20), 18, 18);
            }
        }
    }
}
void GameWindow::mouseReleaseEvent(QMouseEvent *event)
{
    currentX = (event->y() - 60) / 40;
    currentY = (event->x() - 40) / 40;
    if (game->chessBoard[currentX][currentY] != 0)
        return;

    std::cout << currentX << " " << currentY << std::endl;
    if (game->getGameState() != END)
    {
        int currentPlayer = game->getPlayer();
        if (currentPlayer == game->person)
        {
            game->play(currentX, currentY);
            repaint();
        }
        while(true)
        {
            if (game->getPlayer() == game->bot)
            {
                game->botPlay();
                repaint();
            }
            else
                break;
        }
    }
}
