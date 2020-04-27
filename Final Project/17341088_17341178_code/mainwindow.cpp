#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    isChooseFirstPlayer = false;
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{
    if(isChooseFirstPlayer)
    {
        gameui = new GameWindow(this, firstPlayer);
        gameui->show();
    }
}

void MainWindow::on_radioButton_clicked()
{
    isChooseFirstPlayer = true;
    firstPlayer = 1;
}

void MainWindow::on_radioButton_2_clicked()
{
    isChooseFirstPlayer = true;
    firstPlayer = 2;
}
