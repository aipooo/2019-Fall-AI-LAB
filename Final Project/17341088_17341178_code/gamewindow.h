#ifndef GAMEWINDOW_H
#define GAMEWINDOW_H

#include <QMainWindow>
#include <QtGui>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include "blackwhitechess.h"
#include "dialog.h"
namespace Ui {
class GameWindow;
}

class GameWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit GameWindow(QWidget *parent = nullptr,  int firstPlayer = 1);
    ~GameWindow();
    void init();
protected:
    void paintEvent(QPaintEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
private:
    Ui::GameWindow *ui;
    Dialog *dialog;
    BlackWhiteChess *game;
    int moveX, moveY, currentX, currentY;
    bool mouseflag;
};

#endif // GAMEWINDOW_H
