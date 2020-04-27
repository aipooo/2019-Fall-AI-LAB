#include "dialog.h"
#include "ui_dialog.h"

Dialog::Dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog)
{
    ui->setupUi(this);
}

Dialog::~Dialog()
{
    delete ui;
}

void Dialog::setMyText(const QString &text)
{
    ui->plainTextEdit->document()->clear();
    ui->plainTextEdit->appendPlainText(text);
}
