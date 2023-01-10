#include "stdafx.h"
#include "corewindow.h"

CoreWindow::CoreWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CoreWindowClass())
{
    ui->setupUi(this);
}

CoreWindow::~CoreWindow()
{
    delete ui;
}
