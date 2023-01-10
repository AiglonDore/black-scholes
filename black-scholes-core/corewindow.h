#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_corewindow.h"

QT_BEGIN_NAMESPACE
namespace Ui { class CoreWindowClass; };
QT_END_NAMESPACE

class CoreWindow : public QMainWindow
{
    Q_OBJECT

public:
    CoreWindow(QWidget *parent = nullptr);
    ~CoreWindow();

private:
    Ui::CoreWindowClass *ui;
};
