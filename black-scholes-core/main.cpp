#include "stdafx.h"
#include "corewindow.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    CoreWindow w;
    w.show();
    return a.exec();
}
