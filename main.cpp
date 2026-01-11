#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    // Fix scaling for High DPI screens (like 4K monitors)
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QApplication app(argc, argv);

    MainWindow w;
    w.show();

    return app.exec();
}
