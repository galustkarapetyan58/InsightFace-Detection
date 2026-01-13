#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>
#include <QtConcurrent/QtConcurrent> // <--- NEW: For background processing
#include <QFuture>
#include <QFutureWatcher>

#include "facesystem.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void processFrame();
    void onAIResultsReady(); // <--- NEW: Called when AI finishes

private:
    QLabel *videoLabel;
    QTimer *timer;
    cv::VideoCapture cap;
    FaceSystem faceSystem;

    QElapsedTimer fpsTimer;
    int fpsCounter = 0;
    float currentFPS = 0.0f;
    void keyPressEvent(QKeyEvent *event) override;
    //bool eventFilter(QObject *obj, QEvent *event) override;
    // --- THREADING VARIABLES ---
    bool isAIBusy = false;                // Is the AI currently thinking?
    QFutureWatcher<std::vector<FaceResult>> watcher; // Watches the background thread
    std::vector<FaceResult> currentFaces; // Stores the last known faces to draw
};

#endif // MAINWINDOW_H
