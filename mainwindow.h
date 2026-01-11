#pragma once

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include "facesystem.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onStartCamera(); // Starts/Stops the webcam
    void updateFrame();   // Called repeatedly by QTimer

private:
    QWidget *centralWidget;
    QVBoxLayout *layout;
    QLabel *imageLabel;
    QPushButton *btnStart;

    // Camera Resources
    cv::VideoCapture cap;
    QTimer *timer;

    // Face Engine
    FaceSystem faceSystem;
};
