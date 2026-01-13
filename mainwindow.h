#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>
#include "facesystem.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void processFrame(); // Function called every frame

private:
    QLabel *videoLabel;       // The widget that shows the video
    cv::VideoCapture cap;     // Webcam
    QTimer *timer;            // Loop timer
    FaceSystem faceSystem;    // Your AI System

    // FPS Counting
    QElapsedTimer fpsTimer;
    int fpsCounter = 0;
    float currentFPS = 0.0f;
};
