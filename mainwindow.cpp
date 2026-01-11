#include "mainwindow.h"
#include <QMessageBox>
#include <QImage>
#include <QPixmap>

// Helper: Convert cv::Mat to QImage
QImage cvMatToQImage(const cv::Mat &inMat) {
    switch (inMat.type()) {
    case CV_8UC4: {
        QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_ARGB32);
        return image.copy();
    }
    case CV_8UC3: {
        QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    default: return QImage();
    }
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    layout = new QVBoxLayout(centralWidget);

    btnStart = new QPushButton("Start Webcam", this);
    imageLabel = new QLabel("Click Start to open Camera", this);
    imageLabel->setAlignment(Qt::AlignCenter);

    // Set a good window size
    resize(1000, 800);

    layout->addWidget(btnStart);
    layout->addWidget(imageLabel);

    // Initialize Timer
    timer = new QTimer(this);

    // Connect Signals
    connect(btnStart, &QPushButton::clicked, this, &MainWindow::onStartCamera);
    connect(timer, &QTimer::timeout, this, &MainWindow::updateFrame);

    // Initialize InsightFace Models
    if (!faceSystem.loadModels("det_10g.onnx", "genderage.onnx")) {
        QMessageBox::critical(this, "Error", "Failed to load models! Check files next to .exe");
    }
}

MainWindow::~MainWindow() {
    if (cap.isOpened()) cap.release();
}

void MainWindow::onStartCamera() {
    if (timer->isActive()) {
        timer->stop();
        cap.release();
        btnStart->setText("Start Webcam");
        imageLabel->setText("Camera Stopped");
    } else {
        // Open Default Camera (Index 0)
        cap.open(0);
        if (!cap.isOpened()) {
            QMessageBox::warning(this, "Error", "Could not access the webcam!");
            return;
        }
        timer->start(30); // 30ms ~ 33 FPS
        btnStart->setText("Stop Webcam");
    }
}

void MainWindow::updateFrame() {
    cv::Mat frame;
    cap >> frame; // Capture Frame
    if (frame.empty()) return;

    // Optional: Flip it like a mirror
    cv::flip(frame, frame, 1);

    // 1. Run Detection & Estimation
    auto results = faceSystem.detectAndEstimate(frame);

    // 2. Draw Results
    for (const auto& res : results) {
        // Green Box
        cv::rectangle(frame, res.box, cv::Scalar(0, 255, 0), 2);

        // Landmarks
        for (const auto& pt : res.kps) {
            cv::circle(frame, pt, 2, cv::Scalar(0, 0, 255), -1);
        }

        // Text
        std::string genderStr = (res.gender == 1) ? "Male" : "Female";
        std::string label = genderStr + ", " + std::to_string(res.age);

        // Draw text background for readability
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseLine);
        cv::rectangle(frame,
                      cv::Point(res.box.x, res.box.y - labelSize.height - 5),
                      cv::Point(res.box.x + labelSize.width, res.box.y),
                      cv::Scalar(0, 255, 0), -1);

        cv::putText(frame, label, cv::Point(res.box.x, res.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    }

    // 3. Display
    QImage qImg = cvMatToQImage(frame);
    imageLabel->setPixmap(QPixmap::fromImage(qImg).scaled(
        imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation
        ));
}
