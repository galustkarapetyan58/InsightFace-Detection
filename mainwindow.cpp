#include "mainwindow.h"
#include <QVBoxLayout>
#include <QDebug>
#include <QImage>
#include <QPixmap>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    // --- 1. SETUP UI MANUALLY ---
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QVBoxLayout *layout = new QVBoxLayout(centralWidget);
    videoLabel = new QLabel(this);
    videoLabel->setAlignment(Qt::AlignCenter);
    videoLabel->setStyleSheet("background-color: black;"); // Make it look nice
    layout->addWidget(videoLabel);

    resize(800, 600); // Set default window size

    // --- 2. LOAD AI MODELS ---
    // Ensure these files are in your "build" folder (where the .exe is)
    if (!faceSystem.loadModels("det_10g.onnx", "genderage.onnx", "w600k_r50.onnx")) {
        qDebug() << "CRITICAL ERROR: Failed to load ONNX models!";
    }

    // --- 3. START CAMERA ---
    cap.open(0); // Open default webcam
    if (!cap.isOpened()) {
        qDebug() << "Error: Could not open webcam.";
        videoLabel->setText("Error: No Webcam Found");
    } else {
        // --- 4. START LOOP ---
        fpsTimer.start();
        timer = new QTimer(this);
        connect(timer, &QTimer::timeout, this, &MainWindow::processFrame);
        timer->start(30); // ~30 FPS
    }
}

MainWindow::~MainWindow() {
    cap.release();
}

void MainWindow::processFrame() {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) return;

    // --- A. RUN FACE SYSTEM ---
    auto faces = faceSystem.detectAndEstimate(frame);

    // --- B. DRAW RESULTS ---
    for (const auto& face : faces) {
        // Green Box
        cv::rectangle(frame, face.box, cv::Scalar(0, 255, 0), 2);

        // Text Label
        std::string label = (face.name != "Unknown" ? face.name + ", " : "") +
                            (face.gender == 1 ? "Male" : "Female") +
                            ", " + std::to_string(face.age);

        // Black Background for Text
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
        cv::rectangle(frame, cv::Point(face.box.x, face.box.y - labelSize.height - 10),
                      cv::Point(face.box.x + labelSize.width, face.box.y),
                      cv::Scalar(0, 0, 0), cv::FILLED);

        // White Text
        cv::putText(frame, label, cv::Point(face.box.x, face.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }

    // --- C. FPS COUNTER ---
    fpsCounter++;
    if (fpsTimer.elapsed() >= 500) {
        currentFPS = fpsCounter / (fpsTimer.elapsed() / 1000.0f);
        fpsCounter = 0;
        fpsTimer.restart();
    }
    cv::putText(frame, "FPS: " + std::to_string((int)currentFPS), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

    // --- D. DISPLAY ON QLABEL ---
    // OpenCV (BGR) -> Qt (RGB)
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    QImage qimg(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
    videoLabel->setPixmap(QPixmap::fromImage(qimg));

    // Resize label to fit window if needed
    videoLabel->setScaledContents(true);
}
