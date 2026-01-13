#include "mainwindow.h"
#include <QVBoxLayout>
#include <QDebug>
#include <QImage>
#include <QPixmap>
#include <QInputDialog> // Required for asking names
#include <QMessageBox>  // Required for 'Clear' popup
#include <QKeyEvent>    // Required for keyboard input

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    // 1. SETUP UI
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    videoLabel = new QLabel(this);
    videoLabel->setAlignment(Qt::AlignCenter);
    videoLabel->setStyleSheet("background-color: black;");
    layout->addWidget(videoLabel);

    resize(800, 600);

    // --- CRITICAL FIX FOR KEYS ---
    // This ensures the window catches 'R' and 'C' presses
    this->setFocusPolicy(Qt::StrongFocus);

    // 2. LOAD MODELS
    // Ensure these files are in your build folder
    if (!faceSystem.loadModels("det_10g.onnx", "genderage.onnx", "w600k_r50.onnx")) {
        videoLabel->setText("Error: Failed to load models.");
    }

    // Load existing faces
    faceSystem.loadDatabase("faces.db");

    // 3. SETUP THREAD WATCHER
    connect(&watcher, &QFutureWatcher<std::vector<FaceResult>>::finished,
            this, &MainWindow::onAIResultsReady);

    // 4. START CAMERA
    cap.open(0);
    if (cap.isOpened()) {
        fpsTimer.start();
        timer = new QTimer(this);
        connect(timer, &QTimer::timeout, this, &MainWindow::processFrame);
        timer->start(30);
    }
}

MainWindow::~MainWindow() {
    cap.release();
    if (watcher.isRunning()) {
        watcher.waitForFinished();
    }
}

// --- KEYBOARD CONTROLS ---
void MainWindow::keyPressEvent(QKeyEvent *event) {
    // REGISTER FACE (Press R)
    if (event->key() == Qt::Key_R) {
        if (currentFaces.empty()) {
            QMessageBox::warning(this, "Error", "No face detected to register!");
            return;
        }

        // Pause AI so the face doesn't move while typing
        bool wasBusy = isAIBusy;
        isAIBusy = true;

        // Loop through all visible faces
        int count = 0;
        for (const auto& face : currentFaces) {
            count++;

            // Determine Location
            std::string location = "Center";
            int centerX = face.box.x + (face.box.width / 2);
            if (centerX < 213) location = "Left";
            else if (centerX > 426) location = "Right";

            std::string currentName = face.name;

            // Ask User
            bool ok;
            QString label = QString("Face at %1 (Current: %2)\nEnter New Name:")
                                .arg(QString::fromStdString(location))
                                .arg(QString::fromStdString(currentName));

            // Pre-fill with current name to allow easy editing
            QString defaultText = (currentName == "Unknown" || currentName.rfind("ID:", 0) == 0) ? "" : QString::fromStdString(currentName);

            QString newName = QInputDialog::getText(this, "Register Face", label,
                                                    QLineEdit::Normal, defaultText, &ok);

            if (ok && !newName.isEmpty()) {
                // Register (Deleting old name if it existed)
                faceSystem.registerFace(newName.toStdString(), currentName, face.embedding);
                faceSystem.saveDatabase("faces.db");
            }
        }

        // Resume AI
        if (!wasBusy) isAIBusy = false;
    }
    // CLEAR DATABASE (Press C)
    else if (event->key() == Qt::Key_C) {
        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(this, "Clear Database",
                                      "Delete ALL saved faces? This cannot be undone.",
                                      QMessageBox::Yes|QMessageBox::No);
        if (reply == QMessageBox::Yes) {
            faceSystem.clearDatabase();
            QMessageBox::information(this, "Success", "Database wiped.");
        }
    }
}

// --- MAIN LOOP ---
// --- mainwindow.cpp ---

void MainWindow::processFrame() {
    cv::Mat rawFrame;
    cap >> rawFrame;
    if (rawFrame.empty()) return;

    cv::Mat displayFrame;
    cv::resize(rawFrame, displayFrame, cv::Size(640, 480));

    // BACKGROUND AI
    if (!isAIBusy) {
        isAIBusy = true;
        cv::Mat frameForAI = rawFrame.clone();
        QFuture<std::vector<FaceResult>> future = QtConcurrent::run([this, frameForAI]() {
            return faceSystem.detectAndEstimate(frameForAI);
        });
        watcher.setFuture(future);
    }

    // DRAW RESULTS
    float scaleX = (float)displayFrame.cols / rawFrame.cols;
    float scaleY = (float)displayFrame.rows / rawFrame.rows;

    for (const auto& face : currentFaces) {
        cv::Rect scaledBox;
        scaledBox.x = (int)(face.box.x * scaleX);
        scaledBox.y = (int)(face.box.y * scaleY);
        scaledBox.width = (int)(face.box.width * scaleX);
        scaledBox.height = (int)(face.box.height * scaleY);

        // --- FIXED: ALWAYS GREEN ---
        cv::Scalar color = cv::Scalar(0, 255, 0); // Green (B, G, R)

        cv::rectangle(displayFrame, scaledBox, color, 2);

        // Create Label: "Name, Gender, Age"
        std::string genderText = (face.gender == 1) ? "Male" : (face.gender == 0 ? "Female" : "?");

        // Don't show "ID:X" or "Unknown" if we have a real name
        std::string nameDisplay = face.name;

        std::string label = nameDisplay + ", " + genderText + ", " + std::to_string(face.age);

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
        int textY = std::max(scaledBox.y - 10, labelSize.height + 5);

        // Black background for text
        cv::rectangle(displayFrame, cv::Point(scaledBox.x, textY - labelSize.height - 5),
                      cv::Point(scaledBox.x + labelSize.width, textY + 5),
                      cv::Scalar(0, 0, 0), cv::FILLED);

        // White text
        cv::putText(displayFrame, label, cv::Point(scaledBox.x, textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }

    // FPS
    fpsCounter++;
    if (fpsTimer.elapsed() >= 1000) {
        currentFPS = fpsCounter / (fpsTimer.elapsed() / 1000.0f);
        fpsCounter = 0;
        fpsTimer.restart();
    }
    cv::putText(displayFrame, "FPS: " + std::to_string((int)currentFPS), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

    cv::cvtColor(displayFrame, displayFrame, cv::COLOR_BGR2RGB);
    QImage qimg(displayFrame.data, displayFrame.cols, displayFrame.rows, displayFrame.step, QImage::Format_RGB888);
    videoLabel->setPixmap(QPixmap::fromImage(qimg));
    videoLabel->setScaledContents(true);
}
void MainWindow::onAIResultsReady() {
    currentFaces = watcher.result();
    isAIBusy = false;
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event) {
    if (event->type() == QEvent::KeyPress) {
        QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);

        // --- REGISTER FACE ('R' Key) ---
        if (keyEvent->key() == Qt::Key_R) {
            if (currentFaces.empty()) {
                QMessageBox::warning(this, "Error", "No face detected!");
                return true;
            }

            bool wasBusy = isAIBusy;
            isAIBusy = true; // Pause AI

            // We iterate using a reference (&) so we can modify the face directly
            for (auto& face : currentFaces) {

                std::string currentName = face.name;
                QString label = QString("Enter New Name for '%1':").arg(QString::fromStdString(currentName));

                // Don't show "ID:0" or "Unknown" in the text box, keep it empty for easy typing
                QString defaultText = (currentName.find("ID:") != std::string::npos || currentName == "Unknown")
                                          ? "" : QString::fromStdString(currentName);

                bool ok;
                QString newName = QInputDialog::getText(this, "Register Face", label,
                                                        QLineEdit::Normal, defaultText, &ok);

                if (ok && !newName.isEmpty()) {
                    // 1. Update Database
                    faceSystem.registerFace(newName.toStdString(), currentName, face.embedding);
                    faceSystem.saveDatabase("faces.db");

                    // 2. FORCE UPDATE SCREEN IMMEDIATELY
                    // This makes the name appear instantly next to the box
                    face.name = newName.toStdString();
                }
            }

            if (!wasBusy) isAIBusy = false; // Resume AI
            return true;
        }

        // --- CLEAR DATABASE ('C' Key) ---
        else if (keyEvent->key() == Qt::Key_C) {
            QMessageBox::StandardButton reply;
            reply = QMessageBox::question(this, "Clear Database",
                                          "Delete ALL faces?",
                                          QMessageBox::Yes|QMessageBox::No);
            if (reply == QMessageBox::Yes) {
                faceSystem.clearDatabase();
            }
            return true;
        }
    }
    return QMainWindow::eventFilter(obj, event);
}
