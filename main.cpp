#include "facesystem.h"
#include <QCoreApplication>
#include <QDebug>
#include <iostream>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv); // Required for Qt event loop

    FaceSystem fs;

    // Define the NEW "Hard Way" model filenames
    std::string detPath = "scrfd_10g_bnkps.onnx";
    std::string recPath = "w600k_r50.onnx";
    std::string gaPath  = "genderage.onnx";

    std::cout << "--- Professional Face AI (ONNX Runtime) ---" << std::endl;
    std::cout << "Loading models..." << std::endl;

    // Try to load the models
    if (!fs.loadModels(detPath, recPath, gaPath)) {
        qCritical() << "Error: Could not load AI models!";
        qCritical() << "Please make sure these 3 files are in your build folder:";
        qCritical() << "1. " << detPath.c_str();
        qCritical() << "2. " << recPath.c_str();
        qCritical() << "3. " << gaPath.c_str();
        return -1;
    }

    std::cout << "Models loaded successfully!" << std::endl;
    std::cout << "Starting Camera... (Press ESC to quit)" << std::endl;

    fs.runWebcam();

    return 0; // The runWebcam loop handles the exit now
}
