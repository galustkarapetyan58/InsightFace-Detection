#ifndef FACESYSTEM_H
#define FACESYSTEM_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// A structure to hold one face's data
struct FaceObject {
    cv::Rect box;
    std::vector<cv::Point2f> landmarks; // 5 points (Eyes, Nose, Mouth)
    float confidence;
    int age;
    std::string gender;
    std::vector<float> embedding; // The "Identity" vector
};

class FaceSystem {
public:
    FaceSystem();
    bool loadModels(const std::string& detPath, const std::string& recPath, const std::string& gaPath);
    void runWebcam();

private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> det_session;
    std::unique_ptr<Ort::Session> rec_session;
    std::unique_ptr<Ort::Session> ga_session;

    // --- Cache for SCRFD (Fix FPS) ---
    std::vector<std::string> input_names_stg;
    std::vector<const char*> input_names_ptr;
    std::vector<std::string> output_names_stg;
    std::vector<const char*> output_names_ptr;
    std::map<int, std::tuple<int, int, int>> stride_map;
    
    // EMA Smoothing variable
    float smoothed_age = -1.0f;
    float smoothed_gender_diff = 0.0f;

    // --- The Core AI Functions ---
    std::vector<FaceObject> detectSCRFD(const cv::Mat& frame);
    void analyzeFace(const cv::Mat& frame, FaceObject& face); // Gets Age + Recognition

    // Helper to align the face before recognition
    cv::Mat alignFace(const cv::Mat& frame, const std::vector<cv::Point2f>& kps);
};

#endif
