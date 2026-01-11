#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

// Holds the final result for one face
struct FaceResult {
    cv::Rect box;                 // Bounding box
    float score;                  // Detection confidence
    std::vector<cv::Point2f> kps; // 5 landmarks

    int age = -1;                 // Estimated Age
    int gender = -1;              // 0 = Female, 1 = Male
};

class FaceSystem {
public:
    FaceSystem();
    ~FaceSystem();

    // Load models
    bool loadModels(const std::string& detPath, const std::string& agePath);

    // Main function: Detect -> Align -> Estimate Age
    std::vector<FaceResult> detectAndEstimate(const cv::Mat& img);

private:
    Ort::Env env;
    Ort::Session* sessDet = nullptr;
    Ort::Session* sessAge = nullptr;

    // Internal Helpers
    std::vector<FaceResult> runSCRFD(const cv::Mat& img);
    void runAgeGender(const cv::Mat& img, std::vector<FaceResult>& faces);
    cv::Mat alignFace(const cv::Mat& img, const std::vector<cv::Point2f>& kps);
};
