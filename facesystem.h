#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <map>
#include <deque>
#include <fstream>


// Structure to hold one face's complete data
struct FaceResult {
    cv::Rect box;
    float score;
    std::vector<cv::Point2f> kps; // 5 Facial Landmarks
    int age = 0;
    int gender = 0;               // 0=Female, 1=Male
    std::string name = "Unknown"; // Recognized Name
    std::vector<float> embedding; // Face Fingerprint (512 numbers)
};

class FaceSystem {
public:
    FaceSystem();
    ~FaceSystem();

    // Load all 3 AI models: Detection, Age/Gender, Recognition
    bool loadModels(const std::string& detPath, const std::string& agePath, const std::string& recPath);

    // The main function: Takes an image, returns tracked and analyzed faces
    std::vector<FaceResult> detectAndEstimate(const cv::Mat& img);

    // Register a known face (save their fingerprint)
   void registerFace(const std::string& newName, const std::string& oldName, const std::vector<float>& embedding);

    // --- NEW: SAVE & LOAD ---
    void saveDatabase(const std::string& filename);
    void loadDatabase(const std::string& filename);
    void clearDatabase();
private:
    Ort::Env env;
    Ort::Session* sessDet = nullptr; // Detection Model
    Ort::Session* sessAge = nullptr; // Age/Gender Model
    Ort::Session* sessRec = nullptr; // Recognition Model

    // Internal Helper Functions
    void runAgeGender(const cv::Mat& img, std::vector<FaceResult>& faces);
    void runRecognition(const cv::Mat& img, std::vector<FaceResult>& faces);
    cv::Mat alignFace(const cv::Mat& img, const std::vector<cv::Point2f>& kps);
    cv::Mat alignFaceZoomed(const cv::Mat& img, const std::vector<cv::Point2f>& kps);
    // Database of known people
    std::map<std::string, std::vector<float>> known_faces;
};
