// --- facesystem.h ---
#ifndef FACESYSTEM_H
#define FACESYSTEM_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <onnxruntime_cxx_api.h>

struct FaceResult {
    cv::Rect box;
    float score;
    std::vector<cv::Point2f> kps;

    // AI Results
    int age = 0;
    int gender = -1; // 0=Female, 1=Male
    std::string name = "Unknown";
    std::vector<float> embedding;

    // NEW: Unique ID for perfect registration
    int id = -1;
};

class FaceSystem {
public:
    FaceSystem();
    ~FaceSystem();

    bool loadModels(const std::string& detPath, const std::string& agePath, const std::string& recPath);
    std::vector<FaceResult> detectAndEstimate(const cv::Mat& img);

    // NEW: We added 'faceID' to this function
    void registerFace(const std::string& newName, const std::string& oldName, const std::vector<float>& embedding, int faceID);

    void saveDatabase(const std::string& filename);
    void loadDatabase(const std::string& filename);
    void clearDatabase();

private:
    Ort::Env env;
    Ort::Session* sessDet = nullptr;
    Ort::Session* sessAge = nullptr;
    Ort::Session* sessRec = nullptr;

    std::map<std::string, std::vector<float>> known_faces;

    // Helper functions
    cv::Mat alignFace(const cv::Mat& img, const std::vector<cv::Point2f>& kps);
    cv::Mat alignFaceZoomed(const cv::Mat& img, const std::vector<cv::Point2f>& kps);
    void runRecognition(const cv::Mat& img, std::vector<FaceResult>& faces);
    void runAgeGender(const cv::Mat& img, std::vector<FaceResult>& faces);
};

#endif // FACESYSTEM_H
