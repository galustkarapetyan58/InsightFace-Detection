#include "facesystem.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <deque>
#include <numeric>

// --- HIGH PERFORMANCE SETTINGS ---
static int frame_counter = 0;
static std::vector<FaceResult> cached_faces; // Stores the result to reuse
const int DETECT_INTERVAL = 3;  // Run detection every 3 frames (boosts FPS)
const int AGE_INTERVAL = 30;    // Run age check every 30 frames (huge FPS boost)

// --- STABILIZER ---
static std::deque<int> age_history;
static std::deque<int> gender_history;
const int HISTORY_SIZE = 15;

static float FACE_REF_5PTS[5][2] = {
    {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
    {41.5493f, 92.3655f}, {70.7299f, 92.2041f}
};

float get_iou(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    float inter = (float)(w * h);
    float area1 = (float)(box1.width * box1.height);
    float area2 = (float)(box2.width * box2.height);
    return inter / (area1 + area2 - inter);
}

void nms(std::vector<FaceResult>& input, std::vector<FaceResult>& output, float iou_threshold) {
    auto it = std::remove_if(input.begin(), input.end(), [](const FaceResult& f) {
        return std::isnan(f.score) || std::isinf(f.score);
    });
    input.erase(it, input.end());

    std::sort(input.begin(), input.end(), [](const FaceResult& a, const FaceResult& b) {
        return a.score > b.score;
    });

    std::vector<bool> merged(input.size(), false);
    for (size_t i = 0; i < input.size(); i++) {
        if (merged[i]) continue;
        output.push_back(input[i]);
        for (size_t j = i + 1; j < input.size(); j++) {
            if (get_iou(input[i].box, input[j].box) > iou_threshold) {
                merged[j] = true;
            }
        }
    }
}

FaceSystem::FaceSystem() : env(ORT_LOGGING_LEVEL_WARNING, "FaceSystem") {}

FaceSystem::~FaceSystem() {
    if (sessDet) delete sessDet;
    if (sessAge) delete sessAge;
}

bool FaceSystem::loadModels(const std::string& detPath, const std::string& agePath) {
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2); // Use 2 threads for speed
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // Max speed

        sessDet = new Ort::Session(env, std::wstring(detPath.begin(), detPath.end()).c_str(), opts);
        sessAge = new Ort::Session(env, std::wstring(agePath.begin(), agePath.end()).c_str(), opts);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR Loading Models: " << e.what() << std::endl;
        return false;
    }
}

void process_stride(int stride, const Ort::Value& tScore, const Ort::Value& tBox, const Ort::Value& tKps,
                    int expected_rows, float rX, float rY,
                    std::vector<FaceResult>& proposals, float& max_score_seen) {

    auto infoScore = tScore.GetTensorTypeAndShapeInfo();
    size_t countScore = infoScore.GetElementCount();
    if (countScore != (size_t)expected_rows) return;

    const float* scores = tScore.GetTensorData<float>();
    const float* boxes  = tBox.GetTensorData<float>();
    const float* kpss   = tKps.GetTensorData<float>();

    int num_anchors = 2;
    int feat_w = 640 / stride;

    for (int i = 0; i < expected_rows; ++i) {
        float score = scores[i];
        if (std::isnan(score) || std::isinf(score)) continue;
        if (score > max_score_seen) max_score_seen = score;

        if (score > 0.40f) {
            int pixel_idx = i / num_anchors;
            int y = pixel_idx / feat_w;
            int x = pixel_idx % feat_w;

            float anchor_x = x * stride;
            float anchor_y = y * stride;

            float l = boxes[i * 4 + 0] * stride;
            float t = boxes[i * 4 + 1] * stride;
            float r = boxes[i * 4 + 2] * stride;
            float b = boxes[i * 4 + 3] * stride;

            FaceResult face;
            face.score = score;
            face.box.x = (int)((anchor_x - l) * rX);
            face.box.y = (int)((anchor_y - t) * rY);
            face.box.width = (int)((l + r) * rX);
            face.box.height = (int)((t + b) * rY);

            for (int k = 0; k < 5; k++) {
                float kx = kpss[i * 10 + (k * 2)] * stride;
                float ky = kpss[i * 10 + (k * 2 + 1)] * stride;
                face.kps.push_back(cv::Point2f((anchor_x + kx) * rX, (anchor_y + ky) * rY));
            }
            proposals.push_back(face);
        }
    }
}

std::vector<FaceResult> run_inference_pass(Ort::Session* sess, const cv::Mat& img,
                                           double scalefactor, cv::Scalar mean, bool swapRB,
                                           const std::vector<const char*>& inputNames,
                                           const std::vector<const char*>& outputNames,
                                           float& max_score_out) {
    try {
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, scalefactor, cv::Size(640, 640), mean, swapRB, false);

        std::vector<int64_t> inputShape = {1, 3, 640, 640};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputOrt = Ort::Value::CreateTensor<float>(memoryInfo, (float*)blob.data, 640*640*3, inputShape.data(), inputShape.size());

        auto outputTensors = sess->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputOrt, 1, outputNames.data(), outputNames.size());
        if (outputTensors.size() < 9) return {};

        std::vector<FaceResult> proposals;
        float rX = (float)img.cols / 640;
        float rY = (float)img.rows / 640;

        process_stride(8, outputTensors[0], outputTensors[3], outputTensors[6], 12800, rX, rY, proposals, max_score_out);
        process_stride(16, outputTensors[1], outputTensors[4], outputTensors[7], 3200, rX, rY, proposals, max_score_out);
        process_stride(32, outputTensors[2], outputTensors[5], outputTensors[8], 800, rX, rY, proposals, max_score_out);

        return proposals;
    } catch (const std::exception& e) { return {}; }
}

std::vector<FaceResult> FaceSystem::detectAndEstimate(const cv::Mat& img) {
    if (img.empty()) return {};

    frame_counter++;

    // 1. SKIP DETECTION if not needed (Reuse last known faces)
    if (frame_counter % DETECT_INTERVAL != 0 && !cached_faces.empty()) {
        return { cached_faces[0] }; // Return cached result immediately
    }

    try {
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputNamePtr = sessDet->GetInputNameAllocated(0, allocator);
        std::string inputNameStr = inputNamePtr.get();
        const char* inputNames[] = { inputNameStr.c_str() };

        size_t numOutputs = sessDet->GetOutputCount();
        std::vector<std::string> outputNameStrings;
        outputNameStrings.reserve(numOutputs);
        for(size_t i = 0; i < numOutputs; i++) {
            auto namePtr = sessDet->GetOutputNameAllocated(i, allocator);
            outputNameStrings.push_back(namePtr.get());
        }
        std::vector<const char*> outputNames;
        outputNames.reserve(numOutputs);
        for(const auto& s : outputNameStrings) outputNames.push_back(s.c_str());

        // Run Detection
        float max_score = 0.0f;
        std::vector<FaceResult> faces = run_inference_pass(sessDet, img, 1.0/128.0, cv::Scalar(127.5, 127.5, 127.5), true, {inputNames[0]}, outputNames, max_score);

        std::vector<FaceResult> nms_faces;
        nms(faces, nms_faces, 0.4f);

        if (nms_faces.empty()) {
            cached_faces.clear();
            return {};
        }

        std::sort(nms_faces.begin(), nms_faces.end(), [](const FaceResult& a, const FaceResult& b) {
            return (a.box.width * a.box.height) > (b.box.width * b.box.height);
        });

        FaceResult& mainFace = nms_faces[0];

        // 2. SKIP AGE CHECK (Reuse old age if not time yet)
        bool time_to_update_age = (frame_counter % AGE_INTERVAL == 0);

        if (time_to_update_age || cached_faces.empty()) {
            runAgeGender(img, nms_faces);
        } else {
            // Copy old Age/Gender to new detection box
            mainFace.age = cached_faces[0].age;
            mainFace.gender = cached_faces[0].gender;
        }

        // Save result for the next frames
        cached_faces = { mainFace };
        return { mainFace };

    } catch (...) { return {}; }
}

cv::Mat FaceSystem::alignFace(const cv::Mat& img, const std::vector<cv::Point2f>& kps) {
    if (kps.size() < 5) return cv::Mat();
    std::vector<cv::Point2f> dstPts;
    for (int i = 0; i < 5; ++i) dstPts.push_back(cv::Point2f(FACE_REF_5PTS[i][0], FACE_REF_5PTS[i][1]));
    cv::Mat M = cv::estimateAffinePartial2D(kps, dstPts);
    if (M.empty()) return cv::Mat();
    cv::Mat aligned;
    cv::warpAffine(img, aligned, M, cv::Size(112, 112));
    return aligned;
}

void FaceSystem::runAgeGender(const cv::Mat& img, std::vector<FaceResult>& faces) {
    if (!sessAge || faces.empty()) return;

    try {
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputNamePtr = sessAge->GetInputNameAllocated(0, allocator);
        std::string inputNameStr = inputNamePtr.get();
        const char* inputNames[] = { inputNameStr.c_str() };
        auto outputNamePtr = sessAge->GetOutputNameAllocated(0, allocator);
        std::string outputNameStr = outputNamePtr.get();
        const char* outputNames[] = { outputNameStr.c_str() };

        FaceResult& face = faces[0];
        cv::Mat aligned = alignFace(img, face.kps);
        if (aligned.empty()) return;

        cv::Mat blob;
        cv::dnn::blobFromImage(aligned, blob, 1.0, cv::Size(96, 96), cv::Scalar(0, 0, 0), true, false);

        std::vector<int64_t> inputShape = {1, 3, 96, 96};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputOrt = Ort::Value::CreateTensor<float>(memoryInfo, (float*)blob.data, 96*96*3, inputShape.data(), inputShape.size());

        auto outputTensors = sessAge->Run(Ort::RunOptions{nullptr}, inputNames, &inputOrt, 1, outputNames, 1);
        float* outputData = outputTensors[0].GetTensorMutableData<float>();

        int cur_gender = (outputData[1] > outputData[0]) ? 1 : 0;
        int cur_age = static_cast<int>(outputData[2] * 100);

        // Stabilizer
        age_history.push_back(cur_age);
        gender_history.push_back(cur_gender);
        if (age_history.size() > HISTORY_SIZE) age_history.pop_front();
        if (gender_history.size() > HISTORY_SIZE) gender_history.pop_front();

        long sum = 0; for(int a : age_history) sum+=a;
        face.age = sum / age_history.size();

        int male_votes = 0; for(int g : gender_history) male_votes+=g;
        face.gender = (male_votes > (int)gender_history.size()/2) ? 1 : 0;
    } catch (...) {
        faces[0].age = -1;
    }
}

std::vector<FaceResult> FaceSystem::runSCRFD(const cv::Mat&) { return {}; }
