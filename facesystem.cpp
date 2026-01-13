#include "facesystem.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <deque>
#include <numeric>

// --- SETTINGS ---
const float DETECT_THRESHOLD = 0.25f; // Sensitivity
const float REC_THRESHOLD = 0.40f;
const float MATCH_DIST_THRESHOLD = 150.0f;
const int MAX_MISSING_FRAMES = 15;
const int HISTORY_SIZE = 30;
const int ANALYSIS_INTERVAL = 15; // Run analysis every 0.5s

struct TrackedFace {
    int id;
    cv::Rect box;
    std::deque<int> age_hist;
    std::deque<int> gender_hist;
    std::string name;
    std::vector<float> embedding;
    int stable_age = 0;
    int stable_gender = 0;
    int missing_frames = 0;
    int frames_since_analysis = 999;
};

static std::vector<TrackedFace> tracker_db;
static int next_face_id = 0;

static float FACE_REF_5PTS[5][2] = {
    {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
    {41.5493f, 92.3655f}, {70.7299f, 92.2041f}
};

// --- MATH HELPERS ---
float get_iou(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    float inter = (float)(w * h);
    float union_area = (float)((box1.width * box1.height) + (box2.width * box2.height) - inter);
    return (union_area <= 0.0f) ? 0.0f : (inter / union_area);
}

float calculateSimilarity(const std::vector<float>& f1, const std::vector<float>& f2) {
    if (f1.empty() || f2.empty()) return 0.0f;
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (size_t i = 0; i < f1.size(); i++) {
        dot += f1[i] * f2[i];
        normA += f1[i] * f1[i];
        normB += f2[i] * f2[i];
    }
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

void nms(std::vector<FaceResult>& input, std::vector<FaceResult>& output, float iou_threshold) {
    auto it = std::remove_if(input.begin(), input.end(), [](const FaceResult& f) {
        return f.score < DETECT_THRESHOLD;
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
            if (get_iou(input[i].box, input[j].box) > iou_threshold) merged[j] = true;
        }
    }
}

// --- FACE SYSTEM IMPLEMENTATION ---

FaceSystem::FaceSystem() : env(ORT_LOGGING_LEVEL_WARNING, "FaceSystem") {}

FaceSystem::~FaceSystem() {
    if (sessDet) delete sessDet;
    if (sessAge) delete sessAge;
    if (sessRec) delete sessRec;
}

bool FaceSystem::loadModels(const std::string& detPath, const std::string& agePath, const std::string& recPath) {
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        sessDet = new Ort::Session(env, std::wstring(detPath.begin(), detPath.end()).c_str(), opts);
        sessAge = new Ort::Session(env, std::wstring(agePath.begin(), agePath.end()).c_str(), opts);
        sessRec = new Ort::Session(env, std::wstring(recPath.begin(), recPath.end()).c_str(), opts);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR Loading Models: " << e.what() << std::endl;
        return false;
    }
}

void FaceSystem::registerFace(const std::string& name, const std::vector<float>& embedding) {
    if(embedding.size() == 512) known_faces[name] = embedding;
}

void process_stride(int stride, const Ort::Value& tScore, const Ort::Value& tBox, const Ort::Value& tKps,
                    int rows, float rX, float rY, std::vector<FaceResult>& proposals) {
    const float* scores = tScore.GetTensorData<float>();
    const float* boxes  = tBox.GetTensorData<float>();
    const float* kpss   = tKps.GetTensorData<float>();
    int feat_w = 640 / stride;

    for (int i = 0; i < rows; ++i) {
        float score = scores[i];
        if (score > DETECT_THRESHOLD) {
            int grid_idx = i / 2;
            int y = grid_idx / feat_w;
            int x = grid_idx % feat_w;
            float anchor_x = x * stride;
            float anchor_y = y * stride;

            FaceResult face;
            face.score = score;
            float l = boxes[i * 4 + 0] * stride;
            float t = boxes[i * 4 + 1] * stride;
            float r = boxes[i * 4 + 2] * stride;
            float b = boxes[i * 4 + 3] * stride;

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

// --- MAIN LOOP ---
std::vector<FaceResult> FaceSystem::detectAndEstimate(const cv::Mat& img) {
    if (img.empty()) return {};

    try {
        Ort::AllocatorWithDefaultOptions allocator;

        // --- 1. SAFE MEMORY MANAGEMENT FOR NAMES ---
        // Get Input Name
        auto inputNamePtr = sessDet->GetInputNameAllocated(0, allocator);
        const char* inputNames[] = { inputNamePtr.get() };

        // Get Output Names (Keep smart pointers alive in a vector)
        size_t numOutputs = sessDet->GetOutputCount();
        std::vector<Ort::AllocatedStringPtr> outputNamePtrs; // Keeps memory alive
        std::vector<const char*> outputNames;                // Passed to Run()

        for(size_t i=0; i<numOutputs; i++) {
            auto ptr = sessDet->GetOutputNameAllocated(i, allocator);
            outputNames.push_back(ptr.get());
            outputNamePtrs.push_back(std::move(ptr));
        }

        // --- 2. PREPROCESS ---
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0/128.0, cv::Size(640, 640), cv::Scalar(127.5, 127.5, 127.5), true, false);

        std::vector<int64_t> inputShape = {1, 3, 640, 640};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputOrt = Ort::Value::CreateTensor<float>(memoryInfo, (float*)blob.data, 640*640*3, inputShape.data(), inputShape.size());

        // --- 3. RUN ---
        auto outputs = sessDet->Run(Ort::RunOptions{nullptr}, inputNames, &inputOrt, 1, outputNames.data(), outputNames.size());

        // --- 4. PARSE ---
        std::vector<FaceResult> proposals;
        float rX = (float)img.cols / 640;
        float rY = (float)img.rows / 640;
        process_stride(8, outputs[0], outputs[3], outputs[6], 12800, rX, rY, proposals);
        process_stride(16, outputs[1], outputs[4], outputs[7], 3200, rX, rY, proposals);
        process_stride(32, outputs[2], outputs[5], outputs[8], 800, rX, rY, proposals);

        std::vector<FaceResult> nms_faces;
        nms(proposals, nms_faces, 0.45f);

        if (nms_faces.empty()) {
            for (auto& t : tracker_db) t.missing_frames++;
            return {};
        }

        // --- 5. TRACKER ---
        for (auto& t : tracker_db) {
            t.missing_frames++;
            t.frames_since_analysis++;
        }

        std::vector<FaceResult> final_results;

        for (const auto& detected : nms_faces) {
            int best_idx = -1;
            float min_dist = MATCH_DIST_THRESHOLD;
            float cx = detected.box.x + detected.box.width / 2.0f;
            float cy = detected.box.y + detected.box.height / 2.0f;

            for (size_t i = 0; i < tracker_db.size(); i++) {
                TrackedFace& t = tracker_db[i];
                float tx = t.box.x + t.box.width / 2.0f;
                float ty = t.box.y + t.box.height / 2.0f;
                float dist = std::sqrt(std::pow(cx - tx, 2) + std::pow(cy - ty, 2));
                if (dist < min_dist) { min_dist = dist; best_idx = i; }
            }

            if (best_idx != -1) {
                // MATCHED
                TrackedFace& t = tracker_db[best_idx];
                t.box = detected.box;
                t.missing_frames = 0;

                // Update Logic: Check Age/Rec periodically
                if (t.frames_since_analysis >= ANALYSIS_INTERVAL || t.age_hist.size() < 5) {
                    std::vector<FaceResult> single = {detected};
                    runAgeGender(img, single);
                    runRecognition(img, single);

                    FaceResult& fresh = single[0];
                    t.age_hist.push_back(fresh.age);
                    t.gender_hist.push_back(fresh.gender);
                    if(t.age_hist.size() > HISTORY_SIZE) t.age_hist.pop_front();
                    if(t.gender_hist.size() > HISTORY_SIZE) t.gender_hist.pop_front();

                    long sum = 0; for(int a : t.age_hist) sum+=a;
                    t.stable_age = sum / t.age_hist.size();

                    int m_vote = 0; for(int g : t.gender_hist) m_vote+=g;
                    t.stable_gender = (m_vote > (int)t.gender_hist.size()/2) ? 1 : 0;

                    if (fresh.name != "Unknown") t.name = fresh.name;
                    t.embedding = fresh.embedding;
                    t.frames_since_analysis = 0;
                }

                FaceResult res = detected;
                res.age = t.stable_age;
                res.gender = t.stable_gender;
                res.name = t.name.empty() ? "Unknown" : t.name;
                res.embedding = t.embedding;
                final_results.push_back(res);

            } else {
                // NEW FACE
                std::vector<FaceResult> single = {detected};
                runAgeGender(img, single);
                runRecognition(img, single);
                FaceResult& fresh = single[0];

                TrackedFace t;
                t.id = next_face_id++;
                t.box = fresh.box;
                t.age_hist.push_back(fresh.age);
                t.gender_hist.push_back(fresh.gender);
                t.stable_age = fresh.age;
                t.stable_gender = fresh.gender;
                t.name = fresh.name;
                t.embedding = fresh.embedding;
                t.frames_since_analysis = 0;

                tracker_db.push_back(t);
                final_results.push_back(fresh);
            }
        }

        tracker_db.erase(std::remove_if(tracker_db.begin(), tracker_db.end(), [](const TrackedFace& t){
                             return t.missing_frames > MAX_MISSING_FRAMES;
                         }), tracker_db.end());

        return final_results;

    } catch (const std::exception& e) {
        std::cerr << "Run Error: " << e.what() << std::endl;
        return {};
    }
}

void FaceSystem::runRecognition(const cv::Mat& img, std::vector<FaceResult>& faces) {
    if (!sessRec) return;
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputNamePtr = sessRec->GetInputNameAllocated(0, allocator);
        auto outputNamePtr = sessRec->GetOutputNameAllocated(0, allocator);
        const char* inName = inputNamePtr.get();
        const char* outName = outputNamePtr.get();

        for (auto& face : faces) {
            cv::Mat aligned = alignFace(img, face.kps);
            if (aligned.empty()) continue;

            cv::Mat blob;
            cv::dnn::blobFromImage(aligned, blob, 1.0/127.5, cv::Size(112, 112), cv::Scalar(127.5, 127.5, 127.5), true, false);
            std::vector<int64_t> inputShape = {1, 3, 112, 112};
            auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputOrt = Ort::Value::CreateTensor<float>(memoryInfo, (float*)blob.data, 112*112*3, inputShape.data(), inputShape.size());

            auto outputs = sessRec->Run(Ort::RunOptions{nullptr}, &inName, &inputOrt, 1, &outName, 1);
            float* outputData = outputs[0].GetTensorMutableData<float>();

            face.embedding.assign(outputData, outputData + 512);

            float max_sim = 0.0f;
            std::string best_name = "Unknown";
            for (auto const& [name, known_emb] : known_faces) {
                float sim = calculateSimilarity(face.embedding, known_emb);
                if (sim > max_sim) { max_sim = sim; best_name = name; }
            }
            if (max_sim > REC_THRESHOLD) face.name = best_name;
        }
    } catch(...) {}
}

void FaceSystem::runAgeGender(const cv::Mat& img, std::vector<FaceResult>& faces) {
    if (!sessAge || faces.empty()) return;
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputNamePtr = sessAge->GetInputNameAllocated(0, allocator);
        auto outputNamePtr = sessAge->GetOutputNameAllocated(0, allocator);
        const char* inName = inputNamePtr.get();
        const char* outName = outputNamePtr.get();

        for (auto& face : faces) {
            cv::Mat aligned = alignFace(img, face.kps);
            if (aligned.empty()) continue;
            cv::Mat blob;
            cv::dnn::blobFromImage(aligned, blob, 1.0, cv::Size(96, 96), cv::Scalar(0, 0, 0), true, false);
            std::vector<int64_t> inputShape = {1, 3, 96, 96};
            auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputOrt = Ort::Value::CreateTensor<float>(memoryInfo, (float*)blob.data, 96*96*3, inputShape.data(), inputShape.size());

            auto outputs = sessAge->Run(Ort::RunOptions{nullptr}, &inName, &inputOrt, 1, &outName, 1);
            float* outputData = outputs[0].GetTensorMutableData<float>();

            face.gender = (outputData[1] > outputData[0]) ? 1 : 0;
            face.age = static_cast<int>(outputData[2] * 100);
        }
    } catch(...) {}
}
