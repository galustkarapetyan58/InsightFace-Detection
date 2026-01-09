#include "facesystem.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>

using namespace cv;
using namespace std;

FaceSystem::FaceSystem() : env(ORT_LOGGING_LEVEL_WARNING, "FaceSystem") {}

bool FaceSystem::loadModels(const string& detPath, const string& recPath, const string& gaPath) {
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // Load all 3 models
        wstring wDet(detPath.begin(), detPath.end());
        wstring wRec(recPath.begin(), recPath.end());
        wstring wGa(gaPath.begin(), gaPath.end());

        det_session = make_unique<Ort::Session>(env, wDet.c_str(), options);
        rec_session = make_unique<Ort::Session>(env, wRec.c_str(), options);
        ga_session  = make_unique<Ort::Session>(env, wGa.c_str(), options);

        // --- PRE-CALCULATE SCRFD METADATA (Safely) ---
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 1. Input Names
        size_t num_in = det_session->GetInputCount();
        input_names_stg.clear(); 
        input_names_ptr.clear();
        input_names_stg.reserve(num_in);
        input_names_ptr.reserve(num_in);

        for(size_t i=0; i<num_in; i++) {
            auto name = det_session->GetInputNameAllocated(i, allocator);
            input_names_stg.push_back(name.get());
        }
        for(size_t i=0; i<num_in; i++) {
            input_names_ptr.push_back(input_names_stg[i].c_str());
        }

        // 2. Output Names
        size_t num_out = det_session->GetOutputCount();
        output_names_stg.clear(); 
        output_names_ptr.clear();
        output_names_stg.reserve(num_out);
        output_names_ptr.reserve(num_out);

        for(size_t i=0; i<num_out; i++) {
            auto name = det_session->GetOutputNameAllocated(i, allocator);
            output_names_stg.push_back(name.get());
        }
        for(size_t i=0; i<num_out; i++) {
             output_names_ptr.push_back(output_names_stg[i].c_str());
        }

        return true;
    } catch (const Ort::Exception& e) {
        cerr << "[AI ERROR]: " << e.what() << endl;
        return false;
    }
}

// --- 1. DETECTION (SCRFD 9-OUTPUT FORMAT) ---
vector<FaceObject> FaceSystem::detectSCRFD(const Mat& frame) {
    vector<FaceObject> faces;
    try {
        if (!det_session) { cerr << "Session Null!" << endl; return faces; }
        if (input_names_ptr.empty()) { cerr << "Names Empty!" << endl; return faces; }

        // 1. Preprocess: Letterbox to 640x640
        int target_size = 640;
        int width = frame.cols;
        int height = frame.rows;
        
        float scale = min((float)target_size / width, (float)target_size / height);
        int new_w = (int)(width * scale);
        int new_h = (int)(height * scale);
        
        Mat resized_img;
        resize(frame, resized_img, Size(new_w, new_h));
        
        Mat input_img(target_size, target_size, CV_8UC3, Scalar(0, 0, 0));
        resized_img.copyTo(input_img(Rect(0, 0, new_w, new_h)));
        
        // Normalize: (x - 127.5) / 128.0 
        Mat blob = dnn::blobFromImage(input_img, 1.0/128.0, Size(target_size, target_size), Scalar(127.5, 127.5, 127.5), true, false);

        // 2. Inference
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        vector<int64_t> input_shape = {1, 3, target_size, target_size};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, blob.total(), input_shape.data(), 4);

        auto outputs = det_session->Run(Ort::RunOptions{nullptr}, 
                                        input_names_ptr.data(), &input_tensor, 1, 
                                        output_names_ptr.data(), output_names_ptr.size());
        
        // 3. Post-Process - SCRFD 9-output format
        // Model outputs: [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
        // Scores are 2-channel (background, face)
        
        struct StrideInfo {
            int score_idx = -1;
            int bbox_idx = -1; 
            int kps_idx = -1;
            int grid_size = 0;
        };
        
        map<int, StrideInfo> stride_info;
        stride_info[8].grid_size = 80;
        stride_info[16].grid_size = 40;
        stride_info[32].grid_size = 20;
        
        // Map outputs by size
        for(int i=0; i<outputs.size(); i++) {
            size_t el = outputs[i].GetTensorTypeAndShapeInfo().GetElementCount();
            
            // Stride 8 (80x80 grid)
            if(el == 12800) stride_info[8].score_idx = i;      // 80*80*2
            else if(el == 51200) stride_info[8].bbox_idx = i;  // 80*80*4*2
            else if(el == 128000) stride_info[8].kps_idx = i;  // 80*80*10*2
            
            // Stride 16 (40x40 grid)
            else if(el == 3200) stride_info[16].score_idx = i;    // 40*40*2
            else if(el == 12800) stride_info[16].bbox_idx = i;    // 40*40*4*2
            else if(el == 32000) stride_info[16].kps_idx = i;     // 40*40*10*2
            
            // Stride 32 (20x20 grid)
            else if(el == 800) stride_info[32].score_idx = i;     // 20*20*2
            else if(el == 3200) stride_info[32].bbox_idx = i;     // 20*20*4*2
            else if(el == 8000) stride_info[32].kps_idx = i;      // 20*20*10*2
        }

        int strides[] = {8, 16, 32};  // All scales
        vector<Rect> boxes;
        vector<float> confs;
        vector<vector<Point2f>> all_kpss;

        for(int stride : strides) {
            auto& info = stride_info[stride];
            if(info.score_idx == -1 || info.bbox_idx == -1 || info.kps_idx == -1) continue;
            
            float* s_ptr = outputs[info.score_idx].GetTensorMutableData<float>();
            float* b_ptr = outputs[info.bbox_idx].GetTensorMutableData<float>();
            float* k_ptr = outputs[info.kps_idx].GetTensorMutableData<float>();
            
            int grid = info.grid_size;
            
            for(int cy=0; cy<grid; cy++) {
                for(int cx=0; cx<grid; cx++) {
                    int idx = cy * grid + cx;
                    
                    // Score is 2-channel: [background, face]
                    // Use softmax: exp(face) / (exp(bg) + exp(face))
                    float score_bg = s_ptr[idx * 2 + 0];
                    float score_face = s_ptr[idx * 2 + 1];
                    
                    float exp_bg = exp(score_bg);
                    float exp_face = exp(score_face);
                    float score = exp_face / (exp_bg + exp_face);
                    
                    // Balanced threshold
                    if(score < 0.7f) continue;
                    
                    float anchor_x = (cx + 0.5f) * stride;
                    float anchor_y = (cy + 0.5f) * stride;
                    
                    // BBox: distance format
                    float dx1 = b_ptr[idx*4 + 0] * stride;
                    float dy1 = b_ptr[idx*4 + 1] * stride;
                    float dx2 = b_ptr[idx*4 + 2] * stride;
                    float dy2 = b_ptr[idx*4 + 3] * stride;
                    
                    float x1 = (anchor_x - dx1) / scale;
                    float y1 = (anchor_y - dy1) / scale;
                    float x2 = (anchor_x + dx2) / scale;
                    float y2 = (anchor_y + dy2) / scale;
                    
                    // Bbox validation - must be reasonable size
                    float w = x2 - x1;
                    float h = y2 - y1;
                    
                    // Filter out invalid boxes
                    if(w < 10 || h < 10 || w > 1000 || h > 1000) continue;
                    if(w/h > 3.0f || h/w > 3.0f) continue;  // Aspect ratio check
                    
                    // Landmarks
                    vector<Point2f> kps;
                    for(int k=0; k<5; k++) {
                       float kx = (anchor_x + k_ptr[idx*10 + k*2] * stride) / scale;
                       float ky = (anchor_y + k_ptr[idx*10 + k*2 + 1] * stride) / scale;
                       kps.push_back(Point2f(kx, ky));
                    }
                    
                    boxes.push_back(Rect(x1, y1, x2-x1, y2-y1));
                    confs.push_back(score);
                    all_kpss.push_back(kps);
                }
            }
        }
        
        // NMS - standard
        vector<int> indices;
        dnn::NMSBoxes(boxes, confs, 0.5f, 0.45f, indices);
        
        vector<FaceObject> temp_faces;
        for(int idx : indices) {
            FaceObject obj;
            obj.box = boxes[idx];
            obj.confidence = confs[idx];
            obj.landmarks = all_kpss[idx];
            temp_faces.push_back(obj);
        }
        
        // --- KEY FIX: Keep ONLY the largest face ---
        // This eliminates all small background detections
        if(!temp_faces.empty()) {
            // Sort by area (descending)
            std::sort(temp_faces.begin(), temp_faces.end(), [](const FaceObject& a, const FaceObject& b) {
                return (a.box.area()) > (b.box.area());
            });
            
            // Keep ONLY the biggest face (the user)
            faces.push_back(temp_faces[0]);
        }

    } catch (const Ort::Exception& e) {
        cerr << "[AI ERROR] SCRFD Inference failed: " << e.what() << endl;
    }

    return faces;
}

// --- 2. ALIGNMENT (Warp Affine) ---
Mat FaceSystem::alignFace(const Mat& frame, const vector<Point2f>& kps) {
    // Standard InsightFace 112x112 target points
    float target[5][2] = {
        {38.2946f, 51.6963f}, {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f}, {70.7299f, 92.2041f}
    };

    Mat src(5, 2, CV_32F);
    Mat dst(5, 2, CV_32F, target);
    for(int i=0; i<5; i++) {
        src.at<float>(i,0) = kps[i].x;
        src.at<float>(i,1) = kps[i].y;
    }

    Mat M = estimateAffinePartial2D(src, dst);
    Mat aligned;
    warpAffine(frame, aligned, M, Size(112, 112));
    return aligned;
}

// --- 3. RECOGNITION & AGE ---
void FaceSystem::analyzeFace(const Mat& frame, FaceObject& face) {
    Mat aligned = alignFace(frame, face.landmarks);

    // B. Run Gender/Age
    if(ga_session) {
         try {
            Mat blobGA = dnn::blobFromImage(aligned, 1.0, Size(96, 96), Scalar(0,0,0), true);
            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            int64_t shapeGA[] = {1, 3, 96, 96};
            Ort::Value tensorGA = Ort::Value::CreateTensor<float>(memory, (float*)blobGA.data, blobGA.total(), shapeGA, 4);

            Ort::AllocatorWithDefaultOptions allocator;
            auto name_in_ptr = ga_session->GetInputNameAllocated(0, allocator);
            auto name_out_ptr = ga_session->GetOutputNameAllocated(0, allocator);
            const char* in_name = name_in_ptr.get();
            const char* out_name = name_out_ptr.get();

            auto outGA_vals = ga_session->Run(Ort::RunOptions{nullptr}, &in_name, &tensorGA, 1, &out_name, 1);
            float* ga_data = outGA_vals[0].GetTensorMutableData<float>();
            
            // Gender: [0]=Male, [1]=Female
            face.gender = (ga_data[0] > ga_data[1]) ? "M" : "F";
            
            // Age: Model outputs 0-1 range, multiply by 100
            float raw_age = ga_data[2];
            face.age = (int)(raw_age * 100);
         } catch(std::exception& e) {
             cerr << "GA Error: " << e.what() << endl;
         }
    }

    // C. Skip Recognition (not needed for display, slows down FPS)
    // Uncomment if you need embeddings:
    /*
    if(rec_session) {
        try {
            Mat blobRec = dnn::blobFromImage(aligned, 1.0/127.5, Size(112, 112), Scalar(127.5, 127.5, 127.5), true);
            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            int64_t shapeRec[] = {1, 3, 112, 112};
            Ort::Value tensorRec = Ort::Value::CreateTensor<float>(memory, (float*)blobRec.data, blobRec.total(), shapeRec, 4);

            Ort::AllocatorWithDefaultOptions allocator;
            auto name_in_ptr = rec_session->GetInputNameAllocated(0, allocator);
            auto name_out_ptr = rec_session->GetOutputNameAllocated(0, allocator);
            const char* in_name = name_in_ptr.get();
            const char* out_name = name_out_ptr.get();

            auto outRec_vals = rec_session->Run(Ort::RunOptions{nullptr}, &in_name, &tensorRec, 1, &out_name, 1);
            float* emb_data = outRec_vals[0].GetTensorMutableData<float>();
            face.embedding.assign(emb_data, emb_data + 512);
        
        } catch(std::exception& e) {
            cerr << "Rec Error: " << e.what() << endl;
            face.embedding = {0};
        }
    }
    */
}

// --- MAIN LOOP ---
void FaceSystem::runWebcam() {
    VideoCapture cap(0);
    
    if(!cap.isOpened()) {
        cerr << "[ERROR] Could not open camera!" << endl;
        return;
    }
    
    // Brief warmup
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    Mat frame;

    while(true) {
        cap >> frame;
        
        if(frame.empty()) {
            cerr << "[ERROR] Empty frame!" << endl;
            break;
        }

        // 1. Detect
        vector<FaceObject> faces = detectSCRFD(frame);

        // 2. Analyze & Draw
        for(auto& face : faces) {
            try {
                analyzeFace(frame, face);
            } catch(std::exception& e) {
                cerr << "[ERROR] analyzeFace failed: " << e.what() << endl;
                continue;  // Skip this face
            }

            // 3. Draw
            rectangle(frame, face.box, Scalar(0, 255, 0), 2);
            for(auto& p : face.landmarks) circle(frame, p, 2, Scalar(0,0,255), -1);
            
            // Clean professional label format
            string gender_full = (face.gender == "M") ? "Male" : "Female";
            string label = gender_full + ", Age: " + to_string(face.age);
            putText(frame, label, Point(face.box.x, face.box.y - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);
        }

        imshow("InsightFace Professional", frame);
        int key = waitKey(1);
        if(key == 27) break;  // ESC to exit
    }
}
