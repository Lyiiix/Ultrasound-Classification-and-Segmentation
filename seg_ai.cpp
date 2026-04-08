#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// ========================================================================
// 1. 定义数据包裹  - 从 AI 线程传给主线程
// ========================================================================
struct FrameResult {
    cv::Mat frame;           // 画好高亮和文字的最终图像
    bool is_empty;           // 是否为空（作为下班信号 Sentinel）
    
    
    float prob_normal;       // 普通概率
    float prob_benign;       // 良性概率
    float prob_malignant;    // 恶性概率
    
    float aspect_ratio;      // 纵横径比
    float relative_area;     // 病灶面积占比

    float center_x_ratio;    // 病灶中心点 X 坐标占 ROI 宽度的比例
    float center_y_ratio;    // 病灶中心点 Y 坐标占 ROI 高度的比例
};

// ========================================================================
// 2. SafeQueue (线程安全队列)
// ========================================================================
template<typename T>
class SafeQueue {
private:
    std::queue<T> q;
    std::mutex m;
    std::condition_variable cv_cond;
    size_t max_size;
    bool active; 

public:
    SafeQueue(size_t size = 30) : max_size(size), active(true) {}

    void stop() {
        std::unique_lock<std::mutex> lock(m);
        active = false;
        cv_cond.notify_all(); 
    }

    bool push(const T& item) {
        std::unique_lock<std::mutex> lock(m);
        cv_cond.wait(lock, [this]() { return q.size() < max_size || !active; });
        if (!active) return false; 
        
        q.push(item);
        lock.unlock();
        cv_cond.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(m);
        cv_cond.wait(lock, [this]() { return !q.empty() || !active; });
        
        if (q.empty() && !active) return false; 
        
        item = std::move(q.front()); 
        q.pop();
        lock.unlock();
        cv_cond.notify_all(); 
        return true;
    }

    bool pop_wait(T& item, int timeout_ms) {
        std::unique_lock<std::mutex> lock(m);
        if (!cv_cond.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                              [this]() { return !q.empty() || !active; })) {
            return false; 
        }
        if (q.empty() && !active) return false;
        
        item = std::move(q.front());
        q.pop();
        lock.unlock();
        cv_cond.notify_all();
        return true;
    }
};

// ========================================================================
// 3. 视频处理流水线核心类 (三线程)
// ========================================================================
class AIVideoPipeline {
private:
    SafeQueue<cv::Mat> queue_raw;
    SafeQueue<FrameResult> queue_processed; 
    SafeQueue<cv::Mat> queue_saved;

    std::atomic<bool> is_running;
    std::thread t_read;
    std::thread t_process;
    std::thread t_write;

    std::string video_path;
    std::string output_path;

    // 为三种概率分别建立滑动平均历史队列
    std::vector<float> hist_normal;
    std::vector<float> hist_benign;
    std::vector<float> hist_malignant;
    const int history_size = 30; // 历史 15 帧平滑滤波

    void abort() {
        is_running = false;
        queue_raw.stop();
        queue_processed.stop();
        queue_saved.stop();
    }

    // --- 极速预处理 ---
    std::vector<float> preprocess(const cv::Mat& roi_img) {
        cv::Mat resized, float_img;
        cv::resize(roi_img, resized, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
        resized.convertTo(float_img, CV_32FC1, 1.0f / 255.0f);
        float_img = (float_img - 0.5f) / 0.5f;
        std::vector<float> input_tensor(256 * 256);
        std::memcpy(input_tensor.data(), float_img.data, 256 * 256 * sizeof(float));
        return input_tensor;
    }

    // --- 读取线程 ---
    void threadRead() { 
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << " 读取线程：无法打开视频！" << std::endl;
            abort();
            return;
        }
        cv::Mat frame;
        while (is_running) {
            cap >> frame;
            if (frame.empty()) {
                std::cout << " 读取完毕，发送下班信号 (空图)..." << std::endl;
                queue_raw.push(cv::Mat()); // 发送 Sentinel (结束标志)
                break;
            }
            if (!queue_raw.push(frame.clone())) break; 
        }
        cap.release();
    }

    // --- 保存线程 ---
    void threadWrite(int fps, int width, int height) { 
        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
        cv::Mat frame_to_save;
        
        while (is_running && queue_saved.pop(frame_to_save)) {
            if (frame_to_save.empty()) break; // 接力结束，完美收工
            writer.write(frame_to_save);
        }
        writer.release();
        std::cout << " 写入线程：视频保存完毕，下班。" << std::endl;
    }

    // --- AI 处理线程 ---
    void threadProcess() {
        try {
            //  检查你的 ONNX 模型名字是否正确
            const wchar_t* model_path = L"ultrasound_multitask.onnx"; 
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Pipeline");
            Ort::SessionOptions session_options;
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            Ort::Session session(env, model_path, session_options);
            
            const char* input_names[] = {"input_image"};
            const char* output_names[] = {"cls_output", "seg_output"};

            cv::Mat frame;
            while (is_running && queue_raw.pop(frame)) {
                if (frame.empty()) { 
                    FrameResult end_signal;
                    end_signal.is_empty = true;
                    queue_processed.push(end_signal); 
                    break;
                }

                //  检查这里的 ROI 坐标
                //cv::Rect roi_rect(100, 50, 600, 500); 
                cv::Rect roi_rect(310, 180, 420, 420); // 根据实际视频调整
                roi_rect &= cv::Rect(0, 0, frame.cols, frame.rows); 
                //cv::rectangle(frame, roi_rect, cv::Scalar(0, 255, 0), 2);
                cv::Mat gray_frame;
                cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
                cv::Mat roi_img = gray_frame(roi_rect);

                // 1. 推理
                auto input_data = preprocess(roi_img);
                std::vector<int64_t> input_shape = {1, 1, 256, 256};
                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
                auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);

                // ----------------------------------------------------
                // 2. 解析分类概率 (分别计算三种概率的滑动平均)
                // ----------------------------------------------------
                float* cls_ptr = output_tensors[0].GetTensorMutableData<float>();
                float max_logit = std::max({cls_ptr[0], cls_ptr[1], cls_ptr[2]});
                float sum_exp = std::exp(cls_ptr[0]-max_logit) + std::exp(cls_ptr[1]-max_logit) + std::exp(cls_ptr[2]-max_logit);
                
                // 计算当前帧的三个概率
                float curr_normal = std::exp(cls_ptr[0] - max_logit) / sum_exp;
                float curr_benign = std::exp(cls_ptr[1] - max_logit) / sum_exp;
                float curr_malignant = std::exp(cls_ptr[2] - max_logit) / sum_exp;
                
                // 计算 Normal 平滑
                hist_normal.push_back(curr_normal);
                if (hist_normal.size() > history_size) hist_normal.erase(hist_normal.begin());
                float avg_normal = std::accumulate(hist_normal.begin(), hist_normal.end(), 0.0f) / hist_normal.size();

                // 计算 Benign 平滑
                hist_benign.push_back(curr_benign);
                if (hist_benign.size() > history_size) hist_benign.erase(hist_benign.begin());
                float avg_benign = std::accumulate(hist_benign.begin(), hist_benign.end(), 0.0f) / hist_benign.size();

                // 计算 Malignant 平滑
                hist_malignant.push_back(curr_malignant);
                if (hist_malignant.size() > history_size) hist_malignant.erase(hist_malignant.begin());
                float avg_malignant = std::accumulate(hist_malignant.begin(), hist_malignant.end(), 0.0f) / hist_malignant.size();

                // ----------------------------------------------------
                // 3. 解析分割 Mask
                // ----------------------------------------------------
                float* seg_ptr = output_tensors[1].GetTensorMutableData<float>();
                cv::Mat mask_256(256, 256, CV_32FC1, seg_ptr);
                cv::Mat sigmoid_mask;
                cv::exp(-mask_256, sigmoid_mask);
                sigmoid_mask = 1.0f / (1.0f + sigmoid_mask);
                cv::Mat binary_mask;
                cv::threshold(sigmoid_mask, binary_mask, 0.5, 255.0, cv::THRESH_BINARY);
                binary_mask.convertTo(binary_mask, CV_8UC1);

                cv::Mat resized_mask;
                cv::resize(binary_mask, resized_mask, cv::Size(roi_rect.width, roi_rect.height), 0, 0, cv::INTER_NEAREST);

                // ----------------------------------------------------
                // 渲染画面 (根据概率动态决定颜色)
                // ----------------------------------------------------
                cv::Scalar mask_color;
                // 业务逻辑：如果恶性概率 > 良性概率，说明危险系数高，涂红色！
                // 否则，说明大概率是良性结节，涂绿色！
                if (avg_malignant > avg_benign) {
                    mask_color = cv::Scalar(0, 0, 255); // BGR: 红色
                } else {
                    mask_color = cv::Scalar(0, 255, 0); // BGR: 绿色
                }

                cv::Mat roi_color = frame(roi_rect); 
                cv::Mat overlay;
                roi_color.copyTo(overlay);
                overlay.setTo(mask_color, resized_mask); // 动态颜色涂色
                cv::addWeighted(overlay, 0.4, roi_color, 0.6, 0, roi_color); 
                
                // ----------------------------------------------------
                // 打印全部概率
                // ----------------------------------------------------
                char buf_n[64], buf_b[64], buf_m[64];
                // %.2f%% 保证永远只显示两位小数，排版像死一样稳定
                snprintf(buf_n, sizeof(buf_n), "Normal:    %5.2f%%", avg_normal * 100);
                snprintf(buf_b, sizeof(buf_b), "Benign:    %5.2f%%", avg_benign * 100);
                snprintf(buf_m, sizeof(buf_m), "Malignant: %5.2f%%", avg_malignant * 100);

                // 分别在不同高度画出三行字 (普通白色，良性绿色，恶性红色)
                cv::putText(frame, buf_n, cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                cv::putText(frame, buf_b, cv::Point(30, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, buf_m, cv::Point(30, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

                // 提取特征参数
                cv::Rect bbox = cv::boundingRect(resized_mask);
                float aspect_ratio = (bbox.width > 0) ? ((float)bbox.height / bbox.width) : 0.0f;
                float relative_area = (float)cv::countNonZero(resized_mask) / (roi_rect.width * roi_rect.height);
                //   计算中心点坐标
                float center_x = bbox.x + bbox.width / 2.0f;
                float center_y = bbox.y + bbox.height / 2.0f;

                //    计算相对位置 (除以 ROI 的宽高，变成 0~1 的比例)
                float center_x_ratio = center_x / roi_rect.width;  
                float center_y_ratio = center_y / roi_rect.height;
                // 打包并传递 (记得补充新增的两个概率)
                FrameResult result;
                result.frame = frame;
                result.is_empty = false;
                result.prob_normal = avg_normal;     
                result.prob_benign = avg_benign;     
                result.prob_malignant = avg_malignant;
                result.aspect_ratio = aspect_ratio;
                result.relative_area = relative_area;
                result.center_x_ratio = center_x_ratio;
                result.center_y_ratio = center_y_ratio;
                if (!queue_processed.push(result)) break;
            }
        } catch (const std::exception& e) {
            std::cerr << "AI 线程异常: " << e.what() << std::endl;
            abort();
        }
    }

public:
    AIVideoPipeline(const std::string& input, const std::string& output) 
        : video_path(input), output_path(output), 
          queue_raw(50), queue_processed(50), queue_saved(50), is_running(false) {}

    // ========================================================================
    // UI 渲染及流程控制
    // ========================================================================
    void run() {
        // --- 获取视频原始信息 ---
        cv::VideoCapture temp_cap(video_path);
        if (!temp_cap.isOpened()) {
            std::cerr << "无法打开输入视频 " << video_path << std::endl;
            return;
        }
        int width = (int)temp_cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)temp_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int fps = (int)temp_cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 25; // 防止某些视频读不出 fps 导致写入崩溃
        temp_cap.release();

        is_running = true;
        std::cout << "正在启动流水线..." << std::endl;

        // 启动三个工作线程
        t_read = std::thread(&AIVideoPipeline::threadRead, this);
        t_process = std::thread(&AIVideoPipeline::threadProcess, this);
        t_write = std::thread(&AIVideoPipeline::threadWrite, this, fps, width, height);

        FrameResult display_res;
        auto timer_start = std::chrono::high_resolution_clock::now();
        int frame_count = 0;
        float current_fps = 0.0f; // 初始帧率设为 0
        while (is_running) {
            auto frame_start_time = std::chrono::high_resolution_clock::now();

            // 尝试从 AI 线程拿快递
            if (queue_processed.pop_wait(display_res, 15)) { 
                
                // 哨兵检测：如果拿到空包裹，通知保存线程并打烊
                if (display_res.is_empty) { 
                    queue_saved.push(cv::Mat()); 
                    break; 
                }

                //计算单帧耗时 (Latency)
                auto frame_end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> iteration_ms = frame_end_time - frame_start_time;

                // 计算并更新 FPS (每秒更新一次)
                frame_count++;
                auto timer_now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = timer_now - timer_start;

                if (elapsed.count() >= 1.0) { // 满一秒了，更新一次
                    current_fps = frame_count / (float)elapsed.count();
                    frame_count = 0;
                    timer_start = timer_now;

                    // 在控制台打印当前状态 (实时吞吐量)
                    std::cout << "  >>> [System Monitor] 实时处理帧率: " 
                            << std::fixed << std::setprecision(1) << current_fps << " FPS" << std::endl;
                }

                // 控制台实时打印单帧耗时 (不换行，用 \r 或者间隔符)
                // 这样不会刷屏，方便观察抖动
                std::cout << "\r[Frame Info] 处理耗时: " << std::fixed << std::setprecision(1) 
                        << iteration_ms.count() << " ms    " << std::flush;

                // UI 渲染：把 FPS 画在图上
                std::string fps_text = "FPS: " + std::to_string((int)current_fps); // 转 int 让 UI 更整洁
                cv::putText(display_res.frame, fps_text, cv::Point(width - 150, 40), 
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
                // 发送给保存线程
                queue_saved.push(display_res.frame.clone());

                // 显示画面
                cv::imshow("AI Multi-task Ultrasound", display_res.frame);
            }

            if (cv::waitKey(1) == 27) { // 监听 ESC 键
                std::cout << "用户手动中断！" << std::endl;
                abort(); 
                break; 
            }
        }

        // --- 完整补充：等待线程安全结束清理 ---
        std::cout << "正在等待所有线程安全退出..." << std::endl;
        if (t_read.joinable()) t_read.join();
        if (t_process.joinable()) t_process.join();
        if (t_write.joinable()) t_write.join();

        cv::destroyAllWindows();
        std::cout << "系统安全关闭！" << std::endl;
    }
};

// ========================================================================
// main 入口
// ========================================================================
int main() {
    system("chcp 65001 > nul");
    AIVideoPipeline pipeline("62321-3955051_20130502103029.webm", "output_segmented.mp4");
    pipeline.run();
    return 0;
}
