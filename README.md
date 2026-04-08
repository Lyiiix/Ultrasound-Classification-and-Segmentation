# Ultrasound-Classification-and-Segmentation

本项目针对实时医疗超声影像分析场景，通过 C++ 构建了一套高性能异步处理流水线。

高吞吐量：在1050上稳定运行于 45-52 FPS，满足实时临床需求。

双任务并行：单次模型前向传播同时实现 病灶分类 (良恶性判别) 与 实时分割 (ROI Mask)。

鲁棒并发设计：彻底杜绝多线程环境下的死锁与内存溢出。

项目采用解耦的四线程流水线架构，各模块通过高效率的 SafeQueue 通信：

Read Thread (视频读取)：负责 I/O 操作，通过 frame.clone() 确保内存隔离。

---AI Process Thread (推理核心)---

预处理：灰度化、维度缩放、归一化。

多任务推理：解析分类 Logits 与分割 Mask。

平滑滤波器：引入滑动平均算法（Moving Average）消除数值跳变。

UI/Main Thread：负责 OpenCV 渲染、FPS/Latency 监控及键盘交互。

Write Thread ：负责将处理后的数据压制为 MP4 视频，避免磁盘写入阻塞推理。
