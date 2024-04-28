#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
// #include <immintrin.h>
#include <arm_neon.h>
struct Node {
    std::vector<float> dimensions;
};

// SIMD 版本的计算距离函数
// NEON 版本的计算距离函数
inline float VectorizedDistance(const Node& x, const Node& z, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (int i = 0; i < n; i += 4) {
        float32x4_t x_vec = vld1q_f32(&x.dimensions[i]);
        float32x4_t z_vec = vld1q_f32(&z.dimensions[i]);
        float32x4_t diff = vsubq_f32(x_vec, z_vec);
        float32x4_t squared = vmulq_f32(diff, diff);
        sum = vaddq_f32(sum, squared);
    }

    // 计算 sum 的平方根
    float32x2_t sum2 = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    float result = vget_lane_f32(vsqrt_f32(sum2), 0);

    return result;
}
// NEON 版本的更新簇中心函数
void VectorizedUpdateClusterCenter(const std::vector<Node>& data, const std::vector<int>& idx, int k, int n,int m) {
    for (int j = 0; j < k; j++) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        int count = 0;

        // 使用 NEON 对每个簇进行并行处理
        for (int i = 0; i < n; i += 4) {
            // 将整型簇索引转换为浮点型
            uint32x4_t mask = vceqq_u32(vld1q_u32(reinterpret_cast<const uint32_t*>(&idx[i])), vdupq_n_u32(static_cast<uint32_t>(j)));
            float32x4_t data_vec = vld1q_f32(&data[i].dimensions[0]);
            float32x4_t selected_data = vbslq_f32(mask, data_vec, vdupq_n_f32(0.0f));
            sum = vaddq_f32(sum, selected_data);
            count += vaddvq_u32(mask);
        }

        if (count > 0) {
            float32x4_t center = vdivq_f32(sum, vdupq_n_f32(static_cast<float>(count)));
            vst1q_f32(const_cast<float*>(&data[j].dimensions[0]), center);
        }
    }
}
void KMeans(int k, std::vector<Node>& data, int n, int m) {
    std::vector<Node> centers(k);
    std::vector<int> idx(n);
    std::vector<std::vector<float>> distances(n, std::vector<float>(k));

    // 初始化簇中心
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);
    for (int i = 0; i < k; i++) {
        centers[i] = data[dis(gen)];
    }

    bool cluster_changed = true;
    while (cluster_changed) {
        cluster_changed = false;

        // 计算每个样本点到簇中心的距离
        for (int i = 0; i < n; i++) {
            float min_distance = std::numeric_limits<float>::max();
            int min_index = -1;
            for (int j = 0; j < k; j++) {
                float distance = VectorizedDistance(data[i], centers[j], m);
                distances[i][j] = distance;
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = j;
                }
            }
            if (idx[i] != min_index) {
                idx[i] = min_index;
                cluster_changed = true;
            }
        }

        // 更新簇中心
        VectorizedUpdateClusterCenter(data, idx, k, n, m);
    }

    // 输出聚类结果
    if(cluster_changed ==false){
    for (int i = 0; i < k; i++) {
        std::cout << "Cluster " << i + 1 << " center: ";
        for (float dim : centers[i].dimensions) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
    }
}
void generateStructuredData(std::vector<Node>& data, int n, int m) {
    data.resize(n);
    for (int i = 0; i < n; i++) {
        data[i].dimensions.resize(m);
        for (int j = 0; j < m; j++) {
            data[i].dimensions[j] = static_cast<float>(i + 1);
        }
    }
}
#include <stdio.h>
#include <time.h>
int main() {
    int n = 100000, m = 10, k = 5;
    std::vector<Node> data;

    // 生成随机数据点
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::normal_distribution<float> distribution(0.0f, 1.0f);
    // for (Node& node : data) {
    //     node.dimensions.resize(m);
    //     for (float& dim : node.dimensions) {
    //         dim = distribution(gen);
    //     }
    // }

    generateStructuredData(data, n, m);
    // 输出 data 容器中的数据
    std::cout << "Generated data:\n";
    for (const auto& node : data) {
        std::cout << "Node: ";
        for (float dim : node.dimensions) {
            std::cout << dim << " ";
        }
        std::cout << "\n";
    }

    // auto start_time = std::chrono::high_resolution_clock::now();
    // for(int p=1;p<=100;p++){
    // KMeans(k, data, n, m);
    // }
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // std::cout << "K-Means algorithm execution time: " << elapsed_time << " ms" << std::endl;

    // return 0;
    struct timespec start_time, end_time;
    long long elapsed_time;

    // 获取开始时间
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    KMeans(k, data, n, m);
    // 获取结束时间
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    // 计算执行时间
    elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;

    printf("Elapsed time: %.9f seconds\n", elapsed_time);


}