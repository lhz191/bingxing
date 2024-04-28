#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>
#include <xmmintrin.h>

using namespace std;

struct Node {
    vector<float> dimensions;
};

// 计算两个节点之间的欧氏距离
float Distance(const Node& x, const Node& z, int n) {
    float dist = 0.0;
    for (int i = 0; i < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(&x.dimensions[i]);
        __m128 z_vec = _mm_loadu_ps(&z.dimensions[i]);
        __m128 diff = _mm_sub_ps(x_vec, z_vec);
        __m128 diff_squared = _mm_mul_ps(diff, diff);
        __m128 sum = _mm_hadd_ps(diff_squared, diff_squared);
        sum = _mm_hadd_ps(sum, sum);
        dist += _mm_cvtss_f32(sum);
    }
    for (int i = (n / 4) * 4; i < n; i++) {
        float diff = x.dimensions[i] - z.dimensions[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}

// K-Medoids 算法实现
void KMedoids(int k, std::vector<Node>& data, int n, int m) {
    std::vector<int> idx(n);
    std::vector<Node> medoids(k);

    // 初始化 medoids 为随机选择的 data 点
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);
    for (int i = 0; i < k; i++) {
        medoids[i] = data[dis(gen)];
    }

    bool cluster_changed = true;
    while (cluster_changed) {
        cluster_changed = false;

        // 分配数据点到最近的 medoid
        for (int i = 0; i < n; i++) {
            float min_distance = std::numeric_limits<float>::max();
            int min_index = -1;
            for (int j = 0; j < k; j++) {
                float distance = Distance(data[i], medoids[j], m);
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

        // 更新 medoids
        for (int j = 0; j < k; j++) {
            std::vector<Node> cluster;
            for (int i = 0; i < n; i++) {
                if (idx[i] == j) {
                    cluster.push_back(data[i]);
                }
            }
            if (!cluster.empty()) {
                int medoid_idx = 0;
                float min_total_distance = std::numeric_limits<float>::max();
                for (int i = 0; i < cluster.size(); i++) {
                    float total_distance = 0.0;
                    for (const auto& point : cluster) {
                        total_distance += Distance(cluster[i], point, m);
                    }
                    if (total_distance < min_total_distance) {
                        min_total_distance = total_distance;
                        medoid_idx = i;
                    }
                }
                medoids[j] = cluster[medoid_idx];
            }
        }
    }

    // 输出聚类结果
    for (int i = 0; i < k; i++) {
        std::cout << "Cluster " << i + 1 << " medoid: ";
        for (float dim : medoids[i].dimensions) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
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
#include <windows.h>
#include <sysinfoapi.h>
int main() {
    int n = 1000, m = 10, k = 5;
    std::vector<Node> data;

    generateStructuredData(data, n, m);

    FILETIME start_time, end_time;
    ULARGE_INTEGER start_time_us, end_time_us;

    GetSystemTimePreciseAsFileTime(&start_time);
    KMedoids(k, data, n, m);
    GetSystemTimePreciseAsFileTime(&end_time);

    start_time_us.LowPart = start_time.dwLowDateTime;
    start_time_us.HighPart = start_time.dwHighDateTime;
    end_time_us.LowPart = end_time.dwLowDateTime;
    end_time_us.HighPart = end_time.dwHighDateTime;

    ULONGLONG elapsed_time = end_time_us.QuadPart - start_time_us.QuadPart;
    ULONGLONG elapsed_seconds = elapsed_time / 10000000;
    ULONGLONG elapsed_nanoseconds = (elapsed_time % 10000000) * 100;

    printf("%llu.%09llu seconds\n", elapsed_seconds, elapsed_nanoseconds);

    return 0;
}