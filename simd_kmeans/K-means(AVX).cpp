#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <immintrin.h>

struct Node {
    std::vector<float> dimensions;
};

// 利用AVX指令集优化距离计算函数
inline float VectorizedDistance(const Node& x, const Node& z, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&x.dimensions[i]);
        __m256 z_vec = _mm256_loadu_ps(&z.dimensions[i]);
        __m256 diff = _mm256_sub_ps(x_vec, z_vec);
        __m256 squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    float result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_hadd_ps(_mm_hadd_ps(sum128, sum128), sum128)));
    return result;
}

// AVX 版本的更新簇中心函数
void VectorizedUpdateClusterCenter(const std::vector<Node>& data, const std::vector<int>& idx,
                                  std::vector<Node>& centers, int k, int n, int m) {
    for (int j = 0; j < k; j++) {
        __m256 sum[8] = {_mm256_setzero_ps()}; // 为每个8个维度初始化一个 sum 向量
        int count = 0;
        for (int i = 0; i < n - (n % 8); i += 8) {
            __m256 mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&idx[i])), _mm256_set1_epi32(j)));
            for (int d = 0; d < m; d += 8) {
                __m256 data_vec = _mm256_and_ps(_mm256_loadu_ps(&data[i].dimensions[d]), mask);
                sum[d / 8] = _mm256_add_ps(sum[d / 8], data_vec);
            }
            count += _mm_popcnt_u32(_mm256_movemask_ps(mask));
        }
        // 处理最后几个不足 8 个的数据
        for (int i = n - (n % 8); i < n; i++) {
            if (idx[i] == j) {
                for (int d = 0; d < m; d++) {
                    sum[d / 8] = _mm256_add_ps(sum[d / 8], _mm256_set1_ps(data[i].dimensions[d]));
                }
                count++;
            }
        }
        if (count > 0) {
            for (int d = 0; d < m; d += 8) {
                __m256 center = _mm256_div_ps(sum[d / 8], _mm256_set1_ps(static_cast<float>(count)));
                _mm256_storeu_ps(&centers[j].dimensions[d], center);
            }
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
        for (int i = 0; i < n - (n % 8); i += 8) {
            __m256 min_distance = _mm256_set1_ps(std::numeric_limits<float>::max());
            __m256i min_index = _mm256_set_epi32(i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i);
            for (int j = 0; j < k; j++) {
                __m256 distance = _mm256_set_ps(VectorizedDistance(data[i + 7], centers[j], m),
                                              VectorizedDistance(data[i + 6], centers[j], m),
                                              VectorizedDistance(data[i + 5], centers[j], m),
                                              VectorizedDistance(data[i + 4], centers[j], m),
                                              VectorizedDistance(data[i + 3], centers[j], m),
                                              VectorizedDistance(data[i + 2], centers[j], m),
                                              VectorizedDistance(data[i + 1], centers[j], m),
                                              VectorizedDistance(data[i], centers[j], m));
                __m256 mask = _mm256_cmp_ps(distance, min_distance, _CMP_LT_OS);
                min_distance = _mm256_min_ps(min_distance, distance);
                min_index = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(min_index), _mm256_set1_ps(static_cast<float>(j)), mask));
            }
            _mm256_storeu_ps(&distances[i][0], min_distance);
            // 将 min_index 转换为整型并存储到 idx 数组
            alignas(32) int min_idx[8];
            _mm256_store_si256((__m256i*)min_idx, min_index);
            for (int j = 0; j < 8; j++)
                idx[i + j] = min_idx[j];
            // 检查是否有样本点发生了簇分配的变化
            __m256i idx_vec = _mm256_loadu_si256((__m256i*)&idx[i]);
            __m256i min_idx_vec = _mm256_load_si256((__m256i*)min_idx);
            __m256i cmp_result = _mm256_cmpeq_epi32(idx_vec, min_idx_vec);
            int cmp_mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_result));
            if (cmp_mask != 0xFF)
                cluster_changed = true;
        }

        // 处理最后几个不足 8 个的数据
        for (int i = n - (n % 8); i < n; i++) {
            float min_distance = std::numeric_limits<float>::max();
            int min_index = 0;
            for (int j = 0; j < k; j++) {
                float distance = VectorizedDistance(data[i], centers[j], m);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = j;
                }
            }
            distances[i][0] = min_distance;
            idx[i] = min_index;
            if (idx[i] != min_index)
                cluster_changed = true;
        }

        // 更新簇中心
        VectorizedUpdateClusterCenter(data, idx, centers, k, n, m);
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
for (int i = 0; i < n; i++) 
{
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
    FILETIME start_time, end_time;
    ULARGE_INTEGER start_time_us, end_time_us;
    // auto start_time = std::chrono::high_resolution_clock::now();
    // for(int p=1;p<=100;p++){
    // KMeans(k, data, n, m);
    // }
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_tim

    // std::cout << "K-Means algorithm execution time: " << elapsed_time << " ms" << std::endl;
    GetSystemTimePreciseAsFileTime(&start_time);

    KMeans(k, data, n, m);
    // 获取结束时间
    GetSystemTimePreciseAsFileTime(&end_time);

    // 计算执行时间
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
