#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <fstream>
#include <sstream>

using namespace std;

struct Node {
    vector<float> dimensions;
};

// AVX 版本的计算距离函数
inline float VectorizedDistance(const Node& x, const Node& z, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&x.dimensions[i]);
        __m256 z_vec = _mm256_loadu_ps(&z.dimensions[i]);
        __m256 diff = _mm256_sub_ps(x_vec, z_vec);
        __m256 squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
    }
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 total = _mm_add_ps(low, high);
    total = _mm_hadd_ps(total, total);
    total = _mm_hadd_ps(total, total);
    float result = _mm_cvtss_f32(_mm_sqrt_ps(total));
    return result;
}

// AVX 版本的更新簇中心函数
void VectorizedUpdateClusterCenter(const std::vector<Node>& data, const std::vector<int>& idx,
          std::vector<Node>& centers, int k, int n, int m) {
    for (int j = 0; j < k; j++) {
        __m256 sum[m];
        for (int d = 0; d < m; d++) {
            sum[d] = _mm256_setzero_ps();
        }
        int count = 0;
        for (int i = 0; i < n - (n % 8); i += 8) {
            __m256 mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&idx[i])),
                                                                 _mm256_set1_epi32(j)));
            for (int d = 0; d < m; d++) {
                __m256 data_vec = _mm256_and_ps(_mm256_loadu_ps(&data[i].dimensions[d]), mask);
                sum[d] = _mm256_add_ps(sum[d], data_vec);
            }
            count += _mm256_movemask_ps(mask);
        }
        // 处理最后几个不足 8 个的数据
        for (int i = n - (n % 8); i < n; i++) {
            if (idx[i] == j) {
                for (int d = 0; d < m; d++) {
                    sum[d] = _mm256_add_ps(sum[d], _mm256_set1_ps(data[i].dimensions[d]));
                }
                count++;
            }
        }
        if (count > 0) {
            for (int d = 0; d < m; d++){
                __m256 center = _mm256_div_ps(sum[d], _mm256_set1_ps(static_cast<float>(count)));
                _mm256_storeu_ps(&centers[j].dimensions[d], center);
            }
        }
    }
}

void MiniBatchKMeans(int k, std::vector<Node>& data, int n, int m, int batchSize) {
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

    int num_batches = n / batchSize;
    bool cluster_changed = true;
    while (cluster_changed) {
        cluster_changed = false;

        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            int start = batch_idx * batchSize;
            int end = min((batch_idx + 1) * batchSize, n);
            int batch_size = end - start;

            // 计算小批量数据点到簇中心的距离
            for (int i = start; i < end; i++) {
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
            VectorizedUpdateClusterCenter(data, idx, centers, k, batch_size, m);
        }
    }

    // 输出聚类结果
    if (!cluster_changed) {
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
#include <windows.h>
#include <sysinfoapi.h>
int main() {
    int n = 1000, m = 10, k = 5, batchSize = 32;
    std::vector<Node> data;

    generateStructuredData(data, n, m);

    FILETIME start_time, end_time;
    ULARGE_INTEGER start_time_us, end_time_us;

    GetSystemTimePreciseAsFileTime(&start_time);
    MiniBatchKMeans(k, data, n, m, batchSize);
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