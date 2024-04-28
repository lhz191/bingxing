#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <arm_neon.h>

struct Node {
    std::vector<float> dimensions;
};

// Neon version of the distance calculation function
inline float VectorizedDistance(const Node& x, const Node& z, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (int i = 0; i < n; i += 4) {
        float32x4_t x_vec = vld1q_f32(&x.dimensions[i]);
        float32x4_t z_vec = vld1q_f32(&z.dimensions[i]);
        float32x4_t diff = vsubq_f32(x_vec, z_vec);
        float32x4_t squared = vmulq_f32(diff, diff);
        sum = vaddq_f32(sum, squared);
    }

    // Horizontal reduction of the sum vector
    float32x2_t pairwise_sum = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    float total_sum = vget_lane_f32(pairwise_sum, 0) + vget_lane_f32(pairwise_sum, 1);
    return sqrtf(total_sum);
}
// Neon version of the cluster center update function
void VectorizedUpdateClusterCenter(const std::vector<Node>& data, const std::vector<int>& idx,
                                  std::vector<Node>& centers, int k, int n, int m) {
    for (int j = 0; j < k; j++) {
        float32x4_t sum[m];
        for (int d = 0; d < m; d++) {
            sum[d] = vdupq_n_f32(0.0f);
        }
        int count = 0;
        for (int i = 0; i < n - (n % 4); i += 4) {
            uint32x4_t mask = vceqq_u32(vld1q_u32(reinterpret_cast<const uint32_t*>(&idx[i])),
                                        vdupq_n_u32(static_cast<uint32_t>(j)));
            for (int d = 0; d < m; d++) {
                float32x4_t data_vec = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(&data[i].dimensions[d])), mask));
                sum[d] = vaddq_f32(sum[d], data_vec);
            }
            count += vaddvq_u32(mask);
        }
        // Handle the last few data points that are less than 4
        for (int i = n - (n % 4); i < n; i++) {
            if (idx[i] == j) {
                for (int d = 0; d < m; d++) {
                    sum[d] = vaddq_f32(sum[d], vdupq_n_f32(data[i].dimensions[d]));
                }
                count++;
            }
        }
        if (count > 0) {
            for (int d = 0; d < m; d++) {
                float32x4_t center = vdivq_f32(sum[d], vdupq_n_f32(static_cast<float>(count)));
                vst1q_f32(&centers[j].dimensions[d], center);
            }
        }
    }
}

void KMeans(int k, std::vector<Node>& data, int n, int m) {
    std::vector<Node> centers(k);
    std::vector<int> idx(n);
    std::vector<std::vector<float>> distances(n, std::vector<float>(k));

    // Initialize cluster centers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);
    for (int i = 0; i < k; i++) {
        centers[i] = data[dis(gen)];
    }

    bool cluster_changed = true;
    while (cluster_changed) {
        cluster_changed = false;

        // Calculate the distance between each data point and cluster centers
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

        // Update cluster centers
        VectorizedUpdateClusterCenter(data, idx, centers, k, n, m);
    }

    // Output the clustering results
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

int main() {
    int n = 100000, m = 10, k = 5;
    std::vector<Node> data;

    generateStructuredData(data, n, m);

    auto start_time = std::chrono::high_resolution_clock::now();
    KMeans(k, data, n, m);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

    std::cout << "K-Means algorithm execution time: " << elapsed_time.count() << " s" << std::endl;

    return 0;
}