#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <chrono>
#include <omp.h>

using namespace std;

struct node {
    vector<float> dimen;
};

void generateStructuredData(vector<node>& data, int n, int m) {
    data.resize(n);
    for (int i = 0; i < n; i++) {
        data[i].dimen.resize(m);
        for (int j = 0; j < m; j++) {
            data[i].dimen[j] = static_cast<float>(i + 1);
        }
    }
}

float Distance(const node& X, const node& Z, long long n) {
    float result = 0;
    #pragma omp target teams distribute parallel for reduction(+:result) map(to: X.dimen[0:n], Z.dimen[0:n])
    for (long long i = 0; i < n; i++) {
        result += (X.dimen[i] - Z.dimen[i]) * (X.dimen[i] - Z.dimen[i]);
    }
    return sqrt(result);
}

void Kmeans(long long k, vector<node>& data, long long n, long long m) {
    vector<node> C(k); // 存储簇中心
    vector<long int> idx(n, -1);
    vector<float> D(((n * k + 63) / 64) * 64, 0.0f); // 填充到 64 字节对齐

    // 1. 从数据中随机选择k个样本作为初始簇中心
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n - 1);
    for (long long i = 0; i < k; ++i) {
        int idx_init = dis(gen);
        C[i] = data[idx_init];
    }

    // 将簇中心和数据传输到加速器设备
    #pragma omp target enter data map(to: data[0:n]) map(to: C[0:k])

    // 2. 迭代聚类过程，直到簇中心不再变化
    bool cluster_changed = true;
    int iteration = 1; // 迭代次数

    while (cluster_changed) {
        cluster_changed = false;
        cout << "Iteration " << iteration << ":" << endl;

        // 2.1 计算每个样本点到簇中心的距离，并重新分配簇
        #pragma omp target teams distribute parallel for map(tofrom: cluster_changed) map(to: D[0:n*k], idx[0:n])
        for (long long i = 0; i < n; i++) {
            float min_distance = numeric_limits<float>::max();
            long long min_index = -1;
            for (long long j = 0; j < k; j++) {
                float distance = Distance(data[i], C[j], m);
                D[i * k + j] = distance;
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

        // 2.2 更新簇中心
        vector<float> new_centers(((k * m + 63) / 64) * 64, 0.0f); // 填充到 64 字节对齐
        vector<long long> counts(((k + 7) / 8) * 8, 0); // 填充到 64 字节对齐

        #pragma omp target teams distribute parallel for reduction(+:new_centers[:k*m], counts[:k]) map(to: data[0:n], idx[0:n]) map(tofrom: new_centers[0:k*m], counts[0:k])
        for (long long i = 0; i < n; i++) {
            int cluster = idx[i];
            for (long long j = 0; j < m; j++) {
                new_centers[cluster * m + j] += data[i].dimen[j];
            }
            counts[cluster]++;
        }

        #pragma omp target teams distribute parallel for map(to: counts[0:k]) map(tofrom: C[0:k])
        for (long long i = 0; i < k; i++) {
            if (counts[i] > 0) {
                for (long long j = 0; j < m; j++) {
                    C[i].dimen[j] = new_centers[i * m + j] / counts[i];
                }
            }
        }

        iteration++; // 更新迭代次数
    }

    // 将簇中心从加速器设备传回主机
    #pragma omp target exit data map(from: C[0:k])

    // 输出聚类结果
    for (long long i = 0; i < k; ++i) {
        cout << "第 " << i + 1 << " 个簇的中心点：";
        for (int j = 0; j < m; ++j) {
            cout << C[i].dimen[j] << " ";
        }
        cout << endl;
    }
}

int main() {
    long long n = 100000, m = 10, k = 5;
    vector<node> data;

    generateStructuredData(data, n, m);

    auto start_time = chrono::high_resolution_clock::now();

    Kmeans(k, data, n, m);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    return 0;
}
