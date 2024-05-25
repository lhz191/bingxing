#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
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
    #pragma omp parallel for for num_threads(4) reduction(+:result) schedule(guided,256)
    for (long long i = 0; i < n; i++) {
        result += (X.dimen[i] - Z.dimen[i]) * (X.dimen[i] - Z.dimen[i]);
    }
    return sqrt(result);
}

void Add(node& result, const node& X, long long n) {
    #pragma omp parallel for num_threads(4) schedule(guided,256) reduction(+:result.dimen[:n])
    for (long long i = 0; i < n; i++) {
        result.dimen[i] += X.dimen[i];
    }
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

    // 2. 迭代聚类过程，直到簇中心不再变化
    bool cluster_changed = true;
    int iteration = 1; // 迭代次数

    while (cluster_changed) {
        cluster_changed = false;
        cout << "Iteration " << iteration << ":" << endl;

        // 2.1 计算每个样本点到簇中心的距离，并重新分配簇
        #pragma omp parallel for num_threads(4) schedule(guided,256)
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

       #pragma omp parallel for num_threads(4) schedule(guided,256)\
        reduction(+:new_centers[:k*m], counts[:k])
        for (long long i = 0; i < n; i++) {
            int cluster = idx[i];
            for (long long j = 0; j < m; j++) {
                new_centers[cluster * m + j] += data[i].dimen[j];
            }
            counts[cluster]++;
        }

        #pragma omp parallel for num_threads(4) schedule(guided,256)
        for (long long i = 0; i < k; i++) {
            if (counts[i] > 0) {
                for (long long j = 0; j < m; j++) {
                    C[i].dimen[j] = new_centers[i * m + j] / counts[i];
                }
            }
        }

        iteration++; // 更新迭代次数
    }

    // 输出聚类结果
    for (long long i = 0; i < k; ++i) {
        cout << "第 " << i + 1 << " 个簇的中心点：";
        for (int j = 0; j < m; ++j) {
            cout << C[i].dimen[j] << " ";
        }
        cout << endl;
    }
}
#include <stdio.h>
#include <windows.h>
#include <sysinfoapi.h>
int main()
{
    long long n =100000, m = 10, k = 5;
    vector<node> data;

    generateStructuredData(data, n, m);
    FILETIME start_time, end_time;
    ULARGE_INTEGER start_time_us, end_time_us;
    GetSystemTimePreciseAsFileTime(&start_time);

    Kmeans(k, data, n, m);
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
