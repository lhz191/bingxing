#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>

using namespace std;

struct node {
    vector<float> dimen;
};

void generateStructuredData(vector<node>& data, long long n, long long m) {
    data.resize(n);
    for (long long i = 0; i < n; i++) {
        data[i].dimen.resize(m);
        for (long long j = 0; j < m; j++) {
            data[i].dimen[j] = static_cast<float>(i + 1);
        }
    }
}

float Distance(const node& X, const node& Z, long long n) {
    float result = 0;
    for (long long i = 0; i < n; i++) {
        result += (X.dimen[i] - Z.dimen[i]) * (X.dimen[i] - Z.dimen[i]);
    }
    return sqrt(result);
}

void Add(node& result, const node& X, long long n) {
    for (long long i = 0; i < n; i++) {
        result.dimen[i] += X.dimen[i];
    }
}

void Kmeans(long long k, vector<node>& data, long long n, long long m) {
    vector<node> C(k); // 存储簇中心
    vector<long int> idx(n);
    vector<vector<float>> D(n, vector<float>(k)); // 存储样本点到簇中心的距离

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
    while (cluster_changed) {
        cluster_changed = false;
        // 2.1 计算每个样本点到簇中心的距离，并重新分配簇
        for (long long i = 0; i < n; ++i) {
            float min_distance = numeric_limits<float>::max();
            long long min_index = -1;
            for (long long j = 0; j < k; ++j) {
                float distance = Distance(data[i], C[j], m);
                D[i][j] = distance;
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
        for (long long j = 0; j < k; ++j) {
            node* sum_cluster = new node;
            sum_cluster->dimen.resize(m); // 设置 dimen 向量的大小为 m
            long long count_cluster = 0;
            for (long long i = 0; i < n; ++i) {
                if (idx[i] == j) {
                    Add(*sum_cluster, data[i], m);
                    ++count_cluster;
                }
            }
            if (count_cluster > 0) {
                for (long long i = 0; i < m; ++i) {
                    C[j].dimen[i] = sum_cluster->dimen[i] / count_cluster;
                }
            }
        }
    }
    if (!cluster_changed) {
        // 输出聚类结果
        for (long long i = 0; i < k; ++i) {
            cout << "第 " << i + 1 << " 个簇的中心点：";
            for (long long j = 0; j < m; ++j) {
                cout << C[i].dimen[j] << " ";
            }
            cout << endl;
        }
    }
}

int main()
{
    long long n = 100000, m = 10, k = 5;
    vector<node> data;

    generateStructuredData(data, n, m);

    struct timespec start_time, end_time;
    long long elapsed_time_ns;

    // 获取开始时间
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);

    Kmeans(k, data, n, m);

    // 获取结束时间
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);

    // 计算执行时间(纳秒)
    elapsed_time_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000 + (end_time.tv_nsec - start_time.tv_nsec);

    // 转换成秒
    double elapsed_time = static_cast<double>(elapsed_time_ns) / 1000000000.0;

    printf("Elapsed time: %.9f seconds\n", elapsed_time);

    return 0;
}