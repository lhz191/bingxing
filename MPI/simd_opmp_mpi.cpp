#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include <immintrin.h>

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

float Distance(const node& X, const node& Z, long long m) {
    float result = 0;
#pragma omp simd reduction(+:result)
    for (long long i = 0; i < m; i++) {
        float diff = X.dimen[i] - Z.dimen[i];
        result += diff * diff;
    }
    return sqrt(result);
}

void Add(node& result, const node& X, long long m) {
#pragma omp simd
    for (long long i = 0; i < m; i++) {
        result.dimen[i] += X.dimen[i];
    }
}

void Kmeans(long long k, vector<node>& data, long long n, long long m) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<node> C(k); // 存储簇中心
    for (long long i = 0; i < k; ++i) {
        C[i].dimen.resize(m); // 确保每个簇中心的维度向量已正确分配内存
    }
    vector<long int> idx(n); // 每个进程负责 n/size 个数据样本
    vector<vector<float>> D(n, vector<float>(k)); // 存储样本点到簇中心的距离

    // 1. 在 0 号进程初始化聚类中心
    if (rank == 0) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, n - 1);
        for (long long i = 0; i < k; ++i) {
            int idx_init = dis(gen);
            C[i] = data[idx_init];
        }
    }

    // 2. 广播聚类中心到其他进程
    for (long long i = 0; i < k; ++i) {
        MPI_Bcast(C[i].dimen.data(), m, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // 3. 迭代聚类过程，直到簇中心不再变化
    bool cluster_changed = true;
    int iteration = 1; // 迭代次数
    while (cluster_changed) {
        cluster_changed = false;

        // 动态负载均衡
        long long chunk_size = 100; // 每次处理的块大小
        long long total_chunks = (n + chunk_size - 1) / chunk_size;

#pragma omp parallel for schedule(guided) num_threads(4)
        for (long long chunk = rank; chunk < total_chunks; chunk += size) {
            long long start = chunk * chunk_size;
            long long end = min(start + chunk_size, n);
            for (long long i = start; i < end; ++i) {
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
        }

        // 3.2 更新簇中心
        vector<node> local_centers(k);
        vector<long long> local_counts(k, 0);