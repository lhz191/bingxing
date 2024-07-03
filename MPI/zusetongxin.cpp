#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <mpi.h>

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
    for (long long i = 0; i < m; i++) {
        result += (X.dimen[i] - Z.dimen[i]) * (X.dimen[i] - Z.dimen[i]);
    }
    return sqrt(result);
}

void Add(node& result, const node& X, long long m) {
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
    vector<long int> idx(n / size); // 每个进程负责 n/size 个数据样本
    vector<vector<float>> D(n / size, vector<float>(k)); // 存储样本点到簇中心的距离

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
        // 3.1 计算每个样本点到簇中心的距离，并重新分配簇
        for (long long i = 0; i < n / size; ++i) {
            float min_distance = numeric_limits<float>::max();
            long long min_index = -1;
            for (long long j = 0; j < k; ++j) {
                float distance = Distance(data[i + rank * (n / size)], C[j], m);
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

        // 3.2 更新簇中心
        vector<node> local_centers(k);
        vector<long long> local_counts(k, 0);
        for (long long j = 0; j < k; ++j) {
            local_centers[j].dimen.resize(m, 0.0f);
        }

        for (long long i = 0; i < n / size; ++i) {
            Add(local_centers[idx[i]], data[i + rank * (n / size)], m);
            local_counts[idx[i]]++;
        }

        vector<node> global_centers(k);
        vector<long long> global_counts(k, 0);
        for (long long j = 0; j < k; ++j) {
            global_centers[j].dimen.resize(m, 0.0f);
        }

        for (long long j = 0; j < k; ++j) {
            MPI_Allreduce(local_centers[j].dimen.data(), global_centers[j].dimen.data(), m, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_counts[j], &global_counts[j], 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        }

        for (long long j = 0; j < k; ++j) {
            if (global_counts[j] > 0) {
                for (long long l = 0; l < m; ++l) {
                    C[j].dimen[l] = global_centers[j].dimen[l] / global_counts[j];
                }
            }
        }

        // 所有进程同步 cluster_changed 变量
        int local_cluster_changed = cluster_changed ? 1 : 0;
        int global_cluster_changed = 0;
        MPI_Allreduce(&local_cluster_changed, &global_cluster_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        cluster_changed = global_cluster_changed;

        if (rank == 0) {
            cout << "Iteration " << iteration << " completed." << endl;
        }
        iteration++;
    }

    // 4. 在 0 号进程输出聚类结果
    if (rank == 0) {
        for (long long i = 0; i < k; ++i) {
            cout << "第 " << i + 1 << " 个簇的中心点：";
            for (int j = 0; j < m; ++j) {
                cout << C[i].dimen[j] << " ";
            }
            cout << endl;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    long long n = 100000, m = 10, k = 5;
    vector<node> data;

    generateStructuredData(data, n, m);

    auto start_time = chrono::high_resolution_clock::now();
    Kmeans(k, data, n, m);
    auto end_time = chrono::high_resolution_clock::now();
    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        cout << "Kmeans 算法执行时间: " << elapsed_time.count() << " 毫秒" << endl;
    }

    MPI_Finalize();
    return 0;
}