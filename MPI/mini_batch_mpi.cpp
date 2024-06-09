#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
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

void Mini_Batch_Kmeans(long long k, vector<node>& data, long long n, long long m, long long batch_size) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    // 2. 广播初始簇中心给所有进程
    for (long long i = 0; i < k; ++i) {
        C[i].dimen.resize(m); // 确保每个簇中心的维度向量已正确分配内存
        MPI_Bcast(C[i].dimen.data(), m, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    // 3. 迭代聚类过程，直到簇中心不再变化
    bool cluster_changed = true;
    int iteration = 1; // 迭代次数
    while (cluster_changed) {
        cluster_changed = false;
        cout << "Iteration " << iteration << ":" << endl;
        iteration++;
        // 3.1 随机选择一个小批量的数据进行处理
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, n - 1);
        vector<long long> batch_idx(batch_size);
        for (long long i = 0; i < batch_size; ++i) {
            batch_idx[i] = dis(gen);
        }

        // 3.2 计算小批量数据点到簇中心的距离,并重新分配簇
        for (long long i = 0; i < batch_size; ++i) {
            float min_distance = numeric_limits<float>::max();
            long long min_index = -1;
            for (long long j = 0; j < k; ++j) {
                float distance = Distance(data[batch_idx[i]], C[j], m);
                D[batch_idx[i]][j] = distance;
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = j;
                }
            }
            if (idx[batch_idx[i]] != min_index) {
                idx[batch_idx[i]] = min_index;
                cluster_changed = true;
            }
        }

        // 3.3 更新簇中心
        for (long long j = 0; j < k; ++j) {
            node* sum_cluster = new node;
            sum_cluster->dimen.resize(m); // 设置 dimen 向量的大小为 m
            long long count_cluster = 0;
            for (long long i = 0; i < batch_size; ++i) {
                if (idx[batch_idx[i]] == j) {
                    Add(*sum_cluster, data[batch_idx[i]], m);
                    ++count_cluster;
                }
            }
            if (count_cluster > 0) {
                for (int i = 0; i < m; ++i) {
                    C[j].dimen[i] = sum_cluster->dimen[i] / count_cluster;
                }
            }
            delete sum_cluster;
        }

        // 3.4 同步 cluster_changed 变量
        int local_cluster_changed = cluster_changed ? 1 : 0;
        int global_cluster_changed = 0;
        MPI_Allreduce(&local_cluster_changed, &global_cluster_changed, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        cluster_changed = global_cluster_changed;

    }

    // 输出聚类结果
    for (long long i = 0; i < k; ++i) {
        if (rank == 0) {
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

    long long n = 100000, m = 10, k = 5, batch_size = 100;
    vector<node> data;

    generateStructuredData(data, n, m);

    auto start_time = chrono::high_resolution_clock::now();

    Mini_Batch_Kmeans(k, data, n, m, batch_size);

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