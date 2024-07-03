#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

using namespace std;

#define NUM_THREADS 4

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

typedef struct {
    vector<node>* C;
    vector<long int>* idx;
    vector<node>* data;
    long long k;
    long long n;
    long long m;
    long long start;
    long long end;
} ThreadArgsForUpdate;

void* update_cluster_centers(void* arg) {
    ThreadArgsForUpdate* args = (ThreadArgsForUpdate*)arg;
    vector<node>* C = args->C;
    vector<long int>* idx = args->idx;
    vector<node>* data = args->data;
    long long k = args->k;
    long long n = args->n;
    long long m = args->m;
    long long start = args->start;
    long long end = args->end;

    for (long long j = start; j < end; ++j) {
        node* sum_cluster = new node;
        sum_cluster->dimen.resize(m); // 设置 dimen 向量的大小为 m
        long long count_cluster = 0;
        for (long long i = 0; i < n; ++i) {
            if ((*idx)[i] == j) {
                Add(*sum_cluster, (*data)[i], m);
                ++count_cluster;
            }
        }
        if (count_cluster > 0) {
            for (int i = 0; i < m; ++i) {
                (*C)[j].dimen[i] = sum_cluster->dimen[i] / count_cluster;
            }
        }
        delete sum_cluster;
    }

    pthread_exit(NULL);
}

// 在全局范围内创建工作线程
pthread_t threads[NUM_THREADS];
ThreadArgsForUpdate args[NUM_THREADS];

void initThreads() {
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, update_cluster_centers, &args[i]);
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
        long long block_size = k / NUM_THREADS;
        for (int i = 0; i < NUM_THREADS; i++) {
            args[i].C = &C;
            args[i].idx = &idx;
            args[i].data = &data;
            args[i].k = k;
            args[i].n = n;
            args[i].m = m;
            args[i].start = i * block_size;
            args[i].end = (i == NUM_THREADS - 1) ? k : (i + 1) * block_size;
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    if (cluster_changed == false) {
        // 输出聚类结果
        for (long long i = 0; i < k; ++i) {
            cout << "第 " << i + 1 << " 个簇的中心点：";
            for (int j = 0; j < m; ++j) {
                cout << C[i].dimen[j] << " ";
            }
            cout << endl;
        }
    }
}

#include <stdio.h>
#include <windows.h>
#include <sysinfoapi.h>
int main()
{
    long long n = 1000, m = 10, k = 5;
    vector<node> data;

    generateStructuredData(data, n, m);

        // 输出 data 容器中的数据
    // std::cout << "Generated data:\n";
    // for (const auto& node : data) {
    //     std::cout << "Node: ";
    //     for (float dim : node.dimen) {
    //         std::cout << dim << " ";
    //     }
    //     std::cout << "\n";
    // }
    FILETIME start_time, end_time;
    ULARGE_INTEGER start_time_us, end_time_us;
    // auto start_time = chrono::high_resolution_clock::now(); // 记录开始时间
    // for (int i = 1; i <= 100; i++) {
        // Kmeans(k, data, n, m);
    // }
    // auto end_time = chrono::high_resolution_clock::now(); // 记录结束时间
    // auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time); // 计算经过的时间
    // cout << "Kmeans 算法执行时间: " << elapsed_time.count() << " 毫秒" << endl;
        // 获取开始时间
    GetSystemTimePreciseAsFileTime(&start_time);
    initThreads(); // 在程序开始时创建所有工作线程
      // 等待所有工作线程结束
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    Kmeans(k, data, n, m);
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