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
#include <semaphore.h>
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

sem_t sem_main, sem_worker;

typedef struct {
    vector<node>* data;
    vector<node>* C;
    vector<long int>* idx;
    long long k;
    long long n;
    long long m;
    bool* cluster_changed;
} ThreadArgs;

void* worker_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    vector<node>* data = args->data;
    vector<node>* C = args->C;
    vector<long int>* idx = args->idx;
    long long n = args->n;
    long long m = args->m;
    long long k = args->k;
    bool* cluster_changed = args->cluster_changed;

    while (true) {
        sem_wait(&sem_worker);

        // 2.1 计算每个样本点到簇中心的距离，并重新分配簇
        for (long long i = 0; i < n; i++) {
            float min_distance = numeric_limits<float>::max();
            long long min_index = -1;
            for (long long j = 0; j < k; j++) {
                float distance = Distance((*data)[i], (*C)[j], m);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = j;
                }
            }
            if ((*idx)[i] != min_index) {
                (*idx)[i] = min_index;
                *cluster_changed = true;
            }
        }

        sem_post(&sem_main);
        sem_wait(&sem_worker);

        // 2.2 更新簇中心
        for (long long i = 0; i < k; i++) {
            node new_center;
            new_center.dimen.resize(m, 0.0f);
            long long count = 0;
            for (long long j = 0; j < n; j++) {
                if ((*idx)[j] == i) {
                    Add(new_center, (*data)[j], m);
                    count++;
                }
            }
            for (long long m_index = 0; m_index < m; m_index++) {
                (*C)[i].dimen[m_index] = new_center.dimen[m_index] / count;
            }
        }

        sem_post(&sem_main);
    }

    pthread_exit(NULL);
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
    int iteration = 1; // 迭代次数

    sem_init(&sem_main, 0, 0);
    sem_init(&sem_worker, 0, NUM_THREADS);

    pthread_t worker_threads[NUM_THREADS];
    ThreadArgs worker_args[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        worker_args[i].data = &data;
        worker_args[i].C = &C;
        worker_args[i].idx = &idx;
        worker_args[i].k = k;
        worker_args[i].n = n;
        worker_args[i].m = m;
        worker_args[i].cluster_changed = &cluster_changed;
        pthread_create(&worker_threads[i], NULL, worker_thread, &worker_args[i]);
    }

    while (cluster_changed) {
        cluster_changed = false;
        cout << "Iteration " << iteration << ":" << endl;

        // 2.1 计算每个样本点到簇中心的距离，并重新分配簇
        for (int i = 0; i < NUM_THREADS; i++) {
            sem_post(&sem_worker);
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            sem_wait(&sem_main);
        }

        // 2.2 更新簇中心
        for (int i = 0; i < NUM_THREADS; i++) {
            sem_post(&sem_worker);
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            sem_wait(&sem_main);
        }

        iteration++; // 更新迭代次数
    }

    cout << "Cluster centers after iteration " << iteration << ":" << endl;

    // 输出聚类结果
    for (long long i = 0; i < k; ++i) {
        cout << "第 " << i + 1 << " 个簇的中心点：";
        for (int j = 0; j < m; ++j) {
            cout << C[i].dimen[j] << " ";
        }
        cout << endl;
    }

    // 销毁线程
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_cancel(worker_threads[i]);
    }
    sem_destroy(&sem_main);
    sem_destroy(&sem_worker);
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