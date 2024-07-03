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
#include <thread>
#include <semaphore.h>
using namespace std;

const int NUM_THREADS = 4;

struct node {
    vector<float> dimen;
};

struct threadParam_t {
    vector<node>* data;
    vector<node>* C;
    vector<long int>* idx;
    long long start;
    long long k;
    long long m;
};

struct updateParam_t {
    vector<node>* data;
    vector<node>* C;
    vector<long int>* idx;
    long long j;
    long long start;
    long long m;
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

void worker_thread(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    float min_distance = numeric_limits<float>::max();
    long long min_index = -1;
    for (long long j = 0; j < p->k; ++j) {
        float distance = Distance((*p->data)[p->start], (*p->C)[j], p->m);
        if (distance < min_distance) {
            min_distance = distance;
            min_index = j;
        }
    }
    (*p->idx)[p->start] = min_index;
}

void update_thread(void* param) {
    updateParam_t* p = (updateParam_t*)param;
    node new_center = {};
    new_center.dimen.resize(p->m);
    long long count = 0;
    for (long long i = p->start; i < (*p->data).size(); i += NUM_THREADS) {
        if ((*p->idx)[i] == p->j) {
            Add(new_center, (*p->data)[i], p->m);
            count++;
        }
    }
    if (count > 0) {
        for (long long i = 0; i < p->m; i++) {
            (*p->C)[p->j].dimen[i] = new_center.dimen[i] / count;
        }
    }
}

void Kmeans(long long k, vector<node>& data, long long n, long long m) {
    vector<node> C(k); // 存储簇中心
    vector<long int> idx(n);

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
        vector<pthread_t> worker_threads(NUM_THREADS);
        vector<threadParam_t> params(NUM_THREADS);
        for (long long i = 0; i < n; i += NUM_THREADS) {
            // 创建线程并执行计算任务
            for (int t_id = 0; t_id < NUM_THREADS && i + t_id < n; ++t_id) {
                params[t_id].data = &data;
                params[t_id].C = &C;
                params[t_id].idx = &idx;
                params[t_id].start = i + t_id;
                params[t_id].k = k;
                params[t_id].m = m;
                pthread_create(&worker_threads[t_id], NULL, (void*(*)(void*))worker_thread, &params[t_id]);
            }

            // 等待线程完成
            for (int t_id = 0; t_id < NUM_THREADS && i + t_id < n; ++t_id) {
                pthread_join(worker_threads[t_id], NULL);
            }
        }

        // 2.2 更新簇中心
        for (long long j = 0; j < k; ++j) {
            vector<pthread_t> update_threads(NUM_THREADS);
            vector<updateParam_t> update_params(NUM_THREADS);
            for (long long i = 0; i < n; i += NUM_THREADS) {
                // 创建线程并执行更新簇中心的任务
                for (int t_id = 0; t_id < NUM_THREADS && i + t_id < n; ++t_id) {
                    update_params[t_id].data = &data;
                    update_params[t_id].C = &C;
                    update_params[t_id].idx = &idx;
                    update_params[t_id].j = j;
                    update_params[t_id].start = i + t_id;
                    update_params[t_id].m = m;
                    pthread_create(&update_threads[t_id], NULL, (void*(*)(void*))update_thread, &update_params[t_id]);
                }

                // 等待线程完成
                for (int t_id = 0; t_id < NUM_THREADS && i + t_id < n; ++t_id) {
                    pthread_join(update_threads[t_id], NULL);
                }
            }
        }
        cout<<1<<endl;

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
    long long n = 100000, m = 10, k = 5;
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