#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <pthread.h>
#include <semaphore.h>
#include <limits>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <windows.h>
#include <sysinfoapi.h>

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
    vector<node>* data;
    vector<node>* C;
    vector<long int>* idx;
    vector<vector<float>>* D;
    vector<long long>* batch_idx;
    long long k;
    long long m;
    long long start;
    long long end;
    bool* cluster_changed;
    sem_t* sem_main;
    sem_t* sem_worker;
} ThreadArgs;

void* worker_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    vector<node>* data = args->data;
    vector<node>* C = args->C;
    vector<long int>* idx = args->idx;
    vector<vector<float>>* D = args->D;
    vector<long long>* batch_idx = args->batch_idx;
    long long k = args->k;
    long long m = args->m;
    long long start = args->start;
    long long end = args->end;
    bool* cluster_changed = args->cluster_changed;
    sem_t* sem_main = args->sem_main;
    sem_t* sem_worker = args->sem_worker;

    while (true) {
        sem_wait(sem_worker);

        for (long long i = start; i < end; ++i) {
            float min_distance = numeric_limits<float>::max();
            long long min_index = -1;
            for (long long j = 0; j < k; ++j) {
                float distance = Distance((*data)[(*batch_idx)[i]], (*C)[j], m);
                (*D)[(*batch_idx)[i]][j] = distance;
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = j;
                }
            }
            if ((*idx)[(*batch_idx)[i]] != min_index) {
                (*idx)[(*batch_idx)[i]] = min_index;
                *cluster_changed = true;
            }
        }

        sem_post(sem_main);
    }

    pthread_exit(NULL);
}

void Mini_Batch_Kmeans(long long k, vector<node>& data, long long n, long long m, long long batch_size) {
    vector<node> C(k);
    vector<long int> idx(n, -1);
    vector<vector<float>> D(n, vector<float>(k));

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n - 1);
    for (long long i = 0; i < k; ++i) {
        C[i] = data[dis(gen)];
    }

    bool cluster_changed = true;
    int iteration = 1;

    sem_t sem_main, sem_worker;
    sem_init(&sem_main, 0, 0);
    sem_init(&sem_worker, 0, NUM_THREADS);

    pthread_t worker_threads[NUM_THREADS];
    ThreadArgs worker_args[NUM_THREADS];

    while (cluster_changed) {
        cluster_changed = false;
        cout << "Iteration " << iteration << ":" << endl;
        iteration++;

        vector<long long> batch_idx(batch_size);
        for (long long i = 0; i < batch_size; ++i) {
            batch_idx[i] = dis(gen);
        }

        long long chunk_size = batch_size / NUM_THREADS;
        for (int i = 0; i < NUM_THREADS; ++i) {
            worker_args[i].data = &data;
            worker_args[i].C = &C;
            worker_args[i].idx = &idx;
            worker_args[i].D = &D;
            worker_args[i].batch_idx = &batch_idx;
            worker_args[i].k = k;
            worker_args[i].m = m;
            worker_args[i].start = i * chunk_size;
            worker_args[i].end = (i == NUM_THREADS - 1) ? batch_size : (i + 1) * chunk_size;
            worker_args[i].cluster_changed = &cluster_changed;
            worker_args[i].sem_main = &sem_main;
            worker_args[i].sem_worker = &sem_worker;
            pthread_create(&worker_threads[i], NULL, worker_thread, &worker_args[i]);
        }

        for (int i = 0; i < NUM_THREADS; ++i) {
            sem_post(&sem_worker);
        }

        for (int i = 0; i < NUM_THREADS; ++i) {
            sem_wait(&sem_main);
        }

        for (long long j = 0; j < k; ++j) {
            node sum_cluster;
            sum_cluster.dimen.resize(m, 0.0f);
            long long count_cluster = 0;
            for (long long i = 0; i < batch_size; ++i) {
                if (idx[batch_idx[i]] == j) {
                    Add(sum_cluster, data[batch_idx[i]], m);
                    count_cluster++;
                }
            }
            if (count_cluster > 0) {
                for (int i = 0; i < m; ++i) {
                    C[j].dimen[i] = sum_cluster.dimen[i] / count_cluster;
                }
            }
        }
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_cancel(worker_threads[i]);
    }
    sem_destroy(&sem_main);
    sem_destroy(&sem_worker);

    for (long long i = 0; i < k; ++i) {
        cout << "第 " << i + 1 << " 个簇的中心点：";
        for (int j = 0; j < m; ++j) {
            cout << C[i].dimen[j] << " ";
        }
        cout << endl;
    }
}

int main() {
    long long n = 10000, m = 10, k = 5, batch_size = 50;
    vector<node> data;

    generateStructuredData(data, n, m);

    FILETIME start_time, end_time;
    ULARGE_INTEGER start_time_us, end_time_us;

    GetSystemTimePreciseAsFileTime(&start_time);

    Mini_Batch_Kmeans(k, data, n, m, batch_size);

    GetSystemTimePreciseAsFileTime(&end_time);

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