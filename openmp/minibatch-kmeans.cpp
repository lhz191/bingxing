#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <omp.h>
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
#pragma omp parallel for for num_threads(16) reduction(+:result) schedule(guided,256)
    for (long long i = 0; i < n; i++) {
        result += (X.dimen[i] - Z.dimen[i]) * (X.dimen[i] - Z.dimen[i]);
    }
    return sqrt(result);
}

void Add(node& result, const node& X, long long n) {
#pragma omp parallel for num_threads(16) schedule(guided,256) reduction(+:result.dimen[:n])
    for (long long i = 0; i < n; i++) {
        result.dimen[i] += X.dimen[i];
    }
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

    while (cluster_changed) {
        cluster_changed = false;
        cout << "Iteration " << iteration << ":" << endl;
        iteration++;

        vector<long long> batch_idx(batch_size);
        for (long long i = 0; i < batch_size; ++i) {
            batch_idx[i] = dis(gen);
        }

#pragma omp parallel for num_threads(16) schedule(guided, 256)
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
#pragma omp critical
            {
                if (idx[batch_idx[i]] != min_index) {
                    idx[batch_idx[i]] = min_index;
                    cluster_changed = true;
                }
            }
        }

        vector<node> new_C(k, node{ vector<float>(m, 0.0f) });
        vector<long long> counts(k, 0);

#pragma omp parallel for num_threads(16) schedule(guided, 256) reduction(+:new_C[:k], counts[:k])
        for (long long i = 0; i < batch_size; ++i) {
            int cluster = idx[batch_idx[i]];
            Add(new_C[cluster], data[batch_idx[i]], m);
            counts[cluster]++;
        }

#pragma omp parallel for num_threads(16) schedule(guided, 256)
        for (long long j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (long long i = 0; i < m; ++i) {
                    C[j].dimen[i] = new_C[j].dimen[i] / counts[j];
                }
            }
        }
    }

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
int main() {
    long long n = 1000, m = 10, k = 5, batch_size = 100;
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