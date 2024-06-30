#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <limits>
#include <windows.h>
#include <sysinfoapi.h>

using namespace std;

struct node {
    vector<float> dimen;
};

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

void readData(const string& filename, vector<node>& data, long long& n, long long& m) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error opening file!" << endl;
        exit(1);
    }

    string line;
    getline(infile, line); // Skip header line

    while (getline(infile, line)) {
        stringstream ss(line);
        string id;
        node temp;
        getline(ss, id, ','); // Skip ID
        string value;
        while (getline(ss, value, ',')) {
            temp.dimen.push_back(stof(value));
        }
        data.push_back(temp);
    }

    n = data.size();
    if (!data.empty()) {
        m = data[0].dimen.size();
    } else {
        m = 0;
    }
}

void Mini_Batch_Kmeans(long long k, vector<node>& data, long long n, long long m, long long batch_size, int max_iterations) {
    FILETIME start_time, end_time;
    ULARGE_INTEGER start_time_us, end_time_us;

    GetSystemTimePreciseAsFileTime(&start_time);

    // 1. 从数据中随机选择k个样本作为初始簇中心
    vector<node> C(k);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n - 1);
    for (long long i = 0; i < k; ++i) {
        int idx_init = dis(gen);
        C[i] = data[idx_init];
    }

    vector<long int> idx(n, -1);
    vector<vector<float>> D(batch_size, vector<float>(k, 0.0f)); // 只计算小批量样本点到簇中心的距离

    bool cluster_changed = true;
    int iteration = 1; // 迭代次数

    while (cluster_changed && iteration <= max_iterations) {
        cluster_changed = false;
        cout << "Iteration " << iteration << ":" << endl;
        iteration++;
        auto iteration_start = chrono::high_resolution_clock::now();
        // 2.1 随机选择一个小批量的数据进行处理
        vector<long long> batch_idx(batch_size);
        for (long long i = 0; i < batch_size; ++i) {
            batch_idx[i] = dis(gen);
        }

        // 2.2 计算小批量数据点到簇中心的距离，并重新分配簇
        for (long long i = 0; i < batch_size; ++i) {
            float min_distance = numeric_limits<float>::max();
            long long min_index = -1;
            for (long long j = 0; j < k; ++j) {
                float distance = Distance(data[batch_idx[i]], C[j], m);
                D[i][j] = distance;
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

        // 2.3 更新簇中心，只更新参与小批量数据的簇中心
        vector<node> C_new(k);
        vector<long long> counts(k, 0);
        for (auto& c : C_new) {
            c.dimen.resize(m, 0.0f);
        }

        for (long long i = 0; i < batch_size; ++i) {
            long long cluster_idx = idx[batch_idx[i]];
            if (cluster_idx != -1) {
                Add(C_new[cluster_idx], data[batch_idx[i]], m);
                counts[cluster_idx]++;
            }
        }

        for (long long j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (int i = 0; i < m; ++i) {
                    C_new[j].dimen[i] /= counts[j];
                }
                C[j] = C_new[j];
            }
        }
        auto iteration_end = chrono::high_resolution_clock::now();
        chrono::duration<double> iteration_time = iteration_end - iteration_start;
        cout << "Iteration time: " << iteration_time.count() << " seconds" << endl;
    }

    // 输出聚类结果
    for (long long i = 0; i < k; ++i) {
        cout << "第 " << i + 1 << " 个簇的中心点：";
        for (int j = 0; j < m; ++j) {
            cout << C[i].dimen[j] << " ";
        }
        cout << endl;
    }

    GetSystemTimePreciseAsFileTime(&end_time);

    start_time_us.LowPart = start_time.dwLowDateTime;
    start_time_us.HighPart = start_time.dwHighDateTime;
    end_time_us.LowPart = end_time.dwLowDateTime;
    end_time_us.HighPart = end_time.dwHighDateTime;

    ULONGLONG elapsed_time = end_time_us.QuadPart - start_time_us.QuadPart;
    ULONGLONG elapsed_seconds = elapsed_time / 10000000;
    ULONGLONG elapsed_nanoseconds = (elapsed_time % 10000000) * 100;

    printf("%llu.%09llu seconds\n", elapsed_seconds, elapsed_nanoseconds);
}

int main() {
    long long n, m, k = 80, batch_size = 1000;
    int max_iterations = 10000; // 设置最大迭代次数
    vector<node> data;

    readData("AirPlane1.txt", data, n, m);

    Mini_Batch_Kmeans(k, data, n, m, batch_size, max_iterations);

    return 0;
}
