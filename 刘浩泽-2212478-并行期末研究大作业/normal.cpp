#include<iostream>
#include<vector>
#include<random>
#include<ctime>
#include<cmath>
#include<algorithm>
#include <chrono>
#include<fstream>
#include<sstream>
#include <stdio.h>
#include <windows.h>
#include <sysinfoapi.h>
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
    FILETIME start_time, end_time;
    ULARGE_INTEGER start_time_us, end_time_us;

    // 获取开始时间
    GetSystemTimePreciseAsFileTime(&start_time);

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
    while (cluster_changed) {
        cluster_changed = false;
        cout << "Iteration " << iteration << ":" << endl;
        iteration++;
        auto iteration_start = chrono::high_resolution_clock::now();

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
        vector<node> sum_cluster(k);
        vector<long long> count_cluster(k, 0);

        for (long long j = 0; j < k; ++j) {
            sum_cluster[j].dimen.resize(m, 0.0f); // 初始化 sum_cluster 的 dimen
        }

        for (long long i = 0; i < n; ++i) {
            int cluster_idx = idx[i];
            if (cluster_idx != -1) {
                Add(sum_cluster[cluster_idx], data[i], m);
                count_cluster[cluster_idx]++;
            }
        }

        for (long long j = 0; j < k; ++j) {
            if (count_cluster[j] > 0) {
                for (int i = 0; i < m; ++i) {
                    C[j].dimen[i] = sum_cluster[j].dimen[i] / count_cluster[j];
                }
            }
        }

        auto iteration_end = chrono::high_resolution_clock::now();
        chrono::duration<double> iteration_time = iteration_end - iteration_start;
        cout << "Iteration time: " << iteration_time.count() << " seconds" << endl;
    }

    if (!cluster_changed) {
        // 输出聚类结果
        for (long long i = 0; i < k; ++i) {
            cout << "第 " << i + 1 << " 个簇的中心点：";
            for (int j = 0; j < m; ++j) {
                cout << C[i].dimen[j] << " ";
            }
            cout << endl;
        }
    }

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
}

int main()
{
    long long n, m, k = 8;
    vector<node> data;
    cout<<1<<endl;
    readData("AirPlane1.txt", data, n, m);
        // 输出 data 容器中的数据
    // std::cout << "Generated data:\n";
    // for (const auto& node : data) {
    //     std::cout << "Node: ";
    //     for (float dim : node.dimen) {
    //         std::cout << dim << " ";
    //     }
    //     std::cout << "\n";
    // }
    // FILETIME start_time, end_time;
    // ULARGE_INTEGER start_time_us, end_time_us;
    // // auto start_time = chrono::high_resolution_clock::now(); // 记录开始时间
    // // for (int i = 1; i <= 100; i++) {
    //     // Kmeans(k, data, n, m);
    // // }
    // // auto end_time = chrono::high_resolution_clock::now(); // 记录结束时间
    // // auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time); // 计算经过的时间
    // // cout << "Kmeans 算法执行时间: " << elapsed_time.count() << " 毫秒" << endl;
    //     // 获取开始时间
    // GetSystemTimePreciseAsFileTime(&start_time);

    Kmeans(k, data, n, m);
    // 获取结束时间
    // GetSystemTimePreciseAsFileTime(&end_time);

    // // 计算执行时间
    // start_time_us.LowPart = start_time.dwLowDateTime;
    // start_time_us.HighPart = start_time.dwHighDateTime;
    // end_time_us.LowPart = end_time.dwLowDateTime;
    // end_time_us.HighPart = end_time.dwHighDateTime;

    // ULONGLONG elapsed_time = end_time_us.QuadPart - start_time_us.QuadPart;
    // ULONGLONG elapsed_seconds = elapsed_time / 10000000;
    // ULONGLONG elapsed_nanoseconds = (elapsed_time % 10000000) * 100;

    // printf("%llu.%09llu seconds\n", elapsed_seconds, elapsed_nanoseconds);
    return 0;
}