#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include <arm_neon.h>

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

    vector<node> C(k);
    for (long long i = 0; i < k; ++i) {
        C[i].dimen.resize(m);
    }
    vector<long int> idx(n);
    vector<vector<float>> D(n, vector<float>(k));

    if (rank == 0) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, n - 1);
        for (long long i = 0; i < k; ++i) {
            int idx_init = dis(gen);
            C[i] = data[idx_init];
        }
    }

    for (long long i = 0; i < k; ++i) {
        MPI_Bcast(C[i].dimen.data(), m, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    bool cluster_changed = true;
    int iteration = 1;
    while (cluster_changed) {
        cluster_changed = false;

        long long chunk_size = 100;
        long long total_chunks = (n + chunk_size - 1) / chunk_size;

        #pragma omp parallel for schedule(guided)
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

        vector<node> local_centers(k);
        vector<long long> local_counts(k, 0);
        for (long long j = 0; j < k; ++j) {
            local_centers[j].dimen.resize(m, 0.0f);
        }

        #pragma omp parallel
        {
            vector<node> local_thread_centers(k);
            vector<long long> local_thread_counts(k, 0);

            for (long long j = 0; j < k; ++j) {
                local_thread_centers[j].dimen.resize(m, 0.0f);
            }

            #pragma omp for nowait
            for (long long i = 0; i < n; ++i) {
                Add(local_thread_centers[idx[i]], data[i], m);
                local_thread_counts[idx[i]]++;
            }

            #pragma omp critical
            {
                for (long long j = 0; j < k; ++j) {
                    Add(local_centers[j], local_thread_centers[j], m);
                    local_counts[j] += local_thread_counts[j];
                }
            }
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
                #pragma omp parallel for
                for (long long l = 0; l < m; ++l) {
                    C[j].dimen[l] = global_centers[j].dimen[l] / global_counts[j];
                }
            }
        }

        int local_cluster_changed = cluster_changed ? 1 : 0;
        int global_cluster_changed = 0;
        MPI_Allreduce(&local_cluster_changed, &global_cluster_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        cluster_changed = global_cluster_changed;

        if (rank == 0) {
            cout << "Iteration " << iteration << " completed." << endl;
        }
        iteration++;
    }

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
