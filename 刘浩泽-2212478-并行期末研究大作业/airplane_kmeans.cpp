#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include <immintrin.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <sstream>

using namespace std;

struct node {
    vector<float> dimen;
};

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
            try {
                temp.dimen.push_back(stof(value));
            }
            catch (const invalid_argument& e) {
                cerr << "Invalid argument: " << value << " cannot be converted to float." << endl;
                temp.dimen.push_back(0); // 或者根据需要处理，设置默认值
            }
            catch (const out_of_range& e) {
                cerr << "Out of range: " << value << " is out of range for float." << endl;
                temp.dimen.push_back(0); // 或者根据需要处理，设置默认值
            }
        }
        data.push_back(temp);
    }

    n = data.size();
    if (!data.empty()) {
        m = data[0].dimen.size();
    }
    else {
        m = 0;
    }
}

__global__ void calculateDistancesCUDA(float* data, float* centroids, float* distances, int n, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int j = 0; j < k; j++) {
            float distance = 0;
            for (int d = 0; d < m; d++) {
                float diff = data[idx * m + d] - centroids[j * m + d];
                distance += diff * diff;
            }
            distances[idx * k + j] = sqrt(distance);
        }
    }
}

void calculateDistancesAVX(float* data, float* centroids, float* distances, int n, int m, int k) {
#pragma omp parallel for schedule(guided) num_threads(16)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            __m256 distance = _mm256_setzero_ps();
            for (int d = 0; d < m; d += 8) {
                __m256 data_vec = _mm256_loadu_ps(&data[i * m + d]);
                __m256 cent_vec = _mm256_loadu_ps(&centroids[j * m + d]);
                __m256 diff = _mm256_sub_ps(data_vec, cent_vec);
                __m256 diff_sq = _mm256_mul_ps(diff, diff);
                distance = _mm256_add_ps(distance, diff_sq);
            }
            float dist_array[8];
            _mm256_storeu_ps(dist_array, distance);
            distances[i * k + j] = sqrt(dist_array[0] + dist_array[1] + dist_array[2] + dist_array[3] + dist_array[4] + dist_array[5] + dist_array[6] + dist_array[7]);
        }
    }
}

__global__ void assignClustersCUDA(float* distances, int* labels, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float minDistance = distances[idx * k];
        int minIndex = 0;
        for (int j = 1; j < k; j++) {
            if (distances[idx * k + j] < minDistance) {
                minDistance = distances[idx * k + j];
                minIndex = j;
            }
        }
        labels[idx] = minIndex;
    }
}

void assignClustersOMP(float* distances, int* labels, int n, int k) {
#pragma omp parallel for schedule(guided) num_threads(16)
    for (int i = 0; i < n; i++) {
        float minDistance = distances[i * k];
        int minIndex = 0;
        for (int j = 1; j < k; j++) {
            if (distances[i * k + j] < minDistance) {
                minDistance = distances[i * k + j];
                minIndex = j;
            }
        }
        labels[i] = minIndex;
    }
}

void updateCentroidsOMP(float* data, float* centroids, int* labels, int* clusterSizes, int n, int m, int k) {
#pragma omp parallel for schedule(guided) num_threads(16)
    for (int i = 0; i < k * m; ++i) {
        centroids[i] = 0;
    }

#pragma omp parallel num_threads(16)
    {
        vector<float> local_centroids(k * m, 0);
        vector<int> local_clusterSizes(k, 0);

#pragma omp for nowait
        for (int i = 0; i < n; i++) {
            int cluster = labels[i];
            for (int d = 0; d < m; d++) {
                local_centroids[cluster * m + d] += data[i * m + d];
            }
            local_clusterSizes[cluster]++;
        }

#pragma omp critical
        {
            for (int i = 0; i < k * m; i++) {
                centroids[i] += local_centroids[i];
            }
            for (int i = 0; i < k; i++) {
                clusterSizes[i] += local_clusterSizes[i];
            }
        }
    }
}

__global__ void normalizeCentroidsCUDA(float* centroids, int* clusterSizes, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        for (int d = 0; d < m; d++) {
            centroids[idx * m + d] /= clusterSizes[idx];
        }
    }
}

void Kmeans(long long k, vector<node>& data, long long n, long long m) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Distribute data among processes
    long long local_n = n / size;
    long long start_idx = rank * local_n;
    long long end_idx = (rank == size - 1) ? n : start_idx + local_n;
    local_n = end_idx - start_idx;

    float* d_data;
    float* d_centroids;
    float* d_distances;
    int* d_labels;
    int* d_clusterSizes;

    cudaMalloc(&d_data, local_n * m * sizeof(float));
    cudaMalloc(&d_centroids, k * m * sizeof(float));
    cudaMalloc(&d_distances, local_n * k * sizeof(float));
    cudaMalloc(&d_labels, local_n * sizeof(int));
    cudaMalloc(&d_clusterSizes, k * sizeof(int));

    vector<float> h_data(local_n * m);
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < m; ++j) {
            h_data[(i - start_idx) * m + j] = data[i].dimen[j];
        }
    }

    cudaMemcpy(d_data, h_data.data(), local_n * m * sizeof(float), cudaMemcpyHostToDevice);

    vector<float> h_centroids(k * m);
    if (rank == 0) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, n - 1);
        for (int i = 0; i < k; ++i) {
            int idx_init = dis(gen);
            for (int j = 0; j < m; ++j) {
                h_centroids[i * m + j] = data[idx_init].dimen[j];
            }
        }
    }

    MPI_Bcast(h_centroids.data(), k * m, MPI_FLOAT, 0, MPI_COMM_WORLD);
    cudaMemcpy(d_centroids, h_centroids.data(), k * m * sizeof(float), cudaMemcpyHostToDevice);

    vector<int> idx(local_n, -1);
    bool cluster_changed = true;
    int it = 1;
    while (cluster_changed) {
        cout << it << endl;
        it++;
        cluster_changed = false;
        int blockSize = 256;
        int numBlocks = (local_n + blockSize - 1) / blockSize;

        // Use GPU for calculating distances and assigning clusters
        calculateDistancesCUDA << <numBlocks, blockSize >> > (d_data, d_centroids, d_distances, local_n, m, k);
        cudaDeviceSynchronize();

        assignClustersCUDA << <numBlocks, blockSize >> > (d_distances, d_labels, local_n, k);
        cudaDeviceSynchronize();

        // Copy labels from GPU to CPU for updating centroids on CPU
        vector<int> h_labels(local_n);
        cudaMemcpy(h_labels.data(), d_labels, local_n * sizeof(int), cudaMemcpyDeviceToHost);

        // Use CPU (OpenMP + AVX) for updating centroids
        vector<int> h_clusterSizes(k, 0);
        updateCentroidsOMP(h_data.data(), h_centroids.data(), h_labels.data(), h_clusterSizes.data(), local_n, m, k);

        // Reduce centroids and cluster sizes across all processes using non-blocking communication
        vector<float> global_centroids(k * m, 0);
        vector<int> global_clusterSizes(k, 0);
        MPI_Request request_centroids, request_sizes;
        MPI_Iallreduce(h_centroids.data(), global_centroids.data(), k * m, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &request_centroids);
        MPI_Iallreduce(h_clusterSizes.data(), global_clusterSizes.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &request_sizes);

        // Perform computation while waiting for non-blocking communication to complete
#pragma omp parallel for schedule(guided) num_threads(16)
        for (int i = 0; i < local_n; ++i) {
            if (h_labels[i] != idx[i]) {
                idx[i] = h_labels[i];
                cluster_changed = true;
            }
        }

        int local_cluster_changed = cluster_changed ? 1 : 0;
        int global_cluster_changed = 0;
        MPI_Allreduce(&local_cluster_changed, &global_cluster_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        cluster_changed = global_cluster_changed;

        // Wait for non-blocking operations to complete
        MPI_Wait(&request_centroids, MPI_STATUS_IGNORE);
        MPI_Wait(&request_sizes, MPI_STATUS_IGNORE);

        // Normalize centroids on GPU
        cudaMemcpy(d_centroids, global_centroids.data(), k * m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_clusterSizes, global_clusterSizes.data(), k * sizeof(int), cudaMemcpyHostToDevice);
        normalizeCentroidsCUDA << <(k + blockSize - 1) / blockSize, blockSize >> > (d_centroids, d_clusterSizes, m, k);
        cudaDeviceSynchronize();

        // Copy updated centroids back to host
        cudaMemcpy(h_centroids.data(), d_centroids, k * m * sizeof(float), cudaMemcpyHostToHost);

        if (rank == 0) {
            cout << "Iteration completed." << endl;
        }
    }

    if (rank == 0) {
        for (long long i = 0; i < k; ++i) {
            cout << "Cluster " << i + 1 << " centroid: ";
            for (int j = 0; j < m; ++j) {
                cout << h_centroids[i * m + j] << " ";
            }
            cout << endl;
        }
    }

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_distances);
    cudaFree(d_labels);
    cudaFree(d_clusterSizes);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    omp_set_num_threads(16);  // Set the number of OpenMP threads to 16

    long long n, m, k = 8;
    vector<node> data;

    readData("D:\\vs2022\\MPI\\MPI\\AirPlane1.txt", data, n, m);

    auto start_time = chrono::high_resolution_clock::now();

    Kmeans(k, data, n, m);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end_time - start_time;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        cout << "K-means completed in " << duration.count() << " seconds." << endl;
    }


    MPI_Finalize();
    return 0;
}