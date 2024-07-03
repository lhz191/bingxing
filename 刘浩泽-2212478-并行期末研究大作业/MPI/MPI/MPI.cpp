#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h>

using namespace std;

struct node {
    float* dimen;
};

__global__ void calculateDistances(float* data, float* centroids, float* distances, int n, int m, int k) {
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

__global__ void assignClusters(float* distances, int* labels, int n, int k) {
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

__global__ void updateCentroids(float* data, float* centroids, int* labels, int* clusterSizes, int n, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k * m) {
        centroids[idx] = 0;
    }
    __syncthreads();
    if (idx < n) {
        int cluster = labels[idx];
        for (int d = 0; d < m; d++) {
            atomicAdd(&centroids[cluster * m + d], data[idx * m + d]);
        }
        atomicAdd(&clusterSizes[cluster], 1);
    }
}

__global__ void normalizeCentroids(float* centroids, int* clusterSizes, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        for (int d = 0; d < m; d++) {
            centroids[idx * m + d] /= clusterSizes[idx];
        }
    }
}

void generateStructuredData(vector<node>& data, int n, int m) {
    data.resize(n);
    for (int i = 0; i < n; i++) {
        data[i].dimen = new float[m];
        for (int j = 0; j < m; j++) {
            data[i].dimen[j] = static_cast<float>(i + 1);
        }
    }
}

void Kmeans(long long k, vector<node>& data, long long n, long long m) {
    // Allocate and initialize device memory
    float* d_data;
    float* d_centroids;
    float* d_distances;
    int* d_labels;
    int* d_clusterSizes;

    cudaMalloc(&d_data, n * m * sizeof(float));
    cudaMalloc(&d_centroids, k * m * sizeof(float));
    cudaMalloc(&d_distances, n * k * sizeof(float));
    cudaMalloc(&d_labels, n * sizeof(int));
    cudaMalloc(&d_clusterSizes, k * sizeof(int));

    vector<float> h_data(n * m);
#pragma omp parallel for simd
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            h_data[i * m + j] = data[i].dimen[j];
        }
    }

    cudaMemcpy(d_data, h_data.data(), n * m * sizeof(float), cudaMemcpyHostToDevice);

    vector<float> h_centroids(k * m);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n - 1);
    for (int i = 0; i < k; ++i) {
        int idx_init = dis(gen);
        for (int j = 0; j < m; ++j) {
            h_centroids[i * m + j] = data[idx_init].dimen[j];
        }
    }

    cudaMemcpy(d_centroids, h_centroids.data(), k * m * sizeof(float), cudaMemcpyHostToDevice);

    vector<int> idx(n, -1);

    bool cluster_changed = true;
    while (cluster_changed) {
        cluster_changed = false;
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;

        calculateDistances << <numBlocks, blockSize >> > (d_data, d_centroids, d_distances, n, m, k);
        cudaDeviceSynchronize();

        assignClusters << <numBlocks, blockSize >> > (d_distances, d_labels, n, k);
        cudaDeviceSynchronize();

        cudaMemset(d_clusterSizes, 0, k * sizeof(int));
        updateCentroids << <numBlocks, blockSize >> > (d_data, d_centroids, d_labels, d_clusterSizes, n, m, k);
        cudaDeviceSynchronize();

        normalizeCentroids << <(k + blockSize - 1) / blockSize, blockSize >> > (d_centroids, d_clusterSizes, m, k);
        cudaDeviceSynchronize();

        vector<int> h_labels(n);
        cudaMemcpy(h_labels.data(), d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);

#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            if (h_labels[i] != idx[i]) {
                idx[i] = h_labels[i];
                cluster_changed = true;
            }
        }
    }

    cudaMemcpy(h_centroids.data(), d_centroids, k * m * sizeof(float), cudaMemcpyDeviceToHost);

    for (long long i = 0; i < k; ++i) {
        cout << "Cluster " << i + 1 << " centroid: ";
        for (int j = 0; j < m; ++j) {
            cout << h_centroids[i * m + j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_distances);
    cudaFree(d_labels);
    cudaFree(d_clusterSizes);
}

int main() {
    long long n = 500000, m = 10, k = 5;
    vector<node> data;

    generateStructuredData(data, n, m);

    auto start_time = chrono::high_resolution_clock::now();

    Kmeans(k, data, n, m);

    auto end_time = chrono::high_resolution_clock::now();
    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Kmeans algorithm execution time: " << elapsed_time.count() << " milliseconds" << endl;

    return 0;
}
