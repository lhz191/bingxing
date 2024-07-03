#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

struct node {
    float* dimen;
};

__global__ void calculateBatchDistances(float* data, float* centroids, float* distances, int* batchIdx, int batchSize, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        int dataIdx = batchIdx[idx];
        for (int j = 0; j < k; j++) {
            float distance = 0;
            for (int d = 0; d < m; d++) {
                float diff = data[dataIdx * m + d] - centroids[j * m + d];
                distance += diff * diff;
            }
            distances[idx * k + j] = sqrt(distance);
        }
    }
}

__global__ void assignBatchClusters(float* distances, int* labels, int* batchIdx, int batchSize, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        float minDistance = distances[idx * k];
        int minIndex = 0;
        for (int j = 1; j < k; j++) {
            if (distances[idx * k + j] < minDistance) {
                minDistance = distances[idx * k + j];
                minIndex = j;
            }
        }
        labels[batchIdx[idx]] = minIndex;
    }
}

__global__ void updateBatchCentroids(float* data, float* centroids, int* labels, int* clusterSizes, int* batchIdx, int batchSize, int n, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k * m) {
        centroids[idx] = 0;
    }
    __syncthreads();
    if (idx < batchSize) {
        int dataIdx = batchIdx[idx];
        int cluster = labels[dataIdx];
        for (int d = 0; d < m; d++) {
            atomicAdd(&centroids[cluster * m + d], data[dataIdx * m + d]);
        }
        atomicAdd(&clusterSizes[cluster], 1);
    }
}

__global__ void normalizeBatchCentroids(float* centroids, int* clusterSizes, int m, int k) {
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

void Mini_Batch_Kmeans(long long k, vector<node>& data, long long n, long long m, long long batch_size) {
    // Allocate and initialize device memory
    float* d_data;
    float* d_centroids;
    float* d_distances;
    int* d_labels;
    int* d_clusterSizes;
    int* d_batchIdx;

    cudaMalloc(&d_data, n * m * sizeof(float));
    cudaMalloc(&d_centroids, k * m * sizeof(float));
    cudaMalloc(&d_distances, batch_size * k * sizeof(float));
    cudaMalloc(&d_labels, n * sizeof(int));
    cudaMalloc(&d_clusterSizes, k * sizeof(int));
    cudaMalloc(&d_batchIdx, batch_size * sizeof(int));

    vector<float> h_data(n * m);
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
    vector<int> h_batchIdx(batch_size);

    bool cluster_changed = true;
    int iteration = 1;
    while (cluster_changed) {
        cluster_changed = false;
        cout << "Iteration " << iteration << ":" << endl;
        iteration++;

        for (long long i = 0; i < batch_size; ++i) {
            h_batchIdx[i] = dis(gen);
        }
        cudaMemcpy(d_batchIdx, h_batchIdx.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (batch_size + blockSize - 1) / blockSize;

        calculateBatchDistances<<<numBlocks, blockSize>>>(d_data, d_centroids, d_distances, d_batchIdx, batch_size, m, k);
        cudaDeviceSynchronize();

        assignBatchClusters<<<numBlocks, blockSize>>>(d_distances, d_labels, d_batchIdx, batch_size, k);
        cudaDeviceSynchronize();

        cudaMemset(d_clusterSizes, 0, k * sizeof(int));
        updateBatchCentroids<<<numBlocks, blockSize>>>(d_data, d_centroids, d_labels, d_clusterSizes, d_batchIdx, batch_size, n, m, k);
        cudaDeviceSynchronize();

        normalizeBatchCentroids<<<(k + blockSize - 1) / blockSize, blockSize>>>(d_centroids, d_clusterSizes, m, k);
        cudaDeviceSynchronize();

        vector<int> h_labels(n);
        cudaMemcpy(h_labels.data(), d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < batch_size; ++i) {
            if (h_labels[h_batchIdx[i]] != idx[h_batchIdx[i]]) {
                idx[h_batchIdx[i]] = h_labels[h_batchIdx[i]];
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
    cudaFree(d_batchIdx);
}

int main() {
    long long n = 500000, m = 10, k = 5, batch_size = 100;
    vector<node> data;

    generateStructuredData(data, n, m);

    auto start_time = chrono::high_resolution_clock::now();

    Mini_Batch_Kmeans(k, data, n, m, batch_size);

    auto end_time = chrono::high_resolution_clock::now();
    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Mini-Batch Kmeans algorithm execution time: " << elapsed_time.count() << " milliseconds" << endl;

    return 0;
}
