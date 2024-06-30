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
#include <windows.h>
#include <sysinfoapi.h>

using namespace std;

struct Node {
    vector<float> dimen;
      // 定义 operator== 用于比较两个 Node 对象是否相等
    bool operator==(const Node& other) const {
        if (dimen.size() != other.dimen.size()) return false;
        for (size_t i = 0; i < dimen.size(); ++i) {
            if (dimen[i] != other.dimen[i]) return false;
        }
        return true;
    }
};

void readData(const string& filename, vector<Node>& data, long long& n, long long& m) {
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
        Node temp;
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

float Distance(const Node& X, const Node& Z, long long n) {
    float result = 0;
    for (long long i = 0; i < n; i++) {
        result += (X.dimen[i] - Z.dimen[i]) * (X.dimen[i] - Z.dimen[i]);
    }
    return sqrt(result);
}

void Add(Node& result, const Node& X, long long n) {
    for (long long i = 0; i < n; i++) {
        result.dimen[i] += X.dimen[i];
    }
}
class KDTree {
public:
    KDTree(const vector<Node>& points) : root(nullptr) {
        if (!points.empty()) {
            root = build(points, 0);
        }
    }

    Node nearest(const Node& target, float epsilon = 0.1) {
        float best_dist = numeric_limits<float>::max();
        return nearest(root, target, 0, best_dist, epsilon)->point;
    }

private:
    struct TreeNode {
        Node point;
        TreeNode* left;
        TreeNode* right;
        float radius;
        bool needs_update;
        TreeNode(const Node& p) : point(p), left(nullptr), right(nullptr), radius(0), needs_update(false) {}
    };

    TreeNode* root;

    TreeNode* build(vector<Node> points, int depth) {
        if (points.empty()) return nullptr;

        size_t axis = depth % points[0].dimen.size();
        size_t median = points.size() / 2;

        nth_element(points.begin(), points.begin() + median, points.end(),
            [axis](const Node& a, const Node& b) { return a.dimen[axis] < b.dimen[axis]; });

        TreeNode* node = new TreeNode(points[median]);
        vector<Node> leftPoints(points.begin(), points.begin() + median);
        vector<Node> rightPoints(points.begin() + median + 1, points.end());

        node->left = build(leftPoints, depth + 1);
        node->right = build(rightPoints, depth + 1);

        updateRadius(node);
        return node;
    }

    void updateRadius(TreeNode* node) {
        if (!node) return;

        float max_dist = 0;
        if (node->left) {
            max_dist = max(max_dist, distanceSquared(node->point, node->left->point));
            max_dist = max(max_dist, node->left->radius);
        }
        if (node->right) {
            max_dist = max(max_dist, distanceSquared(node->point, node->right->point));
            max_dist = max(max_dist, node->right->radius);
        }
        node->radius = sqrt(max_dist);
        node->needs_update = false;
    }

    TreeNode* nearest(TreeNode* node, const Node& target, int depth, float& best_dist, float epsilon) {
        if (!node) return nullptr;

        if (node->needs_update) {
            updateRadius(node);
        }

        size_t axis = depth % target.dimen.size();
        float d = target.dimen[axis] - node->point.dimen[axis];
        TreeNode* nextNode = d < 0 ? node->left : node->right;
        TreeNode* otherNode = d < 0 ? node->right : node->left;

        TreeNode* best = closer(target, nearest(nextNode, target, depth + 1, best_dist, epsilon), node, best_dist);

        if (d * d < best_dist * (1 + epsilon)) {
            best = closer(target, nearest(otherNode, target, depth + 1, best_dist, epsilon), best, best_dist);
        }

        return best;
    }

    TreeNode* closer(const Node& target, TreeNode* a, TreeNode* b, float& best_dist) {
        if (!a) return b;
        if (!b) return a;

        float dist_a = distanceSquared(target, a->point);
        float dist_b = distanceSquared(target, b->point);

        if (dist_a < dist_b) {
            if (dist_a < best_dist) best_dist = dist_a;
            return a;
        } else {
            if (dist_b < best_dist) best_dist = dist_b;
            return b;
        }
    }

    float distanceSquared(const Node& a, const Node& b) {
        float dist = 0;
        for (size_t i = 0; i < a.dimen.size(); ++i) {
            dist += (a.dimen[i] - b.dimen[i]) * (a.dimen[i] - b.dimen[i]);
        }
        return dist;
    }
};
vector<Node> KMeansPlusPlusInit(vector<Node>& data, long long k, long long n, long long m) {
    vector<Node> centers;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n - 1);

    // 1. 随机选择一个数据点作为第一个簇中心
    centers.push_back(data[dis(gen)]);

    vector<float> min_distances(n, numeric_limits<float>::max());

    // 2. 选择其余的簇中心
    for (long long i = 1; i < k; ++i) {
        float total_distance = 0;
        // 2.1 计算每个点到最近中心点的距离的平方
        for (long long j = 0; j < n; ++j) {
            float distance = Distance(data[j], centers.back(), m);
            min_distances[j] = min(min_distances[j], distance);
            total_distance += min_distances[j] * min_distances[j];
        }

        // 2.2 选择一个点，其被选择的概率与距离平方成正比
        uniform_real_distribution<> dist(0, total_distance);
        float r = dist(gen);
        float cumulative_distance = 0;
        for (long long j = 0; j < n; ++j) {
            cumulative_distance += min_distances[j] * min_distances[j];
            if (cumulative_distance >= r) {
                centers.push_back(data[j]);
                break;
            }
        }
    }

    return centers;
}

void Kmeans(long long k, vector<Node>& data, long long n, long long m) {
    FILETIME start_time, end_time;
    ULARGE_INTEGER start_time_us, end_time_us;

    GetSystemTimePreciseAsFileTime(&start_time);

    vector<long int> idx(n);
    vector<vector<float>> D(n, vector<float>(k)); // 存储样本点到簇中心的距离

    // K-means++ 初始化
    vector<Node> C = KMeansPlusPlusInit(data, k, n, m);

    KDTree kdTree(C); // 使用KDTree初始化中心点

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
            Node nearestCenter = kdTree.nearest(data[i]);
            float min_distance = Distance(data[i], nearestCenter, m);
            long long min_index = distance(C.begin(), find(C.begin(), C.end(), nearestCenter));

            if (idx[i] != min_index) {
                idx[i] = min_index;
                cluster_changed = true;
            }
        }
        
        // 2.2 更新簇中心
        vector<Node> new_centers(k, Node());
        vector<long long> count(k, 0);

        for (long long i = 0; i < k; ++i) {
            new_centers[i].dimen.resize(m, 0);
        }

        for (long long i = 0; i < n; ++i) {
            Add(new_centers[idx[i]], data[i], m);
            count[idx[i]]++;
        }

        for (long long i = 0; i < k; ++i) {
            if (count[i] > 0) {
                for (long long j = 0; j < m; ++j) {
                    new_centers[i].dimen[j] /= count[i];
                }
            }
        }

        // 2.3 更新KDTree
        C = new_centers;
        kdTree = KDTree(C);
        auto iteration_end = chrono::high_resolution_clock::now();
        chrono::duration<double> iteration_time = iteration_end - iteration_start;
        cout << "Iteration time: " << iteration_time.count() << " seconds" << endl;
        // cout << "Cluster centers updated." << endl;
    }
     if (!cluster_changed) {
        // 输出聚类结果
        for (long long i = 0; i < k; ++i) {
            cout << "第 " << i + 1 << " 个簇的中心为: (";
            for (int j = 0; j < m; ++j) {
                cout << C[i].dimen[j];
                if (j != m - 1) cout << ", ";
            }
            cout << ")" << endl;
        }
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
    long long n, m, k = 8, batch_size = 1000;
    // int max_iterations = 10000; // 设置最大迭代次数
    vector<Node> data;
    cout<<1<<endl;
    readData("AirPlane1.txt", data, n, m);
    cout<<1<<endl;
    Kmeans(k, data, n, m);

    return 0;
}