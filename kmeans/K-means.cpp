#include<iostream>
#include<vector>
#include<random>
#include<ctime>
#include<cmath>
#include<algorithm>
#include <chrono>
#include<fstream>
#include<sstream>
using namespace std;
struct node
{
	vector<long double> dimen;
};
vector<node> data1;

long double Distance(const node& X, const node& Z, long long n) {
	long double result = 0;
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
    vector<node> C(k); // 存储簇中心
    vector<long int> idx(n);
    vector<vector<double>> D(n, vector<double>(k)); // 存储样本点到簇中心的距离

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
    while (cluster_changed) {
        cluster_changed = false;
        // 2.1 计算每个样本点到簇中心的距离，并重新分配簇
        for (long long i = 0; i < n; ++i) {
            long double min_distance = numeric_limits<long double>::max();
            long long min_index = -1;
            for (long long j = 0; j < k; ++j) {
                long double distance = Distance(data[i], C[j], m);
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
        for (long long j = 0; j < k; ++j) {
            node* sum_cluster = new node;
            sum_cluster->dimen.resize(m); // 设置 dimen 向量的大小为 m
            long long count_cluster = 0;
            for (long long i = 0; i < n; ++i) {
                if (idx[i] == j) {
                    Add(*sum_cluster, data[i], m);
                    ++count_cluster;
                }
            }
            if (count_cluster > 0) {
                for (int i = 0; i < m; ++i) {
                    C[j].dimen[i] = sum_cluster->dimen[i] / count_cluster;
                }
            }
        }
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
int main()
{
	long long n,m;
    n=899, m = 3;
    string filename = "D:\\CS_work\\files\\ZscoredData.txt";
    ifstream infile(filename);

    if (!infile.is_open()) {
        cerr << "无法打开文件" << endl;
        return 1;
    }

    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        node temp;
        long double value;
        while (ss >> value) {
            temp.dimen.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }
        data1.push_back(temp);
    }

    infile.close();

    // 打印读取的数据点
    for (size_t i = 0; i < data1.size(); ++i) {
        cout << "数据点 " << i + 1 << ": ";
        for (size_t j = 0; j < data1[i].dimen.size(); ++j) {
            cout << data1[i].dimen[j] << " ";
        }
        cout << endl;
    }
	//cout << "请输入数量和维度"<<endl;
	//cin >> n>> m;
	//srand(time(NULL));
	//for (long long i = 0; i < n; i++)
	//{
	//	node*temp = new node;
	//	for (long long j = 0; j < m; j++)
	//	{
	//		temp->dimen.push_back(static_cast<long double>(rand()));
	//	}
	//	data1.push_back(*temp);
	//}
 //   //for (long long i = 0; i < n; i++)
 //   //{
 //   //    node temp;
 //   //    cout << "请输入第 " << i + 1 << " 个数据点的 " << m << " 个维度值: ";
 //   //    for (long long j = 0; j < m; j++)
 //   //    {
 //   //        long double value;
 //   //        cin >> value;
 //   //        temp.dimen.push_back(value);
 //   //    }
 //   //    data1.push_back(temp);
 //   //}
	cout << "请输入类别数" << endl;
    long long k;
	cin >> k;

	for (long long i = 0; i < n; ++i)
	{
		cout << "node " << i + 1 << "：";
		for (long long j = 0; j < m; ++j)
		{
			cout << data1[i].dimen[j] << " ";
		}
		cout << endl;
	}
    auto start_time = chrono::high_resolution_clock::now(); // 记录开始时间
    for (int i = 1; i <= 100; i++) {
        Kmeans(k, data1, n, m);
    }
    auto end_time = chrono::high_resolution_clock::now(); // 记录结束时间
    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time); // 计算经过的时间
    cout << "Kmeans 算法执行时间: " << elapsed_time.count() << " 毫秒" << endl;
    return 0;



}