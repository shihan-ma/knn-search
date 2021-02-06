#include <iostream>
#include <functional>
#include "kdTree.h"

using namespace std;

// for testing unique_ptr deleter
void my_deleter(kdTree *obj) {
    cout << "Delete unique pointer of kdTree." << endl;
    delete obj;
}

int main() {
    // test1 - find k nearest neighbors of each point within the dataset
    // dataset
    vector<vector<double>> data = {{1.3, 1.2, 1.4}, {9.3, 8.3, 1.3}, {1.3, 2.3, 2},
                                   {1.2, 2.2, 1.5}, {7.5, 7.3, 7}, {9.2, 9.3, 9},
                                   {15.5, 15, 15}, {5, 3, 4}, {1.5, 1.1, 1.2},
                                   {10, 12, 11}, {4, 5.5, 2}, {5, 4, 3}};
    int n_samples(data.size());
    int n_features(data[0].size());
    int p = 2;                      // p-norm
    int neighbor = 4;

    // build KD tree
    unique_ptr<kdTree,decltype(my_deleter)*> tree (new kdTree(data, n_samples, n_features, p), &my_deleter);

    vector<vector<size_t>> indexes;
    vector<vector<double>> dists;
    vector<size_t> tmp_id;
    vector<double> tmp_dist;
    for (int i=0; i<n_samples; ++i) {
        find_k_nearests(tree.get(), data[i], neighbor, tmp_id, tmp_dist);
        indexes.push_back(tmp_id);
        dists.push_back(tmp_dist);
    }


    printf("%d-Nearest: \n", neighbor);
    for (int row=0; row<n_samples; ++row) {
        printf("%d sample:\t", row);
        for (int col = 0; col < neighbor; ++col) {
            printf("ID %llu\tdistance %.2f\t", indexes[row][col], dists[row][col]);
        }
        printf("\n");
    }

    return 0;
}
