//
// knnsearch method
// Copyright (C) 2021 shihan-ma mmasss1205@gmail.com
//
// @file    kdTree.h
// @brief   provide knnsearch using KD tree
// @author  shihan-ma
// @email   mmasss1205@gmail.com
// @birth   Created by Shihan on 2021/02/04
//

#ifndef TEST_KNNSEARCH_KDTREE_H
#define TEST_KNNSEARCH_KDTREE_H

#include <cstdio>
#include <vector>
#include <tuple>
#include <stack>
#include <memory>
#include <queue>

struct tree_node {
    size_t id;
    size_t split;
    tree_node *left, *right;
};

class kdTree {
public:
    kdTree(){}
    kdTree(const std::vector<std::vector<double>> &datas, size_t rows, size_t cols, double p);

    std::vector<std::tuple<size_t, double>> FindKNearests(const std::vector<double> &coor, size_t k);


private:
    tree_node * root_;  // root node of tree
    double p_;          // p-distance: dist(x,y) = pow((x^p+y^p),1/p)
    const std::vector<std::vector<double>> datas_;
    size_t n_samples_;  // size of train set
    size_t n_features_; // dimension of each feature
    const std::vector<int> labels_;
    std::vector<bool> visited_buf_;
    std::vector<std::tuple<size_t, double>> get_mid_buf_;

    // Sample with the largest distance at heap top
    struct neighbor_heap_cmp {
        bool operator()(const std::tuple<size_t, double> &i,
                const std::tuple<size_t, double> &j) {
            return std::get<1>(i) < std::get<1>(j);
        }
    };
    typedef std::tuple<size_t, double> neighbor;
    typedef std::priority_queue<neighbor, std::vector<neighbor>, neighbor_heap_cmp> neighbor_heap;

    neighbor_heap k_neighbor_heap_;

    void reset_buf() {
        for (int i=0; i<n_samples_; ++i)
            visited_buf_[i] = false;
    }

    // Build tree
    tree_node* BuildTree(const std::vector<size_t> &points);
    // Find medium of an array
    std::tuple<size_t, double> MidElement(const std::vector<size_t> &points, size_t dim);
    // Push into heap
    void HeapStackPush(std::stack<tree_node*> &paths, tree_node *node, const std::vector<double> &coor, size_t k);
    // Get value of dim th feature of sample th sample
    double GetDimVal(size_t sample, size_t dim);
    // Calculate distance of coor and ith sample in train set
    double CalcDist(size_t i, const std::vector<double> &coor);
    // Find split point
    size_t FindSplitDim(const std::vector<size_t> &points);
};

void find_k_nearests(kdTree *tree, const std::vector<double> &coor, size_t k, std::vector<size_t> &ids, std::vector<double> &dists);

#endif //TEST_KNNSEARCH_KDTREE_H
