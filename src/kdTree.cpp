//
// Created by hana on 2021/2/4.
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <kdTree.h>

using namespace std;

// Find k nearest neighbours of a tree. I
// Index and distance of ki to coor are stored in ids and dists, respectively.
void find_k_nearests(kdTree *tree, const vector<double> &coor, size_t k, vector<size_t> &ids, vector<double> &dists) {
    vector<tuple<size_t, double>> k_nearest = tree->FindKNearests(coor, k+1);
    ids.resize(k);
    dists.resize(k);
    for (size_t i=0; i<k; ++i) {
        ids[i] = get<0>(k_nearest[i]);
        dists[i] = get<1>(k_nearest[i]);
    }
}

// Constructor
kdTree::kdTree(const vector<vector<double>> &datas,
               size_t rows, size_t cols, double p) :
               datas_(datas), n_samples_(rows), n_features_(cols), p_(p) {
    vector<size_t> points;
    for (size_t i=0; i<n_samples_; ++i)
        points.emplace_back(i);
    get_mid_buf_.resize(n_samples_);
    visited_buf_.resize(n_samples_);
    root_ = BuildTree(points);
}

vector<tuple<size_t, double>> kdTree::FindKNearests(const vector<double> &coor, size_t k) {
    stack<tree_node *> paths;
    tree_node *p1 = root_;
    reset_buf();
    while (p1) {
        HeapStackPush(paths, p1, coor, k);
        p1 = coor[p1->split] <= GetDimVal(p1->id, p1->split) ? p1->left : p1->right;
    }
    while (!paths.empty()) {
        p1 = paths.top();
        paths.pop();

        if (!p1->left && !p1->right)
            continue;
        if(k_neighbor_heap_.size()<k) {
            if (p1->left)
                HeapStackPush(paths, p1->left, coor, k);
            if (p1->right)
                HeapStackPush(paths, p1->right, coor, k);
        } else {
            double node_split_val = GetDimVal(p1->id, p1->split);
            double coor_split_val = coor[p1->split];
            double heap_top_val = get<1>(k_neighbor_heap_.top());
            if (coor_split_val > node_split_val) {
                if (p1->right)
                    HeapStackPush(paths, p1->right, coor, k);
                if ((coor_split_val-node_split_val) < heap_top_val && p1->left)
                    HeapStackPush(paths, p1->left, coor, k);
            } else {
                if (p1->left)
                    HeapStackPush(paths, p1->left, coor, k);
                if ((node_split_val-coor_split_val)<heap_top_val && p1->right)
                    HeapStackPush(paths, p1->right, coor, k);
            }
        }
    }
    vector<tuple<size_t,double>> res;
    while (!k_neighbor_heap_.empty()) {
        res.emplace_back(k_neighbor_heap_.top());
        k_neighbor_heap_.pop();
    }
    return res;
}
tree_node* kdTree::BuildTree(const vector<size_t> &points) {
    size_t dim = FindSplitDim(points);
    tuple<size_t, double> t = MidElement(points, dim);
    size_t arg_mid_val = get<0>(t);
    double mid_val = get<1>(t);

    auto *node = new tree_node;
    node->left = nullptr;
    node->right = nullptr;
    node->id = arg_mid_val;
    node->split = dim;
    vector<size_t> left, right;
    for (auto &i:points) {
        if (i==arg_mid_val)
            continue;
        if (GetDimVal(i,dim)<=mid_val)
            left.emplace_back(i);
        else
            right.emplace_back(i);
    }
    if (!left.empty())
        node->left = BuildTree(left);
    if (!right.empty())
        node->right = BuildTree(right);
    return node;
}

tuple<size_t, double> kdTree::MidElement(const vector<size_t> &points, size_t dim) {
    size_t len = points.size();
    for (size_t i=0; i<points.size(); ++i)
        get_mid_buf_[i] = make_tuple(points[i], GetDimVal(points[i], dim));
    nth_element(get_mid_buf_.begin(),
                get_mid_buf_.begin() +len/2,
                get_mid_buf_.begin() +len,
                [](const tuple<size_t,double> &i, const tuple<size_t, double> &j) {
        return get<1>(i)<get<1>(j);
    });
    return get_mid_buf_[len/2];
}

inline void kdTree::HeapStackPush(stack<tree_node *> &paths, tree_node *node, const vector<double> &coor,
                                  size_t k) {
    paths.emplace(node);
    size_t id = node->id;
    if (visited_buf_[id])
        return;
    visited_buf_[id] = true;
    double dist = CalcDist(id, coor);
    tuple<size_t, double> t(id, dist);
    if (k_neighbor_heap_.size()<k)
        k_neighbor_heap_.push(t);
    else if (get<1>(t)<get<1>(k_neighbor_heap_.top())) {
        k_neighbor_heap_.pop();
        k_neighbor_heap_.push(t);
    }
}

inline double kdTree::CalcDist(size_t i, const vector<double> &coor) {
    double dist = 0.0;
    for (int t=0; t<n_features_; ++t)
        dist += pow(datas_[i][t]-coor[t],p_);
    return double(pow(dist, 1.0/p_));
}

size_t kdTree::FindSplitDim(const vector<size_t> &points) {
    if (points.size() == 1)
        return 0;
    size_t cur_best_dim = 0;
    double cur_largest_spread = -1;
    double cur_min_val;
    double cur_max_val;
    for (size_t dim =0; dim<n_features_; ++dim) {
        cur_min_val = GetDimVal(points[0],dim);
        cur_max_val = GetDimVal(points[0], dim);
        for (const auto &id:points) {
            if (GetDimVal(id, dim)>cur_max_val)
                cur_max_val = GetDimVal(id, dim);
            else if (GetDimVal(id, dim)<cur_min_val)
                cur_min_val = GetDimVal(id, dim);
        }
        if (cur_max_val-cur_min_val > cur_largest_spread) {
            cur_largest_spread = cur_max_val - cur_min_val;
            cur_best_dim = dim;
        }
    }
    return cur_best_dim;
}

double kdTree::GetDimVal(size_t sample, size_t dim) {
    return datas_[sample][dim]
;}
