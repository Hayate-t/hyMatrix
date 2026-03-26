#include <iostream>
#include <vector>
#include "Matrix.hpp"

int main() {

    // データ点
    std::vector<double> x = {1, 2, 3, 4};
    std::vector<double> y = {2.0, 2.9, 4.1, 5.05};

    size_t n = x.size();

    // A行列（n×2）
    linalg::Matrix A(n, 2);
    for (size_t i = 0; i < n; i++) {
        A(i, 0) = x[i];
        A(i, 1) = 1.0;
    }

    // bベクトル（n×1）
    linalg::Matrix b(n, 1);
    for (size_t i = 0; i < n; i++) {
        b(i, 0) = y[i];
    }

    // 最小二乗法：x = (A^T A)^-1 A^T b
    linalg::Matrix At = A.transpose();
    linalg::Matrix AtA = At * A;
    linalg::Matrix Atb = At * b;

    linalg::Matrix result = linalg::Matrix::solve(AtA, Atb);

    double a = result(0, 0);
    double b0 = result(1, 0);

    std::cout << "y = " << a << "x + " << b0 << std::endl;

    return 0;
}