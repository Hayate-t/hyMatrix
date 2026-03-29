#include <iostream>
#include <vector>
#include "hyMatrix.hpp"

void print(const linalg::Matrix& M) {
    std::cout << "[\n";
    for (size_t i = 0; i < M.rows; i++) {
        for (size_t j = 0; j < M.cols; j++) {
            std::cout << M(i, j) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "]\n";
}

int main() {

    // 逆行列が存在する行列
    linalg::Matrix A(3, 3);

    A(0, 0) = 2;  A(0, 1) = -1; A(0, 2) = 0;
    A(1, 0) = -1; A(1, 1) = 2;  A(1, 2) = -1;
    A(2, 0) = 0;  A(2, 1) = -1; A(2, 2) = 2;

    std::cout << "Matrix A:\n";
    print(A);

    // 逆行列を計算
    linalg::Matrix A_inv = linalg::Matrix::inverse(A);

    std::cout << "\nInverse of A:\n";
    print(A_inv);

    // 検算① A * A^-1
    std::cout << "\nA * A_inv:\n";
    print(A * A_inv);

    // 検算② A^-1 * A
    std::cout << "\nA_inv * A:\n";
    print(A_inv * A);

    return 0;
}