#include <iostream>
#include "hyMatrix.hpp"

// 行列表示用関数
void print(const linalg::Matrix& M) {
    std::cout << "[\n";
    for (size_t i = 0; i < M.rows; i++) {
        std::cout << "  ";
        for (size_t j = 0; j < M.cols; j++) {
            std::cout << M(i, j) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "]\n";
}

void printVector(const linalg::Vector2& v) {
    std::cout << "[" << v[0] << ", " << v[1] << "]" << std::endl;
}

void printVector(const linalg::Vector3& v) {
    std::cout << "[" << v[0] << ", " << v[1] << ", " << v[2] << "]" << std::endl;
}
 
int main() {

    std::cout << "=== Basic Matrix Operations ===\n";

    // 行列の生成
    linalg::Matrix A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;

    linalg::Matrix B(2, 2);
    B(0, 0) = 5; B(0, 1) = 6;
    B(1, 0) = 7; B(1, 1) = 8;

    std::cout << "A:\n"; print(A);
    std::cout << "B:\n"; print(B);

    // 加算
    std::cout << "\nA + B:\n";
    print(A + B);

    // 減算
    std::cout << "\nA - B:\n";
    print(A - B);

    // 行列積
    std::cout << "\nA * B:\n";
    print(A * B);

    // スカラー倍
    std::cout << "\n2 * A:\n";
    print(2.0 * A);

    // 転置
    std::cout << "\nTranspose of A:\n";
    print(A.transpose());


    std::cout << "\n=== Solve Linear System Ax = b ===\n";

    // 連立方程式
    linalg::Matrix C(3, 3);
    C(0, 0) = 2;  C(0, 1) = -1; C(0, 2) = 0;
    C(1, 0) = -1; C(1, 1) = 2;  C(1, 2) = -1;
    C(2, 0) = 0;  C(2, 1) = -1; C(2, 2) = 2;

    linalg::Matrix b(3, 1);
    b(0, 0) = 1;
    b(1, 0) = 0;
    b(2, 0) = 1;

    std::cout << "Matrix C:\n"; print(C);
    std::cout << "Vector b:\n"; print(b);

    // 解を求める
    linalg::Matrix x = linalg::Matrix::solve(C, b);

    std::cout << "\nSolution x:\n";
    print(x);

    // 検算 Cx
    std::cout << "\nCheck C * x:\n";
    print(C * x);


    std::cout << "\n=== Inverse Matrix ===\n";

    // 逆行列
    linalg::Matrix C_inv = linalg::Matrix::inverse(C);

    std::cout << "C inverse:\n";
    print(C_inv);

    // 検算 C * C^-1
    std::cout << "\nC * C_inv:\n";
    print(C * C_inv);

    std::cout << "\nC_inv * C:\n";
    print(C_inv * C);


    std::cout << "\n=== Solve using std::vector ===\n";

    // vector版
    std::vector<double> v = {1, 0, 1};
    std::vector<double> sol = linalg::Matrix::solve(C, v);

    std::cout << "Solution (std::vector):\n";
    for (double val : sol) {
        std::cout << val << "\n";
    }

    std::cout << "===== Vector Test =====" << std::endl;

    // ===== Vector2 =====
    linalg::Vector2 v2a{1.0, 2.0};
    linalg::Vector2 v2b{3.0, 4.0};

    std::cout << "\n-- Vector2 basic --" << std::endl;
    printVector(v2a);
    printVector(v2b);

    // 加算
    std::cout << "\n-- addition --" << std::endl;
    auto v2_add = v2a + v2b;
    printVector(v2_add);

    // 減算
    std::cout << "\n-- subtraction --" << std::endl;
    auto v2_sub = v2a - v2b;
    printVector(v2_sub);

    // スカラー倍
    std::cout << "\n-- scalar multiply ([1,2]x2, 2x[1,2]) --" << std::endl;
    auto v2_scale = v2a * 2.0;
    printVector(v2_scale);

    auto v2_scale2 = 2.0 * v2a;
    printVector(v2_scale2);

    // 内積
    std::cout << "\n-- dot product --" << std::endl;
    std::cout << v2a.dot(v2b) << std::endl;

    // ノルム
    std::cout << "\n-- norm --" << std::endl;
    std::cout << v2a.norm() << std::endl;

    // 外積（2D）
    std::cout << "\n-- cross (2D) --" << std::endl;
    std::cout << linalg::Vector2::cross(v2a, v2b) << std::endl;

    // ===== Vector3 =====
    linalg::Vector3 v3a{1.0, 0.0, 0.0};
    linalg::Vector3 v3b{0.0, 1.0, 0.0};

    std::cout << "\n-- Vector3 basic --" << std::endl;
    printVector(v3a);
    printVector(v3b);

    // 加算
    std::cout << "\n-- addition --" << std::endl;
    auto v3_add = v3a + v3b;
    printVector(v3_add);

    // 内積
    std::cout << "\n-- dot product --" << std::endl;
    std::cout << v3a.dot(v3b) << std::endl;

    // ノルム
    std::cout << "\n-- norm --" << std::endl;
    std::cout << v3a.norm() << std::endl;

    // 外積（3D）
    std::cout << "\n-- cross (3D) --" << std::endl;
    auto v3_cross = linalg::Vector3::cross(v3a, v3b);
    printVector(v3_cross);

    std::cout << "\n===== Test End =====" << std::endl;

    return 0;
}