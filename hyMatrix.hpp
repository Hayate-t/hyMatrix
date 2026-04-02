#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <utility>
#include <stdexcept>


namespace linalg {

    //行列
    class Matrix {
        private:

        //行列は内部で1次元ベクトルとして保存され、行数と列数によって要素が参照される
        std::vector<double> data_;

        public:

        uint32_t rows;  //行数
        uint32_t cols;  //列数

        Matrix(int32_t _rows, int32_t _cols) : rows(_rows), cols(_cols), data_(_rows * _cols, 0.0) {}

        //値の代入
        double& operator()(size_t i, size_t j) {
            if (i >= rows || j >= cols) {
                throw std::out_of_range("Index out of range");
            }
            return data_[i * cols + j];
        }

        //値の参照
        double operator()(size_t i, size_t j) const {
            if (i >= rows || j >= cols) {
                throw std::out_of_range("Index out of range");
            }
            return data_[i * cols + j];
        }

        //正方行列か
        bool isSquared() const {
            return (rows == cols);
        }

        //n次の単位行列を生成
        static Matrix identity(size_t n) {
            Matrix I(n, n);
            for (size_t i = 0; i < n; i++) {
                I(i, i) = 1;
            }
            return I;
        }

        //転置行列を生成
        Matrix transpose() const {
            Matrix T(cols, rows);
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    T(j, i) = (*this)(i, j);
                }
            }
            return T;
        }

        //加算
        Matrix operator+(const Matrix& B) const {
            if (rows != B.rows || cols != B.cols) {
                throw std::runtime_error("Size mismatch");
            }

            Matrix X(rows, cols);

            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    X(i, j) = (*this)(i, j) + B(i, j);
                }
            }

            return X;
        }

        //減算
        Matrix operator-(const Matrix& B) const {
            if (rows != B.rows || cols != B.cols) {
                throw std::runtime_error("Size mismatch");
            }

            Matrix X(rows, cols);

            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    X(i, j) = (*this)(i, j) - B(i, j);
                }
            }

            return X;
        }

        //行列積
        Matrix operator*(const Matrix& B) const {
            if (cols != B.rows) {
                throw std::runtime_error("Size mismatch");
            }

            Matrix X(rows, B.cols);

            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < B.cols; j++) {
                    double inner_product = 0;
                    for (size_t k = 0; k < cols; k++) {
                        inner_product += (*this)(i, k) * B(k, j);
                    }
                    X(i, j) = inner_product;
                }
            }

            return X;
        }

        //前進消去
        static Matrix forward_elimination(const Matrix& A) {

            Matrix X = A;

            size_t n = X.rows;  //行列の行数
            size_t m = X.cols;  //行列の列数

            size_t l = 0;   //ピボットの行の進行
            //変数kはピボットの列の進行を表す
            for (size_t k = 0; k < m && l < n; k++) {

                //k列の中で絶対値が最大の行を選びピボットを探す
                size_t pivot = l;
                for (size_t i = l; i < n; i++) {
                    if (std::abs(X(i, k)) > std::abs(X(pivot, k))) {
                        pivot = i;
                    }
                }

                //ピボット行の成分が0（->列の全ての成分が0）なら次の列へ
                if (std::abs(X(pivot, k)) < 1e-10) continue;

                //行を交換
                if (pivot != l) {
                    for (size_t j = 0; j < X.cols; j++) {
                        std::swap(X(l, j), X(pivot, j));
                    }
                }

                //前進消去
                //変数iは行、変数jは列の進行を表す
                for (size_t i = l + 1; i < n; i++) {
                    //(i, k)を0にするための係数を求める
                    double factor = X(i, k) / X(l, k);

                    //右へ引いていく
                    for (size_t j = k; j < m; j++) {
                        X(i, j) -= factor * X(l, j);
                    }
                }

                //列の全ての成分が0でないときに、lを進める
                l++;
            }

            return X;
        }

        //後退代入（引数には前進消去後の拡大係数行列を用いる）
        static Matrix back_substitution(const Matrix& Ab) {
            size_t n = Ab.rows;
            Matrix X(n, 1);

            //size_tのオーバーフローで値が大きくならないようintを使う
            for (int32_t i = n - 1; i >= 0; i--) {
                double sum = 0.0;

                for (int32_t j = i + 1; j < n; j++) {
                    sum += Ab(i, j) * X(j, 0);
                }

                X(i, 0) = (Ab(i, n) - sum) / Ab(i, i);
            }

            return X;
        }

        //階数
        //Aが既に上三角行列のときにはisUpperTriangularをtrueにすることで前進消去を省略可
        static size_t rank(const Matrix& A, bool isUpperTriangular = false) {

            Matrix U = (isUpperTriangular) ? A : forward_elimination(A);

            size_t n = U.rows;
            size_t m = U.cols;
            size_t rank = 0;

            for (size_t i = 0; i < U.rows; i++) {
                for (size_t j = i; j < U.cols; j++) {
                    if (std::abs(U(i, j)) > 1e-10) {
                        rank++;
                        break;
                    }
                }
            }

            return rank;
        }

        //連立方程式
        static Matrix solve(const Matrix& A, const Matrix& b) {
            
            if (!A.isSquared()) {
                throw std::runtime_error("Matrix is not Squared");
            }
            if (b.cols != 1) {
                throw std::runtime_error("It is not a Vector");
            }
            if (A.rows != b.rows) {
                throw std::runtime_error("Size mismatch");
            }
            size_t n = A.rows;

            //拡大係数行列を生成
            Matrix Ab(n, n + 1);
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    Ab(i, j) = A(i, j);
                }
                Ab(i, n) = b(i, 0);
            }

            //前進消去
            Matrix U = forward_elimination(Ab);

            //階数から方程式に解が存在するか、解が一意に定まるか判断する
            size_t U_rank = rank(U, true);
            if (U_rank != rank(A, false)) {
                throw std::runtime_error("No solution");
            }
            if (U_rank != n) {
                throw std::runtime_error("Not unique solution");
            } 

            //後退代入
            Matrix X = back_substitution(U);

            return X;
        }

        //std::vectorを用いた連立方程式の計算
        static std::vector<double> solve(const Matrix& A, const std::vector<double>& v) {
            if (A.rows != v.size()) {
                throw std::runtime_error("Size mismatch");
            }
            
            Matrix b(A.rows, 1);
            for (size_t i = 0; i < v.size(); i++) {
                b(i, 0) = v[i];
            }

            Matrix X = solve(A, b);

            std::vector<double> x(X.rows);
            for (size_t i = 0; i < X.rows; i++) {
                x[i] = X(i, 0);
            }
            return x;
        }

        //逆行列
        static Matrix inverse(const Matrix& A) {
            //正方行列でないとエラー
            if (!A.isSquared()) {
                throw std::runtime_error("Matrix is not square");
            }

            //// Gauss-Jordan法による逆行列の生成 ////

            size_t n = A.rows;

            //対象となる行列に単位行列を追加した拡大行列を生成
            Matrix Aug(n, n * 2);
            for(size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    Aug(i, j) = A(i, j);
                }
                for (size_t j = 0; j < n; j++) {
                    Aug(i, n + j) = (i == j) ? 1.0 : 0.0;
                }
            }

            //前進消去
            Aug = forward_elimination(Aug);

            //後方から単位行列にする処理
            //後退処理はオーバーフロー防止のためsize_tではなくint32_tを用いる
            for (int32_t i = n - 1; i >= 0; i--) {
                double pivot = Aug(i, i);

                //ピボットが既に0であるとき、正則でないためエラー
                if (std::abs(pivot) < 1e-10) {
                    throw std::runtime_error("Matrix is singular");
                }

                //ピボットを1にするため、行全体をピボットで割る
                for (size_t j = 0; j < 2 * n; j++) {
                    Aug(i, j) /= pivot;
                }

                //ピボット列の上側を0にする
                for (int32_t k = i - 1; k >= 0; k--) {
                    //ピボットは既に1であるため、係数は成分そのものになる
                    double factor = Aug(k, i);
                    //行全体を引いていく
                    for (size_t j = 0; j < 2 * n; j++) {
                        Aug(k, j) -= factor * Aug(i, j);
                    }
                }
            }

            //拡大行列の右半分から、生成した逆行列を取り出す
            Matrix inv(n, n);
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    inv(i, j) = Aug(i, n + j);
                }
            }

            return inv;
        }
    }; //class Matrix

    //スカラーと行列の積
    Matrix operator*(double k, const Matrix& M) {
        Matrix X(M.rows, M.cols);
        for (size_t i = 0; i < M.rows; i++) {
            for (size_t j = 0; j < M.cols; j++) {
                X(i, j) = M(i, j) * k;
            }
        }

        return X;
    }

    //行列とスカラーの積（同上）
    Matrix operator*(const Matrix& M, double k) {
        return k * M;
    }

    //ベクトル
    //ベクトルの成分数は、固定サイズテンプレートで指定する。
    template<size_t N> 
    class Vector{
        private:

        //std::vectorではなくstd::arrayを用いている。（resize()等を行わないため）
        std::array<double, N> data_;

        public:

        //通常コンストラクタ
        Vector() {
            data_.fill(0.0);
        }

        //リストで定義する場合のコンストラクタ
        Vector(std::initializer_list<double> init) {
            if (init.size() != N) {
                throw std::runtime_error("Size mismatch");
            }
            std::copy(init.begin(), init.end(), data_.begin());
        }

        //値の代入
        double& operator[](size_t i) {
            return data_[i];
        }

        //値の参照
        double operator[](size_t i) const {
            return data_[i];
        }

        //加算
        Vector operator+(const Vector& w) const {
            Vector x;
            for (size_t i = 0; i < N; i++) {
                x[i] = data_[i] + w[i];
            }
            return x;
        }

        //減算
        Vector operator-(const Vector& w) const {
            Vector x;
            for (size_t i = 0; i < N; i++) {
                x[i] = data_[i] - w[i];
            }
            return x;
        }
        
        //スカラー積（左がベクトルの場合）
        Vector operator*(double k) const {
            Vector x;
            for (size_t i = 0; i < N; i++) {
                x[i] = data_[i] * k;
            }
            return x;
        }

        //内積
        double dot(const Vector& w) const {
            double sum = 0.0;
            for (size_t i = 0; i < N; i++) {
                sum += data_[i] * w[i];
            }
            return sum;
        }

        //ノルム（長さ）
        double norm() const {
            //自分自身との内積をとることで、各成分の2乗の和が計算される。これはノルムの2乗に等しい。
            return std::sqrt(dot(*this));
        }

        //正規化（単位ベクトル化）
        Vector normalized() const {
            double n = norm();

            if (n < 1e-10) {
                throw std::runtime_error("Can't normalize zero vector");
            }

            return (*this) * (1.0 / n);
        }

        //外積
        static auto cross(const Vector& v, const Vector& w) {
            if constexpr (N == 3) {
                //3次元ベクトルの際の外積
                return Vector{
                    v[1] * w[2] - v[2] * w[1],
                    v[2] * w[0] - v[0] * w[2],
                    v[0] * w[1] - v[1] * w[0]
                };
            } else if constexpr (N == 2) {
                //2次元ベクトルのときは符号付面積を返す
                return v[0] * w[1] - v[1] * w[0];
            } else {
                static_assert(N == 2 || N == 3, "Cross only for 2D or 3D");
            }
        }
    }; //class Vector

    //ベクトルのスカラー積（ベクトルが右の場合）
    template<size_t N>
    Vector<N> operator*(double k, const Vector<N>& v) {
        return v * k;
    }

    //行列とベクトルの積
    //行列が正方行列のとき、つまりベクトルの次数が変わらないときに限り計算可能
    template<size_t N>
    Vector<N> operator*(const Matrix& A, const Vector<N>& v) {
        if (A.cols != N || A.rows != N) {
            throw std::runtime_error("Size mismatch");
        }

        Vector<N> x;

        for (size_t i = 0; i < A.rows; i++) {
            //内積をとる
            double sum = 0.0;
            for (size_t j = 0; j < A.cols; j++) {
                sum += A(i, j) * v[j];
            }
            x[i] = sum;
        }

        return x;
    }

    //Vector2,Vector3を特殊なクラスとして別で定義する（利便性向上）
    using Vector2 = Vector<2>;
    using Vector3 = Vector<3>;
    

} //namespace linalg