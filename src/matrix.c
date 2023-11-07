#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    int elementLocation = (mat->cols * row) + col;
    return mat->data[elementLocation];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    // int elementLocation = (mat->cols * row) + col;
    mat->data[(mat->cols * row) + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.

    // matrix * ptr;
    // ptr = (matrix *) malloc(sizeof(matrix));
    *mat = (matrix *) malloc(sizeof(matrix));
    if (*mat == NULL) {
        return -2;
    }
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    (*mat) -> data = (double *) calloc(rows * cols, sizeof(double));
    if ( (*mat) -> data == NULL) {
        return -2;
    }
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    (*mat) -> rows = rows;
    (*mat) -> cols = cols;
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    (*mat) -> parent = NULL;
    // 6. Set the `ref_cnt` field to 1.
    (*mat)->ref_cnt = 1;
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    
    // 8. Return 0 upon success.
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    if (mat == NULL) {
        return;
    }
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    if (mat->parent == NULL) {
        mat->ref_cnt--;
        if (mat->ref_cnt == 0) {
        free(mat->data);
        free(mat);
    }
    }
    
    else {
        deallocate_matrix(mat->parent);
        free(mat);
    }
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    *mat = (matrix *) malloc(sizeof(matrix));
    if (*mat == NULL) {
        return -2;
    }
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    (*mat) -> data = (from -> data) + offset;
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    (*mat) -> rows = rows;
    (*mat) -> cols = cols;
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    (*mat) -> parent = from;
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    from -> ref_cnt = (from -> ref_cnt) + 1;
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO
    int numElements = (mat -> rows) * (mat -> cols);
    int upperBound = numElements/16*16;

    __m256d valVector = _mm256_set1_pd(val);
    #pragma omp parallel for if (numElements > 100)
    for (int i = 0; i < upperBound; i+=16) {
        // mat->data[i] = val;
        // (double *a, __m256d b)
        _mm256_storeu_pd( (mat->data)+i, valVector);
        _mm256_storeu_pd( (mat->data)+i+4, valVector);
        _mm256_storeu_pd( (mat->data)+i+8, valVector);
        _mm256_storeu_pd( (mat->data)+i+12, valVector);
    }
    #pragma omp for
    for (int i = upperBound; i < numElements; i++) {
        mat->data[i] = val;
    }
    
    return;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    double * resData = result->data;
    double * matData = mat->data;
    int size = (result->rows) * (result->cols);
    int upperBound = size/16*16;
    __m256d neg1Vector = _mm256_set1_pd(-1);

    #pragma omp parallel for if (size > 100)
    for (int i = 0; i < upperBound; i+=16) {
        __m256d m1Vector0 = _mm256_loadu_pd(matData + i);
        __m256d mulVector0 = _mm256_mul_pd(m1Vector0,neg1Vector);
        __m256d absVector0 = _mm256_max_pd(m1Vector0,mulVector0);
        _mm256_storeu_pd(resData + i, absVector0);

        __m256d m1Vector1 = _mm256_loadu_pd(matData + i+4);
        __m256d mulVector1 = _mm256_mul_pd(m1Vector1,neg1Vector);
        __m256d absVector1 = _mm256_max_pd(m1Vector1,mulVector1);
        _mm256_storeu_pd(resData + i + 4, absVector1);

        __m256d m1Vector2 = _mm256_loadu_pd(matData + i+8);
        __m256d mulVector2 = _mm256_mul_pd(m1Vector2,neg1Vector);
        __m256d absVector2 = _mm256_max_pd(m1Vector2,mulVector2);
        _mm256_storeu_pd(resData + i + 8, absVector2);

        __m256d m1Vector3 = _mm256_loadu_pd(matData + i+12);
        __m256d mulVector3 = _mm256_mul_pd(m1Vector3,neg1Vector);
        __m256d absVector3 = _mm256_max_pd(m1Vector3,mulVector3);
        _mm256_storeu_pd(resData + i + 12, absVector3);


    }
    #pragma omp parallel for
    for (int i = upperBound; i < size; i++) {
        resData[i] = fabs(mat->data[i]);
    }
    return 0;



// Given any number x (positive or negative), we know the absolute value is 
// either x or -x. Can we use some SIMD instructions to first find -x and t
// hen use those two values to figure out what abs(x) is?

    // int size = (result->rows) * (result->cols);
    // for (int i = 0; i < size; i++) {
    //     double newValue = fabs(mat->data[i]);
    //     result -> data[i] = newValue;
    // }
    // return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    //parallel for if(resultCols > 100)

    double * resData = (result->data);
    double * m1Data = (mat1->data);
    double * m2Data = (mat2->data);
    int size = (result->cols) * (result->rows);
    int upperBound = size/16*16;
    #pragma omp parallel for if (size > 100)
    for (int i = 0; i < upperBound; i+=16) {

        __m256d m1_0 = _mm256_loadu_pd(m1Data+i);
        __m256d m2_0 = _mm256_loadu_pd(m2Data+i);
        __m256d temp_0 = _mm256_add_pd(m1_0,m2_0);
        _mm256_storeu_pd(resData+i,temp_0);

        __m256d m1_1 = _mm256_loadu_pd(m1Data+i+4);
        __m256d m2_1 = _mm256_loadu_pd(m2Data+i+4);
        __m256d temp_1 = _mm256_add_pd(m1_1,m2_1);
        _mm256_storeu_pd(resData+i + 4,temp_1);

        __m256d m1_2 = _mm256_loadu_pd(m1Data+i+8);
        __m256d m2_2 = _mm256_loadu_pd(m2Data+i+8);
        __m256d temp_2 = _mm256_add_pd(m1_2,m2_2);
        _mm256_storeu_pd(resData+i+8,temp_2);

        __m256d m1_3 = _mm256_loadu_pd(m1Data+i+12);
        __m256d m2_3 = _mm256_loadu_pd(m2Data+i+12);
        __m256d temp_3 = _mm256_add_pd(m1_3,m2_3);
        _mm256_storeu_pd(resData+i + 12,temp_3);


    }
    #pragma omp parallel for
    for (int i = upperBound; i < size; i++) {
        result -> data[i] = (mat1 -> data[i]) + (mat2 -> data[i]);
    }
    return 0;

    // for (int i = 0; i < size; i++) {
    //     double sum = (mat1 -> data[i]) + (mat2 -> data[i]);
    //     result -> data[i] = sum;
    // }
    // return 0;

}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    return 0;
}

void transpose(matrix * transposedMatrix, matrix * baseMatrix) {
    double * tData = transposedMatrix->data;
    double * mData = baseMatrix->data;
    int numRows = baseMatrix -> rows;
    int numCols = baseMatrix -> cols;
    int tRows = transposedMatrix->rows;
    int tCols = transposedMatrix->cols;
    int numElements = numCols * numRows;

    #pragma omp parallel for if (numElements > 100)
    for (int col = 0; col < numCols; col++) {
        for (int row = 0; row < numRows; row++) {
            int baseLocation = (numCols * row) + col;
            int tLocation = (tRows * col) + row;
            tData[tLocation] = mData[baseLocation];
        }
    }
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */

 int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    
    int rCols = result->cols;
    int rRows = result->rows;
    int m1Rows = mat1 -> rows;
    int m1Cols = mat1 -> cols;
    int m2Rows = mat2 -> rows;
    int m2Cols = mat2 -> cols;

    int numElements = rCols * rRows;

    matrix * temp;
    allocate_matrix(&temp,m2Rows,m2Cols);
    transpose(temp, mat2);

    int tRows = mat2->cols;
    int tCols = mat2->rows;

    double * m1Data = mat1->data;
    double * tempData = temp->data;
    double * resData = result->data;
    int total;
    int upperBound = m1Cols/16*16;

    #pragma omp parallel for if (numElements > 100)
    for (int row = 0; row < m1Rows; row++) {
        for (int col = 0; col < tRows; col++) {
            int location = col + (row * rCols);
            resData[location] = 0;
            int m1 = row*m1Cols;
            int t = col*tCols;
            double mulArray[4];
            __m256d resultVector = _mm256_set1_pd(0);
            for(int offset = 0; offset < upperBound; offset+=16) {
                // resData[location] += m1Data[m1+offset] * tempData[t+offset];

                __m256d m1Vector0 = _mm256_loadu_pd(m1Data + m1 + offset);
                __m256d tVector0 = _mm256_loadu_pd(tempData + t + offset);
                resultVector = _mm256_fmadd_pd(m1Vector0,tVector0,resultVector);

                __m256d m1Vector1 = _mm256_loadu_pd(m1Data + m1 + offset + 4);
                __m256d tVector1 = _mm256_loadu_pd(tempData + t + offset + 4);
                resultVector = _mm256_fmadd_pd(m1Vector1,tVector1,resultVector);

                __m256d m1Vector2 = _mm256_loadu_pd(m1Data + m1 + offset + 8);
                __m256d tVector2 = _mm256_loadu_pd(tempData + t + offset + 8);
                resultVector = _mm256_fmadd_pd(m1Vector2,tVector2,resultVector);

                __m256d m1Vector3 = _mm256_loadu_pd(m1Data + m1 + offset + 12);
                __m256d tVector3 = _mm256_loadu_pd(tempData + t + offset + 12);
                resultVector = _mm256_fmadd_pd(m1Vector3,tVector3,resultVector);

                // __m256d mulVector = _mm256_mul_pd(m1Vector,tVector);
                // _mm_storeu_si128((__m1 *) mulArray, mulVector);
                // resData[location] += m1Data[m1+offset] * tempData[t+offset];
            }
            // _mm256_storeu_pd(mulArray, resultVector);
            resData[location] += resultVector[0] + resultVector[1] + resultVector[2] + resultVector[3];

            for(int offset = upperBound; offset < m1Cols; offset++) {
                resData[location] += m1Data[m1+offset] * tempData[t+offset];
            }
        }
    }
    free(temp);
    return 0;
 }

// int mul_matrix_x(matrix *result, matrix *mat1, matrix *mat2) {
//     // Task 1.6 TODO
//     int resCols = result->cols;
//     int mat1Rows = mat1 -> rows;
//     int mat1Cols = mat1 -> cols;
//     int mat2Rows = mat2 -> rows;
//     int mat2Cols = mat2 -> cols;

//     for (int row = 0; row < mat1Rows; row++) {
//         for (int col = 0; col < mat2Cols; col++) {
//             result->data[col + (row * resCols)] = 0;
//             for (int offset = 0; offset < mat2Rows; offset++) {
//                 (result->data)[col + (row * resCols)] += 
//                 (mat1->data)[offset + (row * mat1Cols)] * (mat2->data)[col + (offset * mat2Cols)];
//             }
//         }
//     }
//     return 0;
// }

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    int numRows = result->rows;
    int numCols = result->cols;
    int numElements ;

    double * rData = result->data;
    double * mData = mat->data;
    int size = numRows * numRows * sizeof(double);

    
    
    // #pragma omp parallel for
    for (int col = 0; col < numCols; col++) {
        for (int row = 0; row < numRows; row++) {
            int location = numCols * row + col;
            if (col==row) {
                rData[location] = 1;
            }
            else {
                rData[location] = 0;
            }
        }
    }
    if (pow == 0) {
        return 0;
    }
    // memcpy(rData,mData,size);

    matrix * temp;
    allocate_matrix(&temp,numRows,numRows);
    double * tData = temp->data;
    matrix * updated;
    allocate_matrix(&updated,numRows,numRows);
    double * uData = updated->data;
    memcpy(uData,mData,size);


    while (pow > 0) {
        if (pow%2 == 1) {
            mul_matrix(temp,result,updated);
            memcpy(rData,tData,size);
            pow-=1;
        }
        else {
        mul_matrix(temp,updated,updated);
        memcpy(uData,tData,size);
        pow = pow/2;
        }
    }



    // while (pow > 0) {
    //     if (pow%2 == 0) {
    //         pow = pow/2;
    //         mul_matrix(temp,result,result);
    //         memcpy(rData,tData,size);
    //     }
    //     else {
    //         mul_matrix(temp,result,mat);
    //         memcpy(rData,tData,size);
    //         pow = (pow-1);
    //     }
    // }

    deallocate_matrix(temp);
    deallocate_matrix(updated);
    return 0;


    // else if n = 0  then return  1;
    // else if n is even  then return exp_by_squaring(x * x,  n / 2);
    // else if n is odd  then return x * exp_by_squaring(x * x, (n - 1) / 2);
}

       
    // else {
    //     matrix * temp;
    //     allocate_matrix(&temp,numRows,numRows);

    //     mul_matrix(temp, mat, mat);
    //     memcpy(resData, (temp)->data, size);

    //     while (pow > 0) {
    //         if (pow%2 = 1) {

    //         }

    //         mul_matrix(temp,result,result);
    //         memcpy(resData, (temp)->data, size);
    //         pow = pow/2;
    //     }
    //     deallocate_matrix(temp);
    // }
//     return 0;
// }
    