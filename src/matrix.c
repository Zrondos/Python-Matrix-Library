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

/* Intel intrinsics:
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
 * Assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    int elementLocation = (mat->cols * row) + col;
    return mat->data[elementLocation];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {

    mat->data[(mat->cols * row) + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns.
 * Return 0 upon success.
 * Return -1 if either `rows` or `cols` or both have invalid values. 
 * Return -2 if any call to allocate memory in this function fails.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if (rows <= 0 || cols <= 0) {
        return -1;
    }

    // Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    *mat = (matrix *) malloc(sizeof(matrix));
    if (*mat == NULL) {
        return -2;
    }

    // Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    (*mat) -> data = (double *) calloc(rows * cols, sizeof(double));
    if ( (*mat) -> data == NULL) {
        return -2;
    }

    // Set the number of rows and columns in the matrix struct according to the arguments provided.
    (*mat) -> rows = rows;
    (*mat) -> cols = cols;

    // Set the `parent` field to NULL, since this matrix was not created from a slice.
    (*mat) -> parent = NULL;

    // Set the `ref_cnt` field to 1.
    (*mat)->ref_cnt = 1;
    
    // Return 0 upon success.
    return 0;
}

/*
 * Only free `mat->data` if `mat` is not a slice and has no existing slices.
 * Only free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {

    // If the matrix pointer `mat` is NULL, return.
    if (mat == NULL) {
        return;
    }

    // If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    if (mat->parent == NULL) {
        mat->ref_cnt--;
        if (mat->ref_cnt == 0) {
        free(mat->data);
        free(mat);
    }
    }

    // Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    else {
        deallocate_matrix(mat->parent);
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data point to the `offset`th entry of `from`'s data for the data field.
 * Return 0 upon success.
 * Return -1 if either `rows` or `cols` or both have invalid values. 
 * Return -2 if any call to allocate memory in this function fails.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if (rows <= 0 || cols <= 0) {
        return -1;
    }

    // Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    *mat = (matrix *) malloc(sizeof(matrix));
    if (*mat == NULL) {
        return -2;
    }

    // Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    (*mat) -> data = (from -> data) + offset;

    // Set the number of rows and columns in the new struct according to the arguments provided.
    (*mat) -> rows = rows;
    (*mat) -> cols = cols;

    // Set the `parent` field of the new struct to the `from` struct pointer.
    (*mat) -> parent = from;

    // Increment the `ref_cnt` field of the `from` struct by 1.
    from -> ref_cnt = (from -> ref_cnt) + 1;

    // Return 0 upon success.
    return 0;
}

/*
 * Sets all entries in mat to val. The matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Calculate the total number of elements in the matrix
    int numElements = (mat -> rows) * (mat -> cols);
    int upperBound = numElements/16*16;

    // Initialize an AVX vector that holds the value 'val' repeated four times
    __m256d valVector = _mm256_set1_pd(val);

    // Parallelize the loop to fill blocks of 16 elements at a time using AVX instructions
    #pragma omp parallel for if (numElements > 100)
    for (int i = 0; i < upperBound; i+=16) {
        _mm256_storeu_pd( (mat->data)+i, valVector);
        _mm256_storeu_pd( (mat->data)+i+4, valVector);
        _mm256_storeu_pd( (mat->data)+i+8, valVector);
        _mm256_storeu_pd( (mat->data)+i+12, valVector);
    }

    // Fill any remaining elements (less than 16) sequentially
    #pragma omp for
    for (int i = upperBound; i < numElements; i++) {
        mat->data[i] = val;
    }
    
    return;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * The matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    
    double * resData = result->data;
    double * matData = mat->data;

    // Calculate the total number of elements in the matrix
    int size = (result->rows) * (result->cols);
    int upperBound = size/16*16;

     // Initialize an AVX vector that holds the value '-1' repeated four times
    __m256d neg1Vector = _mm256_set1_pd(-1);

    // Parallelize the loop to store the asbolute value of elements 16 elements at a time using AVX instructions
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

    // Fill any remaining elements (less than 16) sequentially
    #pragma omp parallel for
    for (int i = upperBound; i < size; i++) {
        resData[i] = fabs(mat->data[i]);
    }
    return 0;


/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {

    double * resData = (result->data);
    double * m1Data = (mat1->data);
    double * m2Data = (mat2->data);

    // Calculate the total number of elements in the matrix
    int size = (result->cols) * (result->rows);
    int upperBound = size/16*16;

    // Parallelize the loop to add vectors of 16 elements at a time using AVX instructions
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

    // Fill any remaining elements (less than 16) sequentially
    #pragma omp parallel for
    for (int i = upperBound; i < size; i++) {
        result -> data[i] = (mat1 -> data[i]) + (mat2 -> data[i]);
    }
    return 0;
}

/* 
* Store the transpose of baseMatrix in transposedMatrix
*/
void transpose(matrix * transposedMatrix, matrix * baseMatrix) {

    double * tData = transposedMatrix->data;
    double * mData = baseMatrix->data;
    int numRows = baseMatrix -> rows;
    int numCols = baseMatrix -> cols;
    int tRows = transposedMatrix->rows;
    int tCols = transposedMatrix->cols;
    int numElements = numCols * numRows;

    // Parallelizes the matrix transpose operation, swapping elements from mData to tData based on their row and column indices.
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
 * Assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
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

    // Transpose mat2 for efficient cache access
    transpose(temp, mat2);

    int tRows = mat2->cols;
    int tCols = mat2->rows;

    double * m1Data = mat1->data;
    double * tempData = temp->data;
    double * resData = result->data;
    int total;
    int upperBound = m1Cols/16*16;

    // Parallelize performing matrix multiplication, 16 elements at a time
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

            }

            resData[location] += resultVector[0] + resultVector[1] + resultVector[2] + resultVector[3];

            for(int offset = upperBound; offset < m1Cols; offset++) {
                resData[location] += m1Data[m1+offset] * tempData[t+offset];
            }
        }
    }
    free(temp);
    return 0;
 }

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {

    int numRows = result->rows;
    int numCols = result->cols;
    int numElements ;

    double * rData = result->data;
    double * mData = mat->data;
    int size = numRows * numRows * sizeof(double);

    // Initialize the result matrix as the identity matrix
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

     // If power is 0, return early (identity matrix is the result for 0 power)
    if (pow == 0) {
        return 0;
    }

    matrix * temp;
    allocate_matrix(&temp,numRows,numRows);
    double * tData = temp->data;
    matrix * updated;
    allocate_matrix(&updated,numRows,numRows);
    double * uData = updated->data;
    memcpy(uData,mData,size);

    // Loop for exponentiation by squaring
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

    deallocate_matrix(temp);
    deallocate_matrix(updated);
    return 0;
}    