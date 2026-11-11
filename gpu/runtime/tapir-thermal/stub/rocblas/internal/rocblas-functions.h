/* Only for building without ROCm, to check that everything compiles: a few rocBLAS prototypes in the real header's style.
   build.rs uses the installed ROCm headers whenever it finds them. */
#define ROCBLAS_EXPORT
ROCBLAS_EXPORT rocblas_status rocblas_dgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m,
                                            rocblas_int n, rocblas_int k, const double* alpha, const double* A, rocblas_int lda,
                                            const double* B, rocblas_int ldb, const double* beta, double* C, rocblas_int ldc);
ROCBLAS_EXPORT rocblas_status rocblas_sgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m,
                                            rocblas_int n, rocblas_int k, const float* alpha, const float* A, rocblas_int lda,
                                            const float* B, rocblas_int ldb, const float* beta, float* C, rocblas_int ldc);
ROCBLAS_EXPORT rocblas_status rocblas_dgemm_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
                                            rocblas_int m, rocblas_int n, rocblas_int k, const double* alpha, const double* A,
                                            rocblas_int lda, rocblas_stride stride_a, const double* B, rocblas_int ldb,
                                            rocblas_stride stride_b, const double* beta, double* C, rocblas_int ldc,
                                            rocblas_stride stride_c, rocblas_int batch_count);
ROCBLAS_EXPORT rocblas_status rocblas_dtrsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA,
                                            rocblas_diagonal diag, rocblas_int m, rocblas_int n, const double* alpha, const double* A,
                                            rocblas_int lda, double* B, rocblas_int ldb);
