/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include "blas.hpp"

// // #include <cublas_v2.h>
extern "C" {
#include <cblas.h>
}

#include <stdexcept>


namespace helpers {
namespace blas {

    template <>
    void multiplyMatrices<float>(
        bool transposeA, bool transposeB,
        int m, int n, int k,
        const float *matrixA, int ldA,
        const float *matrixB, int ldB,
        float *matrixC, int ldC,
        bool addOldMatrixC
        )
    {
        float alpha = 1;
        float beta  = (addOldMatrixC ? 1.0f : 0.0f);

        cblas_sgemm(
            /* order  */ CblasColMajor,
            /* transa */ transposeA ? CblasTrans : CblasNoTrans,
            /* transb */ transposeB ? CblasTrans : CblasNoTrans,
            /* m      */ m,
            /* n      */ n,
            /* k      */ k,
            /* alpha  */ alpha,
            /* A      */ matrixA,
            /* lda    */ ldA,
            /* B      */ matrixB,
            /* ldb    */ ldB,
            /* beta   */ beta,
            /* C      */ matrixC,
            /* ldc    */ ldC
        );
        // if (res != CUBLAS_STATUS_SUCCESS)
        //     throw std::runtime_error("CUBLAS matrix multiplication failed");
    }

    template <>
    void multiplyMatrices<double>(
        bool transposeA, bool transposeB,
        int m, int n, int k,
        const double *matrixA, int ldA,
        const double *matrixB, int ldB,
        double *matrixC, int ldC,
        bool addOldMatrixC
        )
    {
        double alpha = 1;
        double beta  = (addOldMatrixC ? 1 : 0);

        cblas_dgemm(
            /* order  */ CblasColMajor,
            /* transa */ transposeA ? CblasTrans : CblasNoTrans,
            /* transb */ transposeB ? CblasTrans : CblasNoTrans,
            /* m      */ m,
            /* n      */ n,
            /* k      */ k,
            /* alpha  */ alpha,
            /* A      */ matrixA,
            /* lda    */ ldA,
            /* B      */ matrixB,
            /* ldb    */ ldB,
            /* beta   */ beta,
            /* C      */ matrixC,
            /* ldc    */ ldC
        );

    }

} // namespace cublas
} // namespace helpers
