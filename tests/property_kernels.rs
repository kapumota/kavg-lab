use kavg_lab::kernels::{KernelFunction, MahalanobisKernel, SquaredNormKernel};
use proptest::prelude::*;

proptest! {
    #[test]
    fn squared_norm_kernel_is_nonnegative(values in proptest::collection::vec(-100.0f64..100.0, 1..32)) {
        let kernel = SquaredNormKernel;
        prop_assert!(kernel.value(&values) >= -1.0e-12);
    }

    #[test]
    fn diagonal_mahalanobis_kernel_is_nonnegative(values in proptest::collection::vec(-50.0f64..50.0, 1..16)) {
        let dimension = values.len();
        let mut matrix = vec![vec![0.0; dimension]; dimension];
        for (index, row) in matrix.iter_mut().enumerate() {
            row[index] = 1.0 + index as f64;
        }
        let kernel = MahalanobisKernel::new(matrix).unwrap();
        prop_assert!(kernel.value(&values) >= -1.0e-10);
    }
}
