use kavg_lab::optimization::projections::{
    masked_entmax15, project_to_simplex, top_k_masked_sparsemax,
};
use proptest::prelude::*;

fn approx_sum_one(values: &[f64], tolerance: f64) -> bool {
    (values.iter().sum::<f64>() - 1.0).abs() <= tolerance
}

proptest! {
    #[test]
    fn simplex_projection_returns_valid_distribution(values in proptest::collection::vec(-10.0f64..10.0, 1..32)) {
        let projected = project_to_simplex(&values);
        prop_assert_eq!(projected.len(), values.len());
        prop_assert!(projected.iter().all(|value| *value >= -1.0e-10));
        prop_assert!(approx_sum_one(&projected, 1.0e-8));
    }

    #[test]
    fn entmax15_returns_valid_distribution(scores in proptest::collection::vec(-8.0f64..8.0, 1..32)) {
        let allowed = vec![true; scores.len()];
        let weights = masked_entmax15(&scores, &allowed);
        prop_assert_eq!(weights.len(), scores.len());
        prop_assert!(weights.iter().all(|value| *value >= -1.0e-10));
        prop_assert!(approx_sum_one(&weights, 1.0e-7));
    }

    #[test]
    fn topk_attention_keeps_at_most_k_entries(scores in proptest::collection::vec(-8.0f64..8.0, 2..32), k in 1usize..8) {
        let allowed = vec![true; scores.len()];
        let effective_k = k.min(scores.len());
        let weights = top_k_masked_sparsemax(&scores, &allowed, effective_k);
        let positive = weights.iter().filter(|value| **value > 1.0e-10).count();
        prop_assert!(positive <= effective_k);
        prop_assert!(weights.iter().all(|value| *value >= -1.0e-10));
        prop_assert!(approx_sum_one(&weights, 1.0e-8));
    }
}
