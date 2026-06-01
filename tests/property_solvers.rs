#[cfg(feature = "parallel")]
use kavg_lab::parallel::parse_execution_mode;
use kavg_lab::parallel::{map_indexed, ExecutionMode};
use proptest::prelude::*;

proptest! {
    #[test]
    fn sequential_map_indexed_preserves_order(values in proptest::collection::vec(-100i32..100, 0..64)) {
        let mapped = map_indexed(&values, ExecutionMode::Sequential, |index, value| {
            Ok((index, *value))
        }).unwrap();
        let expected: Vec<(usize, i32)> = values.iter().copied().enumerate().collect();
        prop_assert_eq!(mapped, expected);
    }
}

#[cfg(feature = "parallel")]
proptest! {
    #[test]
    fn parallel_map_indexed_matches_sequential(values in proptest::collection::vec(-100i32..100, 0..64)) {
        let sequential = map_indexed(&values, ExecutionMode::Sequential, |index, value| {
            Ok((index, *value * 2))
        }).unwrap();
        let parallel_mode = parse_execution_mode(true, "auto").unwrap();
        let parallel = map_indexed(&values, parallel_mode, |index, value| {
            Ok((index, *value * 2))
        }).unwrap();
        prop_assert_eq!(parallel, sequential);
    }
}
