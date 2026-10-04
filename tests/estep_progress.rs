//! Progress events of the E-step: forward events first, then backward
//! events, `done` never decreasing within a phase and ending at `total`.

use psmc_rs::PsmcModel;
use psmc_rs::hmm::{EStepPhase, EStepProgress, e_step_streaming_with_progress};

fn check_events(events: &[EStepProgress], total: u64) {
    assert!(!events.is_empty());
    let first_backward = events
        .iter()
        .position(|e| e.phase == EStepPhase::Backward)
        .expect("no backward events");
    assert!(
        events[first_backward..]
            .iter()
            .all(|e| e.phase == EStepPhase::Backward),
        "forward event after a backward one"
    );
    for phase in [EStepPhase::Forward, EStepPhase::Backward] {
        let done: Vec<u64> = events
            .iter()
            .filter(|e| e.phase == phase)
            .map(|e| e.done)
            .collect();
        assert!(done.windows(2).all(|w| w[0] <= w[1]), "{phase:?}: {done:?}");
        assert_eq!(*done.last().unwrap(), total, "{phase:?} must end at total");
    }
    assert!(events.iter().all(|e| e.total == total));
}

#[test]
fn progress_is_monotonic_and_complete() {
    let model = PsmcModel::new(15.0, 22, 2e-3, 4e-4, 2.5e-8, Some("4+5*3+4".into())).unwrap();
    let lens = [5000usize, 1, 3000, 129, 8000];
    let rows: Vec<Vec<u8>> = lens
        .iter()
        .enumerate()
        .map(|(s, &len)| (0..len).map(|i| u8::from((i * 7 + s) % 50 == 0)).collect())
        .collect();
    let row_starts = vec![true; rows.len()];
    let total: u64 = lens.iter().map(|&l| l as u64).sum();
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut events = Vec::new();
        let mut cb = |e: EStepProgress| events.push(e);
        pool.install(|| {
            e_step_streaming_with_progress(
                model.prior_matrix(),
                model.transition_matrix(),
                model.emission_matrix(),
                &rows,
                &row_starts,
                false,
                "E",
                Some(&mut cb),
            )
            .unwrap()
        });
        check_events(&events, total);
    }
}
