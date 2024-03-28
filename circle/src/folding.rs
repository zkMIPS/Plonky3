use itertools::{izip, Itertools};
use p3_field::{batch_multiplicative_inverse, extension::ComplexExtendable, ExtensionField};
use p3_util::log2_strict_usize;

use crate::domain::CircleDomain;

pub(crate) fn fold_bivariate<F: ComplexExtendable, EF: ExtensionField<F>>(
    evals: Vec<EF>,
    beta: EF,
) -> Vec<EF> {
    let domain = CircleDomain::standard(log2_strict_usize(evals.len()));
    fold(
        evals,
        beta,
        &batch_multiplicative_inverse(&domain.points().map(|p| p.imag()).collect_vec()),
    )
}

pub(crate) fn fold_univariate<F: ComplexExtendable, EF: ExtensionField<F>>(
    evals: Vec<EF>,
    beta: EF,
) -> Vec<EF> {
    let domain = CircleDomain::standard(log2_strict_usize(evals.len()) + 1);
    fold(
        evals,
        beta,
        &batch_multiplicative_inverse(&domain.points().map(|p| p.real()).collect_vec()),
    )
}

fn fold<F: ComplexExtendable, EF: ExtensionField<F>>(
    evals: Vec<EF>,
    beta: EF,
    twiddles: &[F],
) -> Vec<EF> {
    let n = evals.len();
    let half_n = n >> 1;

    let (los, his) = evals.split_at(half_n);
    izip!(twiddles, los, his.iter().rev())
        .map(|(&t, &lo, &hi)| {
            let sum = lo + hi;
            let diff = (lo - hi) * t;
            (sum + beta * diff).halve()
        })
        .collect_vec()
}
