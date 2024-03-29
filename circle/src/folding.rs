use itertools::Itertools;
use p3_field::{batch_multiplicative_inverse, extension::ComplexExtendable, ExtensionField};
use p3_matrix::MatrixRows;
use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::domain::CircleDomain;

pub(crate) fn fold_bivariate<F: ComplexExtendable, EF: ExtensionField<F>>(
    evals: impl MatrixRows<EF>,
    beta: EF,
) -> Vec<EF> {
    assert_eq!(evals.width(), 2);
    let domain = CircleDomain::standard(log2_strict_usize(evals.height()) + 1);
    let mut twiddles = batch_multiplicative_inverse(
        &domain
            .points()
            .take(evals.height())
            .map(|p| p.imag())
            .collect_vec(),
    );
    twiddles = circle_bitrev_permute(&twiddles);
    fold(evals, beta, &twiddles)
}

pub(crate) fn fold_univariate<F: ComplexExtendable, EF: ExtensionField<F>>(
    evals: impl MatrixRows<EF>,
    beta: EF,
) -> Vec<EF> {
    assert_eq!(evals.width(), 2);
    let domain = CircleDomain::standard(log2_strict_usize(evals.height()) + 2);
    let mut twiddles = batch_multiplicative_inverse(
        &domain
            .points()
            .take(evals.height())
            .map(|p| p.real())
            .collect_vec(),
    );
    twiddles = circle_bitrev_permute(&twiddles);
    fold(evals, beta, &twiddles)
}

fn fold<F: ComplexExtendable, EF: ExtensionField<F>>(
    evals: impl MatrixRows<EF>,
    beta: EF,
    twiddles: &[F],
) -> Vec<EF> {
    evals
        .rows()
        .zip(twiddles)
        .map(|(row, &t)| {
            let mut row_iter = row.into_iter();
            let (lo, hi) = (row_iter.next().unwrap(), row_iter.next().unwrap());
            let sum = lo + hi;
            let diff = (lo - hi) * t;
            (sum + beta * diff).halve()
        })
        .collect_vec()
}

// circlebitrev -> natural
// can make faster with:
// https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
fn circle_bitrev_idx(mut idx: usize, bits: usize) -> usize {
    idx = reverse_bits_len(idx, bits);
    for i in 0..bits {
        if idx & (1 << i) != 0 {
            idx ^= (1 << i) - 1;
        }
    }
    idx
}

// can do in place if use cycles? bitrev makes it harder
pub(crate) fn circle_bitrev_permute<T: Clone>(xs: &[T]) -> Vec<T> {
    let bits = log2_strict_usize(xs.len());
    (0..xs.len())
        .map(|i| xs[circle_bitrev_idx(i, bits)].clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use p3_field::{extension::BinomialExtensionField, AbstractExtensionField};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31;
    use rand::{thread_rng, Rng};

    use crate::Cfft;

    use super::*;

    #[test]
    fn test_circle_bitrev() {
        assert_eq!(circle_bitrev_permute(&[0]), &[0]);
        assert_eq!(circle_bitrev_permute(&[0, 1]), &[0, 1]);
        assert_eq!(circle_bitrev_permute(&[0, 1, 2, 3]), &[0, 3, 1, 2]);
        assert_eq!(
            circle_bitrev_permute(&[0, 1, 2, 3, 4, 5, 6, 7]),
            &[0, 7, 3, 4, 1, 6, 2, 5]
        );
    }

    fn do_test_folding(log_n: usize, log_blowup: usize) {
        dbg!(log_n, log_blowup);

        let mut rng = thread_rng();

        type F = Mersenne31;
        type EF = BinomialExtensionField<F, 3>;

        let mut evals: Vec<EF> = {
            let evals = RowMajorMatrix::<F>::rand(
                &mut rng,
                1 << log_n,
                <EF as AbstractExtensionField<F>>::D,
            );
            let lde = Cfft::default().lde(
                evals,
                CircleDomain::standard(log_n),
                CircleDomain::standard(log_n + log_blowup),
            );
            lde.rows().map(|r| EF::from_base_slice(r)).collect()
        };

        evals = circle_bitrev_permute(&evals);

        evals = fold_bivariate::<F, _>(RowMajorMatrix::new(evals, 2), rng.gen());
        for _ in log_blowup..(log_n + log_blowup - 1) {
            evals = fold_univariate::<F, _>(RowMajorMatrix::new(evals, 2), rng.gen());
        }
        assert_eq!(evals.len(), 1 << log_blowup);
        assert_eq!(
            evals,
            core::iter::repeat(evals[0]).take(evals.len()).collect_vec()
        );
    }

    #[test]
    fn test_folding() {
        do_test_folding(4, 1);
        do_test_folding(5, 2);
    }
}
