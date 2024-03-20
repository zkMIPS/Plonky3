use alloc::vec::Vec;
use itertools::{izip, Itertools};
use p3_field::{
    extension::{Complex, ComplexExtendable, HasFrobenius},
    AbstractField, ExtensionField, Field,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::domain::CircleDomain;

fn frob_line<F: ComplexExtendable, EF: HasFrobenius<F>>(
    point: Complex<EF>,
    value: EF,
    p: Complex<F>,
) -> EF {
    assert_ne!(point.imag(), point.imag().frobenius());
    value
        + (value.frobenius() - value) * (-point.imag() + p.imag())
            / (point.imag().frobenius() - point.imag())
}

// A vanishing polynomial on 2 circle points.
fn pair_vanishing<F: Field>(excluded0: Complex<F>, excluded1: Complex<F>, p: Complex<F>) -> F {
    // The algorithm check computes the area of the triangle formed by the
    // 3 points. This is done using the determinant of:
    // | p.x  p.y  1 |
    // | e0.x e0.y 1 |
    // | e1.x e1.y 1 |
    // This is a polynomial of degree 1 in p.x and p.y, and thus it is a line.
    // It vanishes at e0 and e1.
    let [p_x, p_y] = p.to_array();
    let [e0_x, e0_y] = excluded0.to_array();
    let [e1_x, e1_y] = excluded1.to_array();
    p_x * e0_y + e0_x * e1_y + e1_x * p_y - p_x * e1_y - e0_x * p_y - e1_x * e0_y
}

/*
pub fn complex_conjugate_line(
    point: CirclePoint<SecureField>,
    value: SecureField,
    p: CirclePoint<BaseField>,
) -> SecureField {
    // TODO(AlonH): This assertion will fail at a probability of 1 to 2^62. Use a better solution.
    assert_ne!(
        point.y,
        point.y.complex_conjugate(),
        "Cannot evaluate a line with a single point ({point:?})."
    );
    value
        + (value.complex_conjugate() - value) * (-point.y + p.y)
            / (point.complex_conjugate().y - point.y)
}
*/

fn deep_quotient<F: ComplexExtendable, EF: HasFrobenius<F>>(
    domain: CircleDomain<F>,
    p: RowMajorMatrix<F>,
    zeta: Complex<EF>,
    ps_at_zeta: &[EF],
) -> RowMajorMatrix<EF> {
    RowMajorMatrix::new(
        p.rows()
            .zip(domain.points())
            .flat_map(|(row, x)| {
                izip!(row, ps_at_zeta).map(move |(&p_at_x, &p_at_zeta)| {
                    let num = -frob_line(zeta, p_at_zeta, x) + p_at_x;
                    let denom = pair_vanishing(
                        zeta,
                        Complex::new(zeta.real().frobenius(), zeta.imag().frobenius()),
                        Complex::new(x.real().into(), x.imag().into()),
                    );
                    #[cfg(test)]
                    dbg!(num, denom, denom.inverse());
                    num / denom
                })
            })
            .collect_vec(),
        p.width(),
    )
}

fn reduce_matrix<F: ComplexExtendable>(
    domain: CircleDomain<F>,
    p: RowMajorMatrix<F>,
    zeta: Complex<F>,
    ps_at_zeta: &[F],
    mu: F,
) -> Vec<F> {
    // let num = value - complex_conjugate_line(oods_point, oods_value, point);
    // let denom = pair_vanishing(oods_point, oods_point.complex_conjugate(), point.into_ef(

    p.rows()
        .zip(domain.points())
        .map(|(row, x)| {
            let inv_denom = (Complex::<F>::one() - zeta.conjugate().rotate(x)).inverse();
            izip!(row, ps_at_zeta, mu.square().powers())
                .map(|(&p_at_x, &p_at_zeta, mu2_pow)| {
                    let quotient = Complex::<F>::new_real(p_at_x - p_at_zeta) * inv_denom;
                    mu2_pow * (quotient.real() + mu * quotient.imag())
                })
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use p3_field::{
        extension::{BinomialExtensionField, Complex},
        Field,
    };
    use p3_matrix::{dense::RowMajorMatrix, routines::columnwise_dot_product, Matrix};
    use p3_mersenne_31::Mersenne31;
    use p3_util::log2_strict_usize;
    use rand::{thread_rng, Rng};

    use crate::{
        domain::CircleDomain,
        util::{univariate_to_point, v_n},
        Cfft,
    };

    use super::*;

    type F = Mersenne31;
    type EF = BinomialExtensionField<Mersenne31, 7>;
    // type EF = Complex<Complex<Mersenne31>>;

    fn open_mat_at_point(
        domain: CircleDomain<F>,
        p: RowMajorMatrix<F>,
        pt: Complex<EF>,
    ) -> Vec<EF> {
        let log_n = log2_strict_usize(p.height());
        let basis: Vec<_> = domain.lagrange_basis(pt);
        let v_n_at_zeta = v_n(pt.real(), log_n) - v_n(domain.shift.real(), log_n);
        columnwise_dot_product(p, basis.into_iter())
            .into_iter()
            .map(|x| x * v_n_at_zeta)
            .collect()
    }

    fn is_low_degree(evals: &RowMajorMatrix<F>) -> bool {
        let cfft = Cfft::default();
        cfft.cfft_batch(evals.clone())
            .rows()
            .skip(1)
            .step_by(2)
            .all(|row| row.into_iter().all(|col| col.is_zero()))
    }

    #[test]
    fn test_quotienting() {
        let mut rng = thread_rng();
        let log_n = 3;
        let cfft = Cfft::<F>::default();

        let trace_domain = CircleDomain::<F>::standard(log_n);
        let lde_domain = CircleDomain::<F>::standard(log_n + 1);

        let trace = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_n, 1);
        let lde = cfft.lde(trace.clone(), trace_domain, lde_domain);
        let zeta: EF = rng.gen();
        let zeta_pt: Complex<EF> = univariate_to_point(zeta).unwrap();
        let trace_at_zeta = open_mat_at_point(trace_domain, trace, zeta_pt);
        assert_eq!(
            trace_at_zeta,
            open_mat_at_point(lde_domain, lde.clone(), zeta_pt)
        );

        assert!(is_low_degree(&lde));

        let quotient = deep_quotient(lde_domain, lde, zeta_pt, &trace_at_zeta);
        dbg!(cfft.cfft_batch(quotient.flatten_to_base()));

        // let mu: F = rng.gen();
        // let quotient = reduce_matrix(lde_domain, lde, zeta_pt, &trace_at_zeta, mu);
        // dbg!(cfft.cfft(quotient));
    }
}
