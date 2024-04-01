use p3_field::Field;
use p3_symmetric::Permutation;

use crate::MdsPermutation;

#[derive(Clone)]
pub struct NaiveMds<F: Field, const WIDTH: usize> {
    pub matrix: [[F; WIDTH]; WIDTH],
}

impl <F: Field, const WIDTH: usize> NaiveMds<F, WIDTH> {
    pub fn new(matrix: [[F; WIDTH]; WIDTH]) -> Self {
        Self { matrix }
    }
}

impl<F: Field, const WIDTH: usize> Permutation<[F; WIDTH]> for NaiveMds<F, WIDTH> {
    fn permute(&self, input: [F; WIDTH]) -> [F; WIDTH] {
        let mut output = [F::zero(); WIDTH];
        for i in 0..WIDTH {
            for j in 0..WIDTH {
                output[i] = output[i].clone() + self.matrix[i][j].clone() * input[j].clone();
            }
        }
        output
    }

    fn permute_mut(&self, input: &mut [F; WIDTH]) {
        *input = self.permute(*input);
    }
}

impl<F: Field, const WIDTH: usize> MdsPermutation<F, WIDTH> for NaiveMds<F, WIDTH> {}

impl<F: Field, const WIDTH: usize> NaiveMds<F, WIDTH> {
    pub fn from_circ_and_diag(circulant: [F; WIDTH], diagonal: [F; WIDTH]) -> Self {
        let mut matrix = [[F::zero(); WIDTH]; WIDTH];
        for i in 0..WIDTH {
            for j in 0..WIDTH {
                matrix[i][j] = circulant[(WIDTH + j - i) % WIDTH].clone();
                if i == j {
                    matrix[i][j] += diagonal[i].clone();
                }
            }
        }
        Self::new(matrix)
    }
}