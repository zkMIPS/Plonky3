use alloc::vec::Vec;
use core::arch::x86_64::{__m256i, _mm256_setzero_si256};
use serde::{Deserialize, Serialize};
use serde::ser::SerializeSeq;

/// Converts `__m256i` to a `[i32; 8]` array before serialization.
pub fn serialize_m256i_vec<S>(vec: &Vec<__m256i>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let converted: Vec<[i32; 8]> = vec.iter().map(|&x| unsafe {
        core::mem::transmute::<__m256i, [i32; 8]>(x)
    }).collect();
    converted.serialize(serializer)
}

/// Converts a `[i32; 8]` array back to `__m256i` after deserialization.
pub fn deserialize_m256i_vec<'de, D>(deserializer: D) -> Result<Vec<__m256i>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let converted: Vec<[i32; 8]> = Vec::<[i32; 8]>::deserialize(deserializer)?;
    Ok(converted.iter().map(|&x| unsafe {
        core::mem::transmute::<[i32; 8], __m256i>(x)
    }).collect())
}


/// Converts each `__m256i` element in the array to a `[i32; 8]` array before serialization.
pub fn serialize_packed_m256i<S, const WIDTH: usize>(
    vec: &Vec<[__m256i; WIDTH]>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let converted: Vec<Vec<[i32; 8]>> = vec.iter().map(|array| {
        array.iter().map(|&x| unsafe {
            core::mem::transmute::<__m256i, [i32; 8]>(x)
        }).collect()
    }).collect();

    converted.serialize(serializer)
}


/// Converts each `[i32; 8]` array back to `__m256i` after deserialization.
pub fn deserialize_packed_m256i<'de, D, const WIDTH: usize>(
    deserializer: D,
) -> Result<Vec<[__m256i; WIDTH]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let converted: Vec<Vec<[i32; 8]>> = Vec::<Vec<[i32; 8]>>::deserialize(deserializer)?;

    Ok(converted.iter().map(|vec| {
        let mut result = [unsafe { _mm256_setzero_si256() }; WIDTH];
        for (i, &x) in vec.iter().enumerate() {
            result[i] = unsafe {
                core::mem::transmute::<[i32; 8], __m256i>(x)
            };
        }
        result
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn generate_random_m256i<const WIDTH: usize>(size: usize) -> Vec<[__m256i; WIDTH]> {
        let mut rng = rand::thread_rng();
        (0..size)
            .map(|_| {
                let mut arr = [unsafe { _mm256_setzero_si256() }; WIDTH];
                for i in 0..WIDTH {
                    let random_values: [i32; 8] = rng.gen();
                    arr[i] = unsafe { core::mem::transmute::<[i32; 8], __m256i>(random_values) };
                }
                arr
            })
            .collect()
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct TestPackedStruct<const WIDTH: usize> {
        #[serde(serialize_with = "serialize_packed_m256i", deserialize_with = "deserialize_packed_m256i")]
        packed_values: Vec<[__m256i; WIDTH]>,
    }

    #[test]
    fn test_packed_m256i() {
        const WIDTH: usize = 5;
        const NUM_VECTORS: usize = 3;

        unsafe {
            let data = generate_random_m256i::<WIDTH>(NUM_VECTORS);

            let test_obj = TestPackedStruct::<WIDTH> { packed_values: data.clone() };
            let serialized = serde_json::to_string_pretty(&test_obj).unwrap();
            let deserialized: TestPackedStruct<WIDTH> = serde_json::from_str(&serialized).unwrap();

            for (original_arr, deserialized_arr) in data.iter().zip(&deserialized.packed_values) {
                for i in 0..WIDTH {
                    let original_i32: [i32; 8] = core::mem::transmute(original_arr[i]);
                    let deserialized_i32: [i32; 8] = core::mem::transmute(deserialized_arr[i]);
                    assert_eq!(original_i32, deserialized_i32, "Mismatch found at index {}", i);
                }
            }
        }
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct TestStruct {
        #[serde(serialize_with = "serialize_m256i_vec", deserialize_with = "deserialize_m256i_vec")]
        values: Vec<__m256i>,
    }

    fn random_m256i() -> __m256i {
        let mut rng = rand::thread_rng();
        let random_array: [i32; 8] = rng.gen();
        unsafe { core::mem::transmute::<[i32; 8], __m256i>(random_array) }
    }

    #[test]
    fn test_vec_m256i() {
        let test_data: Vec<__m256i> = (0..5).map(|_| random_m256i()).collect();

        let test_struct = TestStruct { values: test_data.clone() };
        let serialized = serde_json::to_string(&test_struct).expect("Serialization failed");
        let deserialized: TestStruct = serde_json::from_str(&serialized).expect("Deserialization failed");

        // Compare original and deserialized values
        unsafe {
            for (original_arr, deserialized_arr) in test_data.iter().zip(&deserialized.values) {
                let original_i32: [i32; 8] = core::mem::transmute(*original_arr);
                let deserialized_i32: [i32; 8] = core::mem::transmute(*deserialized_arr);
                assert_eq!(original_i32, deserialized_i32, "Mismatch found");
            }
        }
    }
}
