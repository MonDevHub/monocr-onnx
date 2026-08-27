//! Utility Functions
//!
//! This module provides utility functions for OCR accuracy calculation
//! and text comparison.

/// Calculate character error rate (CER) between predicted and ground truth
/// Returns accuracy percentage (0-100)
///
/// This function measures OCR accuracy by computing the Levenshtein distance
/// (edit distance) between the predicted text and the ground truth, then
/// converting it to an accuracy percentage.
///
/// # Arguments
///
/// * `predicted` - The text recognized by the OCR system
/// * `ground_truth` - The expected/correct text
///
/// # Returns
///
/// Accuracy as a percentage (0-100), rounded to 2 decimal places
///
/// # Algorithm
///
/// 1. If either string is empty:
///    - Empty ground truth: returns 0.0
///    - Empty prediction: returns 0.0
///    - Both empty: returns 100.0
/// 2. Compute Levenshtein distance (minimum edits to transform one string to another)
/// 3. Calculate accuracy: `(1 - distance/max_len) * 100`
///
/// # Example
///
/// ```
/// use monocr_onnx::calculate_accuracy;
///
/// let accuracy = calculate_accuracy("hello", "hello");
/// assert_eq!(accuracy, 100.0);
///
/// let accuracy = calculate_accuracy("hellp", "hello");
/// // 1 error out of 5 chars = 80% accuracy
/// assert!((accuracy - 80.0).abs() < 0.01);
/// ```
pub fn calculate_accuracy(predicted: &str, ground_truth: &str) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }
    if predicted.is_empty() {
        return 0.0;
    }

    let distance = levenshtein(predicted, ground_truth);
    let max_len = predicted.chars().count().max(ground_truth.chars().count());

    if max_len == 0 {
        return 100.0;
    }

    ((1.0 - distance as f64 / max_len as f64) * 100.0 * 100.0).round() / 100.0
}

/// Calculate Levenshtein distance between two strings
///
/// The Levenshtein distance (also known as edit distance) is the minimum number
/// of single-character edits required to change one string into another.
/// Edits include:
/// - Insertion of a character
/// - Deletion of a character
/// - Substitution of a character
///
/// This is an internal function used by [`calculate_accuracy`](calculate_accuracy).
///
/// # Arguments
///
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
///
/// The Levenshtein distance as a usize
///
/// # Algorithm
///
/// Uses dynamic programming with a 2D matrix where:
/// - `matrix[i][j]` = edit distance between first i chars of s1 and first j chars of s2
/// - Base cases: distance from empty string to first i chars = i
///
/// # Complexity
///
/// Time: O(m * n) where m and n are the lengths of the strings
/// Space: O(m * n)
fn levenshtein(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    // The matrix is indexed `matrix[j][i]`: s2 selects the row, s1 the column.
    // That is the opposite of how the docs above name it, so any rewrite of the
    // two initialisation loops has to keep the axes straight. Iterating
    // `matrix` where a column walk is meant silently produces wrong distances,
    // and only when len1 != len2.
    let mut matrix = vec![vec![0; len1 + 1]; len2 + 1];

    // Row 0: distance from the empty s2 prefix to each s1 prefix.
    for (i, cell) in matrix[0].iter_mut().enumerate() {
        *cell = i;
    }
    // Column 0: distance from the empty s1 prefix to each s2 prefix.
    for (j, row) in matrix.iter_mut().enumerate() {
        row[0] = j;
    }

    // Fill the matrix
    for j in 1..=len2 {
        for i in 1..=len1 {
            let substitution_cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[j][i] = [
                matrix[j][i - 1] + 1,                     // deletion
                matrix[j - 1][i] + 1,                     // insertion
                matrix[j - 1][i - 1] + substitution_cost, // substitution
            ]
            .into_iter()
            .min()
            .unwrap();
        }
    }

    matrix[len2][len1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("a", ""), 1);
        assert_eq!(levenshtein("", "a"), 1);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    /// Pins the matrix axes. The distance matrix is indexed `matrix[j][i]` with
    /// s2 on the rows and s1 on the columns, so the two initialisation loops
    /// walk different axes. Get one of them wrong and equal-length inputs still
    /// come out right, which is why the older equal-length cases above cannot
    /// catch it. These lengths differ in both directions.
    ///
    /// Distance is 3 either way: "ab" -> "abcde" inserts c, d and e, and edit
    /// distance is symmetric so the reverse deletes the same three.
    #[test]
    fn levenshtein_handles_unequal_lengths() {
        assert_eq!(levenshtein("ab", "abcde"), 3);
        assert_eq!(levenshtein("abcde", "ab"), 3);
    }

    #[test]
    fn test_accuracy() {
        assert_eq!(calculate_accuracy("hello", "hello"), 100.0);
        assert_eq!(calculate_accuracy("", "hello"), 0.0);
        assert_eq!(calculate_accuracy("hello", ""), 0.0);

        // 1 error out of 5 chars = 80% accuracy
        let acc = calculate_accuracy("hellp", "hello");
        assert!((79.99..=80.01).contains(&acc));
    }

    /// The accuracy path over the same unequal-length inputs, so a broken
    /// matrix is caught at the public API too.
    ///
    /// Derivation: levenshtein("ab", "abcde") = 3 (insert c, d, e).
    /// max_len = max(2, 5) = 5. accuracy = (1 - 3/5) * 100 = 40.0, and the
    /// round-to-2-decimals step leaves it unchanged.
    #[test]
    fn accuracy_uses_the_longer_string_when_lengths_differ() {
        assert_eq!(calculate_accuracy("ab", "abcde"), 40.0);
        assert_eq!(calculate_accuracy("abcde", "ab"), 40.0);
    }
}
