use anyhow::{Context, Result};
use image::{GenericImageView, imageops::FilterType};
use ndarray::{Array, Array4, Axis};
use ort::{Environment, SessionBuilder, GraphOptimizationLevel, Value};
use std::fs;
use std::path::Path;
use std::sync::Arc;

pub struct MonOCR {
    session: ort::Session,
    charset: Vec<char>,
}

impl MonOCR {
    pub fn new<P: AsRef<Path>>(model_path: P, charset_path: P) -> Result<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("monocr")
                .build()?
        );

        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_path)?;

        let charset_content = fs::read_to_string(charset_path)
            .context("Failed to read charset file")?;
        
        let charset: Vec<char> = charset_content.trim().chars().collect();

        Ok(Self { session, charset })
    }

    pub fn preprocess(&self, img_path: &Path) -> Result<Array4<f32>> {
        let img = image::open(img_path).context("Failed to open image")?;
        let gray_img = img.grayscale();
        
        let (width, height) = gray_img.dimensions();
        let target_height = 128;
        let target_width = 1024;
        
        let aspect_ratio = width as f32 / height as f32;
        let new_width = ((target_height as f32 * aspect_ratio).round() as u32).min(target_width);

        let resized = gray_img.resize_exact(new_width, target_height, FilterType::Triangle);
        
        // Create 1x1x128x1024 tensor, initialized with white background normalized to 1.0
        // (255 / 127.5) - 1.0 = 1.0
        let mut input_tensor = Array4::<f32>::from_elem((1, 1, target_height as usize, target_width as usize), 1.0);
        
        for (x, y, pixel) in resized.pixels() {
            // Normalize to [-1.0, 1.0]
            let val = (pixel[0] as f32 / 127.5) - 1.0;
            input_tensor[[0, 0, y as usize, x as usize]] = val;
        }

        Ok(input_tensor)
    }

    pub fn decode(&self, preds: &[f32], num_classes: usize) -> String {
        let mut decoded_text = String::new();
        let mut prev_idx = -1;
        
        // Preds is flat [1, sequence_length, num_classes]
        // We only care about sequence_length
        let seq_len = preds.len() / num_classes;
        
        for t in 0..seq_len {
            let start = t * num_classes;
            let end = start + num_classes;
            let logits = &preds[start..end];
            
            // Argmax
            let mut max_val = f32::MIN;
            let mut max_idx = 0;
            
            for (i, &val) in logits.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            
            if max_idx != 0 && max_idx as i32 != prev_idx {
                // max_idx 0 is blank
                // Our charset is 0-indexed, so max_idx-1 maps to charset[i]
                if max_idx > 0 && max_idx <= self.charset.len() {
                    decoded_text.push(self.charset[max_idx - 1]);
                }
            }
            prev_idx = max_idx as i32;
        }

        decoded_text
    }

    pub fn predict(&self, img_path: &Path) -> Result<String> {
        let input_tensor = self.preprocess(img_path)?;
        
        // Run inference
        let outputs = self.session.run(vec![Value::from_array(self.session.allocator(), &input_tensor)?])?;
        
        // Get output tensor
        let output_tensor: ort::Tensor<f32> = outputs[0].try_extract()?;
        let output_view = output_tensor.view();
        
        // Flatten
        let preds: Vec<f32> = output_view.iter().cloned().collect();
        let num_classes = self.charset.len() + 1;
        
        Ok(self.decode(&preds, num_classes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() {
        // Placeholder test
        assert!(true);
    }
}
