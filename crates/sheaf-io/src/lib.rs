//! IO layer: safetensors weight loading, .npz golden fixtures, data views
//! (patchify / grid edges / reassembly), and the maze generator.
//!
//! Ports `sheaf_admm.data.{views,build_maze}` plus the loader side of
//! `tools/export_weights.py`. Golden fixture layout: goldens/CONTRACT.md.

pub mod mazegen;
pub mod npz;
pub mod views;
pub mod weights;

pub use weights::{load_maze_model, WeightCollection};
