//! Thin .npz helpers over `ndarray_npy::NpzReader` for the golden fixtures
//! (goldens/CONTRACT.md). Goldens carry only f32 and i64 arrays; anything
//! else in an archive is a contract violation and errors loudly.

use std::path::Path;

use ndarray::{ArrayD, IxDyn};

/// An opened .npz archive with typed accessors.
pub struct Npz {
    // wraps ndarray_npy::NpzReader<std::fs::File>
    _private: (),
}

impl Npz {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        todo!()
    }

    /// Array names present in the archive (without the `.npy` suffix).
    pub fn names(&self) -> Vec<String> {
        todo!()
    }

    pub fn f32(&mut self, name: &str) -> anyhow::Result<ArrayD<f32>> {
        todo!()
    }

    pub fn i64(&mut self, name: &str) -> anyhow::Result<ArrayD<i64>> {
        todo!()
    }

    /// Fetch + shape-check in one call (golden parity ergonomics).
    pub fn f32_shaped(&mut self, name: &str, shape: &[usize]) -> anyhow::Result<ArrayD<f32>> {
        let _ = IxDyn(shape);
        todo!()
    }
}
