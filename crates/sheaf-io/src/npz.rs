//! Thin .npz helpers over `ndarray_npy::NpzReader` for the golden fixtures
//! (goldens/CONTRACT.md). Goldens carry only f32 and i64 arrays; anything
//! else in an archive is a contract violation and errors loudly (the typed
//! readers refuse dtype mismatches).

use std::fs::File;
use std::path::Path;

use anyhow::Context;
use ndarray::{ArrayD, IxDyn};
use ndarray_npy::NpzReader;

/// An opened .npz archive with typed accessors.
pub struct Npz {
    reader: NpzReader<File>,
    path: String,
}

impl Npz {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let file =
            File::open(path).with_context(|| format!("opening npz {}", path.display()))?;
        let reader = NpzReader::new(file)
            .with_context(|| format!("reading npz archive {}", path.display()))?;
        Ok(Npz {
            reader,
            path: path.display().to_string(),
        })
    }

    /// Array names present in the archive (without the `.npy` suffix).
    pub fn names(&mut self) -> anyhow::Result<Vec<String>> {
        self.reader
            .names()
            .with_context(|| format!("listing arrays in {}", self.path))
    }

    pub fn f32(&mut self, name: &str) -> anyhow::Result<ArrayD<f32>> {
        self.reader
            .by_name::<ndarray::OwnedRepr<f32>, IxDyn>(name)
            .with_context(|| format!("reading f32 array {:?} from {}", name, self.path))
    }

    pub fn i64(&mut self, name: &str) -> anyhow::Result<ArrayD<i64>> {
        self.reader
            .by_name::<ndarray::OwnedRepr<i64>, IxDyn>(name)
            .with_context(|| format!("reading i64 array {:?} from {}", name, self.path))
    }

    /// Fetch + shape-check in one call (golden parity ergonomics).
    pub fn f32_shaped(&mut self, name: &str, shape: &[usize]) -> anyhow::Result<ArrayD<f32>> {
        let arr = self.f32(name)?;
        anyhow::ensure!(
            arr.shape() == shape,
            "array {:?} in {}: expected shape {:?}, got {:?}",
            name,
            self.path,
            shape,
            arr.shape()
        );
        Ok(arr)
    }

    /// Fetch + shape-check for integer arrays.
    pub fn i64_shaped(&mut self, name: &str, shape: &[usize]) -> anyhow::Result<ArrayD<i64>> {
        let arr = self.i64(name)?;
        anyhow::ensure!(
            arr.shape() == shape,
            "array {:?} in {}: expected shape {:?}, got {:?}",
            name,
            self.path,
            shape,
            arr.shape()
        );
        Ok(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};
    use ndarray_npy::NpzWriter;

    fn write_fixture(path: &Path) {
        let mut npz = NpzWriter::new(File::create(path).unwrap());
        let floats = Array3::<f32>::from_shape_fn((2, 3, 4), |(a, b, c)| {
            a as f32 + 10.0 * b as f32 + 100.0 * c as f32
        });
        let ints = Array1::<i64>::from(vec![-3, 0, 7]);
        npz.add_array("floats", &floats).unwrap();
        npz.add_array("ints", &ints).unwrap();
        npz.finish().unwrap();
    }

    #[test]
    fn round_trips_f32_and_i64() {
        let path = std::env::temp_dir().join("sheaf_io_npz_test.npz");
        write_fixture(&path);

        let mut npz = Npz::open(&path).unwrap();
        let mut names = npz.names().unwrap();
        names.sort();
        assert_eq!(names, vec!["floats".to_string(), "ints".to_string()]);

        let floats = npz.f32_shaped("floats", &[2, 3, 4]).unwrap();
        assert_eq!(floats[[1, 2, 3]], 1.0 + 20.0 + 300.0);
        let ints = npz.i64_shaped("ints", &[3]).unwrap();
        assert_eq!(ints.as_slice().unwrap(), &[-3, 0, 7]);

        // Wrong dtype errors loudly.
        assert!(npz.f32("ints").is_err());
        // Wrong shape errors loudly.
        assert!(npz.f32_shaped("floats", &[2, 3, 5]).is_err());
        // Missing name errors loudly.
        assert!(npz.f32("nope").is_err());

        std::fs::remove_file(&path).ok();
    }
}
