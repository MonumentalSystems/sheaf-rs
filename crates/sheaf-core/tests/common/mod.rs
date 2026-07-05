//! Shared helpers for the sheaf-core property tests (PLAN.md §5.1).
#![allow(dead_code)]

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, Array4, Array5, Dimension, s};

use sheaf_core::graph::AgentGraph;
use sheaf_core::tensor::{NodeState, RestrictionMaps};
use sheaf_core::Scalar;

/// Tiny deterministic xorshift64* RNG — no dev-dep needed, seeds are pinned.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15).max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in [-1, 1). (Named `scalar` — routes through the crate `Scalar`
    /// so the same helpers compile under both the default f32 and `f64` builds.)
    pub fn scalar(&mut self) -> Scalar {
        let u = (self.next_u64() >> 40) as Scalar / (1u64 << 24) as Scalar; // [0, 1)
        2.0 * u - 1.0
    }

    /// Uniform in [lo, hi).
    pub fn scalar_in(&mut self, lo: Scalar, hi: Scalar) -> Scalar {
        lo + (self.scalar() * 0.5 + 0.5) * (hi - lo)
    }

    pub fn array1(&mut self, n: usize) -> Array1<Scalar> {
        Array1::from_shape_fn(n, |_| self.scalar())
    }

    pub fn array2(&mut self, d: (usize, usize)) -> Array2<Scalar> {
        Array2::from_shape_fn(d, |_| self.scalar())
    }

    pub fn array3(&mut self, d: (usize, usize, usize)) -> Array3<Scalar> {
        Array3::from_shape_fn(d, |_| self.scalar())
    }

    pub fn array4(&mut self, d: (usize, usize, usize, usize)) -> Array4<Scalar> {
        Array4::from_shape_fn(d, |_| self.scalar())
    }

    pub fn array5(&mut self, d: (usize, usize, usize, usize, usize)) -> Array5<Scalar> {
        Array5::from_shape_fn(d, |_| self.scalar())
    }
}

/// Elementwise |a - b| <= atol + rtol * |b| with shape check.
pub fn assert_close<D: Dimension>(
    a: &ndarray::Array<Scalar, D>,
    b: &ndarray::Array<Scalar, D>,
    atol: Scalar,
    rtol: Scalar,
    what: &str,
) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape mismatch");
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let tol = atol + rtol * bv.abs();
        assert!(
            (av - bv).abs() <= tol,
            "{what}: element {i} differs: {av} vs {bv} (tol {tol})"
        );
    }
}

/// Dense coboundary F in R^{E*d_e x N*d_v} from per-edge endpoint blocks:
/// block row e carries +mask*F_u at u's columns and -mask*F_v at v's columns.
pub fn dense_coboundary(
    edges: &[[u32; 2]],
    fu: &[Array2<Scalar>],
    fv: &[Array2<Scalar>],
    mask: Option<&[Scalar]>,
    n: usize,
    d_v: usize,
) -> Array2<Scalar> {
    let e_cnt = edges.len();
    let d_e = fu[0].nrows();
    let mut f = Array2::zeros((e_cnt * d_e, n * d_v));
    for (ei, &[u, v]) in edges.iter().enumerate() {
        let m = mask.map_or(1.0, |mk| mk[ei]);
        for i in 0..d_e {
            for j in 0..d_v {
                f[[ei * d_e + i, u as usize * d_v + j]] += m * fu[ei][[i, j]];
                f[[ei * d_e + i, v as usize * d_v + j]] -= m * fv[ei][[i, j]];
            }
        }
    }
    f
}

/// Flatten one batch element of a node state to `[N * d_v]` (row-major).
pub fn flatten_batch(z: &NodeState, b: usize) -> Array1<Scalar> {
    z.slice(s![.., b, ..]).iter().cloned().collect()
}

/// Inverse of `flatten_batch` into batch slot `b` of `out`.
pub fn unflatten_batch(flat: &Array1<Scalar>, out: &mut NodeState, b: usize) {
    let (n, _bsz, d_v) = out.dim();
    for ni in 0..n {
        for di in 0..d_v {
            out[[ni, b, di]] = flat[ni * d_v + di];
        }
    }
}

/// A tiny 5-edge, 4-node test graph with a duplicated endpoint (node 0 sits on
/// three edges, so the scatter-add accumulation path is exercised).
pub fn tiny_graph() -> Arc<AgentGraph> {
    let edges = vec![[0, 1], [0, 2], [1, 3], [2, 3], [0, 3]];
    Arc::new(AgentGraph::from_edges(edges, 4))
}

/// A 2x2 grid with 8-way-style edges (E, S and both diagonals), positions
/// (y, x), for directional slot-table + LoRA gather tests.
pub fn grid_2x2_8way() -> Arc<AgentGraph> {
    // nodes: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
    let positions =
        Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    let edges = vec![[0, 1], [0, 2], [1, 3], [2, 3], [0, 3], [1, 2]];
    Arc::new(AgentGraph::new_grid(edges, positions, 8))
}

/// Random restriction maps `[E, 2, d_e, d_v]`, scaled to keep L well-behaved.
pub fn random_maps(rng: &mut Rng, e: usize, d_e: usize, d_v: usize, scale: Scalar) -> RestrictionMaps {
    Array4::from_shape_fn((e, 2, d_e, d_v), |_| scale * rng.scalar())
}

/// Extract per-edge endpoint blocks (F_u, F_v) from `[E, 2, d_e, d_v]` maps.
pub fn maps_to_blocks(maps: &RestrictionMaps) -> (Vec<Array2<Scalar>>, Vec<Array2<Scalar>>) {
    let e = maps.shape()[0];
    let mut fu = Vec::with_capacity(e);
    let mut fv = Vec::with_capacity(e);
    for ei in 0..e {
        fu.push(maps.slice(s![ei, 0, .., ..]).to_owned());
        fv.push(maps.slice(s![ei, 1, .., ..]).to_owned());
    }
    (fu, fv)
}
