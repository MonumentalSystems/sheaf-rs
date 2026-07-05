//! Dense-coboundary reference tests for FixedGeometry (PLAN.md §5.1):
//! `laplacian_apply(z) == F^T F z` per batch element, energy identities,
//! symmetry/PSD via Rayleigh quotients, edge-mask once-only semantics,
//! `consistency_rms(0) == sqrt(1e-6)`.

mod common;

use common::*;

use ndarray::{s, Array1, Array3};
use sheaf_core::geometry::{FixedGeometry, SheafGeometry, CONSISTENCY_EPS};
use sheaf_core::tensor::NodeState;
use sheaf_core::Scalar;

const N: usize = 4;
const B: usize = 3;
const D_V: usize = 3;
const D_E: usize = 2;

fn setup(mask: Option<Vec<Scalar>>) -> (FixedGeometry, NodeState) {
    let mut rng = Rng::new(7);
    let graph = tiny_graph();
    let e = graph.num_edges();
    let maps = random_maps(&mut rng, e, D_E, D_V, 0.8);
    let mut geo = FixedGeometry::new(graph, maps);
    geo.edge_mask = mask;
    let z = rng.array3((N, B, D_V));
    (geo, z)
}

fn dense_f(geo: &FixedGeometry) -> ndarray::Array2<Scalar> {
    let (fu, fv) = maps_to_blocks(&geo.restriction_maps);
    dense_coboundary(
        &geo.graph.edges,
        &fu,
        &fv,
        geo.edge_mask.as_deref(),
        N,
        D_V,
    )
}

#[test]
fn edge_residuals_match_dense_coboundary() {
    let (geo, z) = setup(None);
    let f = dense_f(&geo);
    let r = geo.edge_residuals(&z);
    for b in 0..B {
        let fz = f.dot(&flatten_batch(&z, b)); // [E * d_e]
        let r_flat: Array1<Scalar> = r
            .slice(ndarray::s![.., b, ..])
            .iter()
            .cloned()
            .collect();
        assert_close(&r_flat, &fz, 1e-5, 1e-5, "edge_residuals vs dense F z");
    }
}

#[test]
fn laplacian_apply_matches_dense_ftf() {
    let (geo, z) = setup(None);
    let f = dense_f(&geo);
    let mut lz = NodeState::zeros(z.dim());
    geo.laplacian_apply(&z, &mut lz);
    for b in 0..B {
        let zf = flatten_batch(&z, b);
        let dense = f.t().dot(&f.dot(&zf)); // F^T F z
        let got = flatten_batch(&lz, b);
        assert_close(&got, &dense, 1e-4, 1e-4, "laplacian_apply vs F^T F z");
    }
}

#[test]
fn energy_identities() {
    let (geo, z) = setup(None);
    // energy == 1/2 sum r^2
    let r = geo.edge_residuals(&z);
    let half_r2: Scalar = 0.5 * r.iter().map(|&x| x * x).sum::<Scalar>();
    let energy = geo.energy(&z);
    assert!(
        (energy - half_r2).abs() <= 1e-4 * half_r2.abs().max(1.0),
        "energy {energy} != 1/2 sum r^2 {half_r2}"
    );
    // energy == 1/2 z^T L z (summed over batch)
    let mut lz = NodeState::zeros(z.dim());
    geo.laplacian_apply(&z, &mut lz);
    let ztlz: Scalar = z.iter().zip(lz.iter()).map(|(&a, &b)| a * b).sum();
    assert!(
        (energy - 0.5 * ztlz).abs() <= 1e-3 * energy.abs().max(1.0),
        "energy {energy} != 1/2 z^T L z {}",
        0.5 * ztlz
    );
}

#[test]
fn laplacian_symmetric_psd_rayleigh() {
    let (geo, _z) = setup(None);
    let mut rng = Rng::new(21);
    for _ in 0..8 {
        let a = rng.array3((N, B, D_V));
        let b = rng.array3((N, B, D_V));
        let mut la = NodeState::zeros(a.dim());
        let mut lb = NodeState::zeros(b.dim());
        geo.laplacian_apply(&a, &mut la);
        geo.laplacian_apply(&b, &mut lb);
        let a_lb: Scalar = a.iter().zip(lb.iter()).map(|(&x, &y)| x * y).sum();
        let la_b: Scalar = la.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        assert!(
            (a_lb - la_b).abs() <= 1e-3 * a_lb.abs().max(1.0),
            "L not symmetric: <a, Lb> = {a_lb}, <La, b> = {la_b}"
        );
        let a_la: Scalar = a.iter().zip(la.iter()).map(|(&x, &y)| x * y).sum();
        assert!(a_la >= -1e-5, "L not PSD: Rayleigh <a, La> = {a_la}");
    }
}

#[test]
fn edge_mask_applied_once_not_squared() {
    // A fractional mask discriminates once (F^T (m F z)) from twice
    // (m^2) application: the dense equivalent of masking the residual once is
    // F^T M F with M = diag(m), i.e. the mask enters the dense F ONCE on the
    // residual side and the adjoint uses the UNMASKED maps.
    let mask = vec![1.0, 0.5, 0.0, 1.0, 0.25];
    let (geo, z) = setup(Some(mask.clone()));

    // Dense reference: r = M * (F z) with unmasked F; L z = F^T r.
    let (fu, fv) = maps_to_blocks(&geo.restriction_maps);
    let f_unmasked = dense_coboundary(&geo.graph.edges, &fu, &fv, None, N, D_V);
    let mut lz = NodeState::zeros(z.dim());
    geo.laplacian_apply(&z, &mut lz);
    for b in 0..B {
        let zf = flatten_batch(&z, b);
        let mut r = f_unmasked.dot(&zf); // [E * d_e]
        for ei in 0..geo.graph.num_edges() {
            for i in 0..D_E {
                r[ei * D_E + i] *= mask[ei];
            }
        }
        let dense = f_unmasked.t().dot(&r);
        let got = flatten_batch(&lz, b);
        assert_close(&got, &dense, 1e-4, 1e-4, "masked laplacian (mask once)");
    }

    // Masked-out edge contributes no residual.
    let r = geo.edge_residuals(&z);
    for bi in 0..B {
        for di in 0..D_E {
            assert_eq!(r[[2, bi, di]], 0.0, "mask 0 edge must have zero residual");
        }
    }
}

#[test]
fn batch_axis_is_independent_and_bitwise_equal() {
    // The `parallel` feature fans the geometry ops over the batch axis; this is
    // numerically identical ONLY because the Laplacian never mixes batch
    // elements. Pin that invariant: the b-th slab of a full [N, B, d_v] run must
    // equal — bit for bit — an independent single-batch [N, 1, d_v] run. Holds
    // under the default, `parallel`, and `f64` builds alike.
    let (geo, z) = setup(None);
    let r_full = geo.edge_residuals(&z);
    let mut l_full = NodeState::zeros(z.dim());
    geo.laplacian_apply(&z, &mut l_full);
    let crms_full = geo.consistency_rms(&z);

    for b in 0..B {
        let z_b: Array3<Scalar> = z.slice(s![.., b..b + 1, ..]).to_owned();
        let r_b = geo.edge_residuals(&z_b);
        let mut l_b = NodeState::zeros(z_b.dim());
        geo.laplacian_apply(&z_b, &mut l_b);
        let crms_b = geo.consistency_rms(&z_b);

        for (&got, &want) in r_b.slice(s![.., 0, ..]).iter().zip(r_full.slice(s![.., b, ..]).iter()) {
            assert_eq!(got, want,
                "edge_residuals batch {b} slab must match single-batch run bitwise");
        }
        for (&got, &want) in l_b.slice(s![.., 0, ..]).iter().zip(l_full.slice(s![.., b, ..]).iter()) {
            assert_eq!(got, want,
                "laplacian_apply batch {b} slab must match single-batch run bitwise");
        }
        assert_eq!(crms_b[0], crms_full[b],
            "consistency_rms batch {b} must match single-batch run bitwise");
    }
}

#[test]
fn consistency_rms_at_zero_is_sqrt_eps() {
    let (geo, _z) = setup(None);
    let z0 = NodeState::zeros((N, B, D_V));
    let rms = geo.consistency_rms(&z0);
    let expect = CONSISTENCY_EPS.sqrt();
    for &v in rms.iter() {
        assert!(
            (v - expect).abs() <= 1e-9,
            "consistency_rms(0) = {v}, expected sqrt(1e-6) = {expect}"
        );
    }
}

#[test]
fn consistency_rms_matches_manual_formula() {
    let (geo, z) = setup(None);
    let r = geo.edge_residuals(&z);
    let (e, _b, d_e) = r.dim();
    let rms = geo.consistency_rms(&z);
    for bi in 0..B {
        let mut acc = 0.0 as Scalar;
        for ei in 0..e {
            for di in 0..d_e {
                acc += r[[ei, bi, di]] * r[[ei, bi, di]];
            }
        }
        let expect = (acc / (e * d_e) as Scalar + CONSISTENCY_EPS).sqrt();
        assert!(
            (rms[bi] - expect).abs() <= 1e-6,
            "consistency_rms[{bi}] = {}, expected {expect}",
            rms[bi]
        );
    }
}
