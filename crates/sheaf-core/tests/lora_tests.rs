//! LoRA geometry tests (PLAN.md §5.1): LoRA-with-B=0 == FixedGeometry, dense
//! per-batch reference with materialized `F = R + scale * gate * A B^T`
//! (checks the factored apply AND the gate-on-the-A^T-r-side adjoint), and the
//! `create_directional` edge gather against the slot tables.

mod common;

use common::*;

use ndarray::{s, Array2, Array5};
use sheaf_core::geometry::{FixedGeometry, LoraGeometry, SheafGeometry};
use sheaf_core::tensor::NodeState;

const N: usize = 4;
const B: usize = 2;
const D_V: usize = 3;
const D_E: usize = 2;
const RANK: usize = 2;
const K_DIRS: usize = 8;
const LORA_ALPHA: f32 = 4.0;

/// LoRA geometry over the 2x2 8-way grid, from random per-node factors.
fn setup_lora(
    b_zero: bool,
    with_gate: bool,
    seed: u64,
) -> (LoraGeometry, FixedGeometry, NodeState) {
    let mut rng = Rng::new(seed);
    let graph = grid_2x2_8way();
    let e = graph.num_edges();
    let maps = random_maps(&mut rng, e, D_E, D_V, 0.8);

    let a = rng.array5((N, B, K_DIRS, D_E, RANK));
    let b = if b_zero {
        Array5::zeros((N, B, K_DIRS, D_V, RANK))
    } else {
        rng.array5((N, B, K_DIRS, D_V, RANK))
    };
    let gate = with_gate.then(|| rng.array3((N, B, K_DIRS)));

    let lora = LoraGeometry::create_directional(
        graph.clone(),
        maps.clone(),
        &a,
        &b,
        gate.as_ref(),
        LORA_ALPHA,
    );
    let fixed = FixedGeometry::new(graph, maps);
    let z = rng.array3((N, B, D_V));
    (lora, fixed, z)
}

#[test]
fn lora_scale_is_alpha_over_rank() {
    let (lora, _fixed, _z) = setup_lora(false, false, 3);
    assert_eq!(lora.scale(), LORA_ALPHA / RANK as f32);
}

#[test]
fn lora_with_b_zero_equals_fixed() {
    let (lora, fixed, z) = setup_lora(true, false, 5);
    let r_lora = lora.edge_residuals(&z);
    let r_fixed = fixed.edge_residuals(&z);
    assert_close(&r_lora, &r_fixed, 1e-5, 1e-5, "B=0 edge_residuals == Fixed");

    let mut l_lora = NodeState::zeros(z.dim());
    let mut l_fixed = NodeState::zeros(z.dim());
    lora.laplacian_apply(&z, &mut l_lora);
    fixed.laplacian_apply(&z, &mut l_fixed);
    assert_close(&l_lora, &l_fixed, 1e-5, 1e-5, "B=0 laplacian == Fixed");

    let e_lora = lora.energy(&z);
    let e_fixed = fixed.energy(&z);
    assert!((e_lora - e_fixed).abs() <= 1e-5 * e_fixed.abs().max(1.0));
}

#[test]
fn lora_gate_zero_equals_fixed() {
    let mut rng = Rng::new(11);
    let graph = grid_2x2_8way();
    let e = graph.num_edges();
    let maps = random_maps(&mut rng, e, D_E, D_V, 0.8);
    let a = rng.array5((N, B, K_DIRS, D_E, RANK));
    let b = rng.array5((N, B, K_DIRS, D_V, RANK));
    let gate = ndarray::Array3::zeros((N, B, K_DIRS));
    let lora = LoraGeometry::create_directional(
        graph.clone(),
        maps.clone(),
        &a,
        &b,
        Some(&gate),
        LORA_ALPHA,
    );
    let fixed = FixedGeometry::new(graph, maps);
    let z = rng.array3((N, B, D_V));
    let r_lora = lora.edge_residuals(&z);
    let r_fixed = fixed.edge_residuals(&z);
    assert_close(&r_lora, &r_fixed, 1e-6, 1e-6, "gate=0 == Fixed residuals");
    let mut l_lora = NodeState::zeros(z.dim());
    let mut l_fixed = NodeState::zeros(z.dim());
    lora.laplacian_apply(&z, &mut l_lora);
    fixed.laplacian_apply(&z, &mut l_fixed);
    assert_close(&l_lora, &l_fixed, 1e-6, 1e-6, "gate=0 == Fixed laplacian");
}

/// Materialize the effective per-(edge, batch) map `R + scale * g * A B^T`.
fn effective_blocks(lora: &LoraGeometry, slot: usize, b: usize) -> Vec<Array2<f32>> {
    let (a_e, b_e, gate) = if slot == 0 {
        (&lora.a_u_edge, &lora.b_u_edge, &lora.gate_u_edge)
    } else {
        (&lora.a_v_edge, &lora.b_v_edge, &lora.gate_v_edge)
    };
    let scale = lora.scale();
    let e_cnt = lora.graph.num_edges();
    (0..e_cnt)
        .map(|ei| {
            let r = lora.restriction_maps.slice(s![ei, slot, .., ..]).to_owned();
            let a = a_e.slice(s![ei, b, .., ..]);
            let bm = b_e.slice(s![ei, b, .., ..]);
            let g = gate.as_ref().map_or(1.0, |gm| gm[[ei, b]]);
            let low_rank = a.dot(&bm.t()); // [d_e, d_v]
            r + &(low_rank * (scale * g))
        })
        .collect()
}

#[test]
fn lora_matches_dense_materialized_f() {
    // With gates on — checks both the factored apply and that the adjoint's
    // gate placement (on the A^T r side) is equivalent to materializing
    // F = R + scale * g * A B^T and doing dense F^T F z.
    let (lora, _fixed, z) = setup_lora(false, true, 13);
    let r = lora.edge_residuals(&z);
    let mut lz = NodeState::zeros(z.dim());
    lora.laplacian_apply(&z, &mut lz);

    for b in 0..B {
        let fu = effective_blocks(&lora, 0, b);
        let fv = effective_blocks(&lora, 1, b);
        let f = dense_coboundary(&lora.graph.edges, &fu, &fv, None, N, D_V);
        let zf = flatten_batch(&z, b);

        let fz = f.dot(&zf);
        let r_flat: ndarray::Array1<f32> = r.slice(s![.., b, ..]).iter().cloned().collect();
        assert_close(&r_flat, &fz, 1e-4, 1e-4, "LoRA residuals vs dense F z");

        let dense = f.t().dot(&fz);
        let got = flatten_batch(&lz, b);
        assert_close(&got, &dense, 1e-4, 1e-4, "LoRA laplacian vs dense F^T F z");
    }
}

#[test]
fn create_directional_gathers_by_slot_tables() {
    let mut rng = Rng::new(17);
    let graph = grid_2x2_8way();
    let e_cnt = graph.num_edges();
    let maps = random_maps(&mut rng, e_cnt, D_E, D_V, 1.0);

    // Distinguishable factors: encode (node, slot) in every element.
    let a = Array5::from_shape_fn((N, B, K_DIRS, D_E, RANK), |(n, b, k, i, r)| {
        (n * 1000 + k * 10) as f32 + b as f32 * 0.5 + i as f32 * 0.01 + r as f32 * 0.001
    });
    let b_f = Array5::from_shape_fn((N, B, K_DIRS, D_V, RANK), |(n, b, k, j, r)| {
        -((n * 1000 + k * 10) as f32) - b as f32 * 0.5 - j as f32 * 0.01 - r as f32 * 0.001
    });
    let gate = ndarray::Array3::from_shape_fn((N, B, K_DIRS), |(n, b, k)| {
        (n * 100 + k) as f32 + b as f32 * 0.25
    });

    let lora = LoraGeometry::create_directional(
        graph.clone(),
        maps,
        &a,
        &b_f,
        Some(&gate),
        LORA_ALPHA,
    );

    for (ei, &[u, v]) in graph.edges.iter().enumerate() {
        let (u, v) = (u as usize, v as usize);
        let su = graph.dir_uv[ei] as usize;
        let sv = graph.dir_vu[ei] as usize;
        for bi in 0..B {
            assert_close(
                &lora.a_u_edge.slice(s![ei, bi, .., ..]).to_owned(),
                &a.slice(s![u, bi, su, .., ..]).to_owned(),
                0.0,
                0.0,
                "A_u_edge gather",
            );
            assert_close(
                &lora.a_v_edge.slice(s![ei, bi, .., ..]).to_owned(),
                &a.slice(s![v, bi, sv, .., ..]).to_owned(),
                0.0,
                0.0,
                "A_v_edge gather",
            );
            assert_close(
                &lora.b_u_edge.slice(s![ei, bi, .., ..]).to_owned(),
                &b_f.slice(s![u, bi, su, .., ..]).to_owned(),
                0.0,
                0.0,
                "B_u_edge gather",
            );
            assert_close(
                &lora.b_v_edge.slice(s![ei, bi, .., ..]).to_owned(),
                &b_f.slice(s![v, bi, sv, .., ..]).to_owned(),
                0.0,
                0.0,
                "B_v_edge gather",
            );
            assert_eq!(lora.gate_u_edge.as_ref().unwrap()[[ei, bi]], gate[[u, bi, su]]);
            assert_eq!(lora.gate_v_edge.as_ref().unwrap()[[ei, bi]], gate[[v, bi, sv]]);
        }
    }
}

#[test]
fn lora_laplacian_symmetric_psd() {
    let (lora, _fixed, _z) = setup_lora(false, true, 23);
    let mut rng = Rng::new(29);
    for _ in 0..6 {
        let a = rng.array3((N, B, D_V));
        let b = rng.array3((N, B, D_V));
        let mut la = NodeState::zeros(a.dim());
        let mut lb = NodeState::zeros(b.dim());
        lora.laplacian_apply(&a, &mut la);
        lora.laplacian_apply(&b, &mut lb);
        let a_lb: f32 = a.iter().zip(lb.iter()).map(|(&x, &y)| x * y).sum();
        let la_b: f32 = la.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        assert!(
            (a_lb - la_b).abs() <= 1e-2 * a_lb.abs().max(1.0),
            "LoRA L not symmetric: {a_lb} vs {la_b}"
        );
        let a_la: f32 = a.iter().zip(la.iter()).map(|(&x, &y)| x * y).sum();
        assert!(a_la >= -1e-4, "LoRA L not PSD: {a_la}");
    }
}
