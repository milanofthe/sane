//! OpenVAF's evaluation alone, the way an OSDI host runs it: the 18 PSP103
//! instances of the 9-stage VACASK ring, each evaluated per Newton iteration
//! with the flags a DC iteration sets (resistive residual and Jacobian) and
//! the ones a transient iteration sets (resistive and reactive). The
//! counterpart of `sane-analysis`'s `va_eval` on the same ring.
//!
//! ```text
//! cargo run -q --release -p sane-osdi --example osdi_ring -- psp103.osdi ring_models.inc
//! ```

use std::collections::HashMap;
use std::time::Instant;

use sane_osdi::{OsdiDevice, OsdiLib};

/// The `.model` cards of `src`: name -> parameters (`+ key=value` lines).
fn cards(src: &str) -> HashMap<String, HashMap<String, f64>> {
    let mut out: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let mut cur = None;
    for line in src.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix(".model") {
            let name = rest.split_whitespace().next().unwrap().to_ascii_lowercase();
            out.insert(name.clone(), HashMap::new());
            cur = Some(name);
        } else if let (Some(kv), Some(name)) = (line.strip_prefix('+'), &cur) {
            if let Some((k, v)) = kv.trim().split_once('=') {
                let v: f64 = v.trim().parse().expect("a plain number");
                out.get_mut(name)
                    .unwrap()
                    .insert(k.trim().to_ascii_lowercase(), v);
            }
        }
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: osdi_ring lib.osdi ring_models.inc";
    let lib = OsdiLib::load(std::path::Path::new(&args.next().expect(usage))).expect("load");
    let cards = cards(&std::fs::read_to_string(args.next().expect(usage)).expect("read"));
    let module = lib.modules()[0].clone();

    // The ring's inverters: PMOS W = 20u, NMOS W = 10u, L = 1u, ld = ls = 0.5u.
    let geometry = |w: f64| {
        let ld = 0.5e-6;
        [
            ("w", w),
            ("l", 1e-6),
            ("ad", w * ld),
            ("as", w * ld),
            ("pd", 2.0 * (w + ld)),
            ("ps", 2.0 * (w + ld)),
        ]
    };
    let mut devices = Vec::new();
    for k in 0..9 {
        for (card, w) in [("psp103p", 20e-6), ("psp103n", 10e-6)] {
            let mut params: HashMap<String, f64> = cards[card]
                .iter()
                .filter(|(k, _)| module.has_param(k))
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            params.extend(geometry(w).map(|(k, v)| (k.to_string(), v)));
            let dev = OsdiDevice::new(
                format!("{card}{k}"),
                lib.clone(),
                module.clone(),
                params.into_iter().collect(),
                300.15,
            );
            dev.setup().expect("setup");
            devices.push(dev);
        }
    }

    // The ring's operating point: every stage at 0.66 V, supplies at 1.2 V
    // and 0 V; the internal nodes at their drain.
    let (vdd, vmid) = (1.2, 0.660607);
    let bias = |pmos: bool, n: usize| -> Vec<f64> {
        let (s, b) = if pmos { (vdd, vdd) } else { (0.0, 0.0) };
        let mut v = vec![vmid, vmid, s, b];
        v.resize(n, vmid);
        v
    };
    let n_active = |d: &OsdiDevice| d.debug_eval(&[vmid; 16]).0.len();
    let states: Vec<[Vec<f64>; 2]> = devices
        .iter()
        .enumerate()
        .map(|(k, d)| {
            let v = bias(k % 2 == 0, n_active(d));
            let v2 = v.iter().map(|x| x + 1e-6).collect();
            [v, v2]
        })
        .collect();

    for reactive in [false, true] {
        let ring = |i: usize| {
            let mut acc = 0.0;
            for (d, v) in devices.iter().zip(&states) {
                acc += d.host_eval(&v[i % 2], reactive);
            }
            acc
        };
        for i in 0..1000 {
            std::hint::black_box(ring(i));
        }
        let mut best = f64::INFINITY;
        for _ in 0..25 {
            let s = Instant::now();
            for i in 0..2000 {
                std::hint::black_box(ring(i));
            }
            best = best.min(s.elapsed().as_secs_f64() / 2000.0);
        }
        println!(
            "{} ring_us={:.2}",
            if reactive { "transient" } else { "dc" },
            best * 1e6
        );
    }
}
