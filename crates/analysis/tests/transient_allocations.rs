//! The transient steps without allocating: after a warm run (every
//! buffer sized), a run at half the step cap -- twice the steps, the same
//! span and breakpoints -- allocates exactly as much as one at the cap,
//! whatever its setup allocates. Its own test binary: it counts every allocation of the
//! process, and switches off what rsdag builds on a background thread as a
//! run goes (native code, specialized variants of the programs): a bounded
//! cost of its own, not of the stepping.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use sane_analysis::{Model, TransientOptions};

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const DECKS: [(&str, &str, f64); 4] = [
    (
        "rc",
        "* rc\nV1 in 0 sin(0 1 1k)\nR1 in out 1k\nC1 out 0 100n\n.end\n",
        1e-6,
    ),
    (
        "clipper",
        "* clipper\nV1 in 0 sin(0 2 1k)\nR1 in out 1k\nD1 out 0 dc\n\
         .model dc d is=1e-14 cjo=5n\nC1 out 0 10n\n.end\n",
        1e-6,
    ),
    (
        "inverter",
        "* inverter\nVDD vdd 0 1.8\nVIN in 0 pulse(0 1.8 0 10n 10n 40n 100n)\n\
         MP out in vdd vdd pm\nMN out in 0 0 nm\nCL out 0 100f\n\
         .model nm nmos level=1 vto=0.5 kp=200u\n.model pm pmos level=1 vto=-0.5 kp=100u\n.end\n",
        1e-9,
    ),
    (
        "delay",
        "* delay line\n.veriloga\nmodule vdel(a, b);\n  inout a, b; electrical a, b;\n\
         analog V(b) <+ absdelay(V(a), 2u);\nendmodule\n.endveriloga\n\
         V1 in 0 sin(0 1 100k)\nR1 in 0 1k\nN1 in mid vdel\nRS mid out 100\nR2 out 0 1k\nC2 out 0 1n\n.end\n",
        1e-7,
    ),
];

/// The allocations of a transient over `span` at the step cap `dt`.
fn allocations(m: &Model, span: f64, dt: f64) -> u64 {
    let opts = TransientOptions {
        dt_max: Some(dt),
        ..Default::default()
    };
    let pt = m.at(&[]).expect("parameters");
    let before = ALLOCS.load(Ordering::Relaxed);
    pt.transient(&[0.0, span], &opts).expect("transient");
    ALLOCS.load(Ordering::Relaxed) - before
}

#[test]
fn transient_steps_allocate_nothing() {
    sane_core::update_config(|c| {
        c.jit = false;
        c.tape_specialization = false;
        c.variants = false;
    });
    // rayon's global workers start once, on threads of their own: up before
    // anything is counted.
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    rayon::join(|| (), || ());
    for (name, deck, dt) in DECKS {
        let m = Model::from_netlist(deck).expect("model");
        let span = 200.0 * dt;
        allocations(&m, span, dt / 2.0);
        let (one, two) = (allocations(&m, span, dt), allocations(&m, span, dt / 2.0));
        assert_eq!(
            one, two,
            "{name}: {one} allocations at the step cap, {two} at half of it"
        );
    }
}
