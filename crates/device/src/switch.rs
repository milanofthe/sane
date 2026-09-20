//! Current-controlled switch (`W`): the one device that is topological rather
//! than physical -- it references another element's branch current, which
//! Verilog-A cannot express, so it lowers directly in Rust through the same
//! `lower_behavioral` contract as everything else.

use rsdag::{Crossing, ExprId, Graph};

use crate::common::{param, switch_g};
use crate::{BehavioralFragment, DeviceModel, FragmentEvent, Lowerer};

/// Current-controlled switch (`W`). Terminals: `[a, b]`, controlled by the
/// branch current of element `ctrl`. Conductance interpolates smoothly in log
/// space from `1/Roff` to `1/Ron` as `I(ctrl)` crosses the window
/// `[It - Ih, It + Ih]`; with `Ih = 0` the switch is hard at `It`. The window
/// edges are switching surfaces the transient integrator lands on; current
/// `G*(Va - Vb)` flows a -> b.
pub struct CSwitch {
    pub name: String,
    pub ctrl: String,
}

impl CSwitch {
    pub fn new(name: impl Into<String>, ctrl: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ctrl: ctrl.into(),
        }
    }

    fn lower(
        &self,
        ctx: &mut Graph,
        v: &[ExprId],
        control_i: &[ExprId],
    ) -> (Vec<ExprId>, Vec<FragmentEvent>) {
        let (ron, roff, it, ih) = (
            param(ctx, &self.name, "Ron"),
            param(ctx, &self.name, "Roff"),
            param(ctx, &self.name, "It"),
            param(ctx, &self.name, "Ih"),
        );
        let ictrl = control_i[0];
        let g_on = ctx.recip(ron);
        let g_off = ctx.recip(roff);
        let g = switch_g(ctx, ictrl, it, ih, g_on, g_off);
        let dv = ctx.sub(v[0], v[1]);
        let i = ctx.mul(g, dv);
        let neg_i = ctx.neg(i);
        // The window edges `It -/+ Ih` (one surface when `Ih = 0`).
        let lo = ctx.sub(it, ih);
        let hi = ctx.add(it, ih);
        let g_lo = ctx.sub(ictrl, lo);
        let g_hi = ctx.sub(ictrl, hi);
        let mut events = vec![FragmentEvent {
            g: g_lo,
            dir: Crossing::Either,
        }];
        if g_hi != g_lo {
            events.push(FragmentEvent {
                g: g_hi,
                dir: Crossing::Either,
            });
        }
        (vec![i, neg_i], events)
    }
}

impl DeviceModel for CSwitch {
    fn n_terminals(&self) -> usize {
        2
    }

    fn instance_name(&self) -> Option<&str> {
        Some(&self.name)
    }

    fn control_currents(&self) -> Vec<String> {
        vec![self.ctrl.clone()]
    }

    fn default_params(&self) -> Vec<(&'static str, f64)> {
        // Ih defaults to a narrow 1 uA window (near-hard but differentiable).
        vec![("Ron", 1.0), ("Roff", 1e6), ("It", 0.0), ("Ih", 1e-6)]
    }

    fn lower_behavioral(
        &self,
        lo: &mut Lowerer,
        terminal_v: &[ExprId],
        _terminal_vdot: &[ExprId],
        control_i: &[ExprId],
    ) -> BehavioralFragment {
        let (terminal_currents, events) = self.lower(lo.ctx(), terminal_v, control_i);
        BehavioralFragment {
            param_syms: Vec::new(),
            terminal_currents,
            residuals: Vec::new(),
            noise: Vec::new(),
            events,
            op_vars: Vec::new(),
            limits: Vec::new(),
        }
    }
}
