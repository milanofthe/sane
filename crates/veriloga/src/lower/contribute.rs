//! Contributions: a flow contribution into the nodes' currents, a
//! potential one into a branch row, a switch branch into one merged
//! residual, the indirect `V(a) : f == 0` form.

use super::*;

impl<'a, 'b> Lower<'a, 'b> {
    pub(super) fn contribute(
        &mut self,
        access: &Access,
        hi: &str,
        lo: &Option<String>,
        rhs: &Expr,
    ) -> Result<(), String> {
        let (hn, ln) = self.resolve_pair(hi, lo);
        // A self-branch (both endpoints the same node): a zero potential
        // contribution is a no-op, as is a flow circulating within one node;
        // any other potential contribution is inconsistent.
        if hn == ln {
            if is_potential(access) {
                let v = self.expr(rhs)?;
                if self.ctx().const_f64(v) != Some(0.0) {
                    return Err(format!(
                        "potential contribution on the self-branch ({hn},{ln}) is not zero"
                    ));
                }
            }
            return Ok(());
        }
        // A noise generator in the RHS enters the residuals where the value
        // does, scaled with it (see `noise_scale`).
        let scale = if is_potential(access) {
            1.0
        } else {
            self.mfactor
        };
        let saved_scale = self.noise_scale.replace(scale);
        let val = self.expr(rhs);
        self.noise_scale = saved_scale;
        let val = val?;

        // Switch branch: its current unknown `i` is minted/stamped
        // unconditionally (in `setup`). Each arm only records its constraint as a
        // pseudo-variable, which the conditional machinery merges with `select`;
        // `finish` emits the merged constraint as this branch's residual. This is
        // what makes a `V(..) <+ ..` inside a conditional lowerable.
        let (skey, ssign) = canon(&hn, &ln);
        if is_potential(access) {
            if self.cond_depth > 0 && !self.switched.contains(&skey) {
                self.switched.push(skey.clone());
            }
            let short = self.cond_depth == 0
                && !self.probe_of.contains_key(&skey)
                && self.ctx().const_f64(val) == Some(0.0);
            if short {
                if !self.shorts.contains(&skey) {
                    self.shorts.push(skey.clone());
                }
            } else {
                self.open.insert(skey.clone());
            }
        }
        if let Some(&i) = self.switch_of.get(&skey) {
            // Reserved pseudo-variable names carry the per-branch residual / flow.
            let resid_key = switch_resid_key(&skey);
            if is_potential(access) {
                // V(hi,lo) <+ val  ->  (V(hi) - V(lo)) - val = 0. This branch is a
                // voltage source / node collapse; remember it so a later flow on
                // the same pair adds in parallel instead of clobbering this.
                self.switch_potential.insert(skey.clone());
                let vhi = self.node_voltage(&hn)?;
                let vlo = self.node_voltage(&ln)?;
                let dv = self.ctx().sub(vhi, vlo);
                let r = self.ctx().sub(dv, val);
                self.set_var(resid_key, r);
            } else if self.switch_potential.contains(&skey) {
                // The pair is already a voltage source (collapsed). A further flow
                // contribution (e.g. a compact model's thermal-noise current on a
                // separate named current branch over the same nodes) is a parallel
                // current injection: stamp it into the node balances and leave the
                // potential constraint intact. At DC a `white_noise` value is 0, so
                // this is harmless; the point is to not overwrite the collapse.
                let val = self.scale_m(val);
                self.add_cur(&hn, val);
                let neg = self.ctx().neg(val);
                self.add_cur(&ln, neg);
            } else {
                // I(hi,lo) <+ flow  ->  i - sum(flow) = 0, accumulating flows in
                // the canonical orientation (and scaled by parallel multiplicity).
                let val = self.scale_m(val);
                let signed = if ssign == 1.0 {
                    val
                } else {
                    self.ctx().neg(val)
                };
                let flow_key = switch_flow_key(&skey);
                let zero = self.ctx().zero();
                let prev = self.st.vars.get(&flow_key).copied().unwrap_or(zero);
                let sum = self.ctx().add(prev, signed);
                self.set_var(flow_key, sum);
                let r = self.ctx().sub(i, sum);
                self.set_var(resid_key, r);
            }
            return Ok(());
        }

        if is_potential(access) {
            // Potential (V / Temp / ...) contribution: a source branch. Mint a
            // flow unknown, stamp it into the two node balances, and add the
            // constraint `pot(hi) - pot(lo) - rhs = 0`.
            if self.cond_depth > 0 {
                // The switch-branch path (above) handles conditional potential
                // contributions; this is only reached when the same branch is
                // *also* a current probe, where the probe owns the unknown.
                return Err(
                    "potential contribution inside a conditional on a branch whose current is \
                     also probed is not supported"
                        .into(),
                );
            }
            let (key, sign) = canon(&hn, &ln);
            if self.probe_of.contains_key(&key) {
                // The branch current is probed (`I(a,b)` used somewhere): the
                // probe already owns the flow unknown and its node stamps, so
                // this contribution only REPLACES the probe's open-branch
                // constraint with the KVL -- the voltage-source-with-current-
                // sense idiom (ideal transformer, Branin transmission line).
                // Accumulated in the canonical branch orientation.
                let signed = if sign == 1.0 {
                    val
                } else {
                    self.ctx().neg(val)
                };
                let zero = self.ctx().zero();
                let prev = self.probe_potential.get(&key).copied().unwrap_or(zero);
                let sum = self.ctx().add(prev, signed);
                self.probe_potential.insert(key, sum);
                return Ok(());
            }
            let i = self
                .lo
                .branch_current_of(&self.inst, &format!("flow_{hn}_{ln}"));
            self.add_cur(&hn, i);
            let neg = self.ctx().neg(i);
            self.add_cur(&ln, neg);
            let vhi = self.node_voltage(&hn)?;
            let vlo = self.node_voltage(&ln)?;
            let dv = self.ctx().sub(vhi, vlo);
            let r = self.ctx().sub(dv, val);
            self.branch_resid.push(r);
        } else {
            // Parallel multiplicity: a flow contribution from `m` devices in
            // parallel is `m` times the single-device flow (the LRM `$mfactor`
            // rule). Potential contributions above set a voltage and are not
            // scaled. Internal-node KCL stays balanced (every flow term scales).
            let val = self.scale_m(val);
            let (key, sign) = canon(&hn, &ln);
            if self.probe_of.contains_key(&key) {
                // Probed branch: accumulate the flow; the current unknown is
                // stamped into the nodes at finish (with its constraint).
                let signed = if sign == 1.0 {
                    val
                } else {
                    self.ctx().neg(val)
                };
                let zero = self.ctx().zero();
                let prev = self.flow_sum.get(&key).copied().unwrap_or(zero);
                let s = self.ctx().add(prev, signed);
                self.flow_sum.insert(key, s);
            } else {
                // Flow (I / Pwr / ...) contribution: stamp into the node balances.
                self.add_cur(&hn, val);
                let neg = self.ctx().neg(val);
                self.add_cur(&ln, neg);
            }
        }
        Ok(())
    }

    /// Indirect contribution `access(hi,lo) : lhs == rhs`: introduce a branch
    /// flow unknown (stamped into the node balances) and the implicit-equation
    /// residual `lhs - rhs = 0` that determines it.
    pub(super) fn indirect(
        &mut self,
        hi: &str,
        lo: &Option<String>,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<(), String> {
        if self.cond_depth > 0 {
            return Err("indirect contribution inside a conditional is not supported".into());
        }
        let (hn, ln) = self.resolve_pair(hi, lo);
        // Lower the implicit equation first (may mint nested states), then mint
        // this branch's flow unknown so residual order matches mint order.
        let l = self.expr(lhs)?;
        let r = self.expr(rhs)?;
        let i = self
            .lo
            .branch_current_of(&self.inst, &format!("ind_{hn}_{ln}"));
        self.add_cur(&hn, i);
        let neg = self.ctx().neg(i);
        self.add_cur(&ln, neg);
        let resid = self.ctx().sub(l, r);
        self.branch_resid.push(resid);
        Ok(())
    }
}

/// Canonical branch key (sorted node names) and the orientation sign of the
/// written `(a, b)` relative to it (+1 if already sorted, -1 if swapped).
/// Reserved pseudo-variable name carrying a switch branch's merged residual.
/// The `\u{1}` prefix and separators cannot occur in a Verilog-A identifier, so
/// these never collide with a real variable nor with another branch's key
/// (joining node names with `_` would alias `(a_b,c)` and `(a,b_c)`).
pub(super) fn switch_resid_key(key: &(String, String)) -> String {
    format!("\u{1}swr\u{1}{}\u{1}{}", key.0, key.1)
}

/// Reserved pseudo-variable name accumulating a switch branch's flow.
pub(super) fn switch_flow_key(key: &(String, String)) -> String {
    format!("\u{1}swf\u{1}{}\u{1}{}", key.0, key.1)
}

/// Collect branches that receive a potential contribution `V(..) <+ ..` inside a
/// conditional (the switch-branch idiom). `cond` is true once inside any
/// `if`/`case`/`for`/`while` body. The returned canonical keys get their flow
/// unknown minted unconditionally so the per-arm constraint can be merged.
pub(super) fn collect_switch_keys(
    l: &Lower,
    s: &Stmt,
    cond: bool,
    out: &mut Vec<(String, String)>,
) {
    match s {
        Stmt::Contribution { access, hi, lo, .. } => {
            if cond && is_potential(access) {
                let (hn, ln) = l.resolve_pair(hi, lo);
                if hn == ln {
                    return; // self-branch: nothing to switch
                }
                let (key, _) = canon(&hn, &ln);
                if !out.contains(&key) {
                    out.push(key);
                }
            }
        }
        Stmt::Block(ss) => ss.iter().for_each(|s| collect_switch_keys(l, s, cond, out)),
        Stmt::If { then, els, .. } => {
            collect_switch_keys(l, then, true, out);
            if let Some(e) = els {
                collect_switch_keys(l, e, true, out);
            }
        }
        Stmt::Case { items, default, .. } => {
            for (_, body) in items {
                collect_switch_keys(l, body, true, out);
            }
            if let Some(d) = default {
                collect_switch_keys(l, d, true, out);
            }
        }
        Stmt::For { body, .. } | Stmt::While { body, .. } => {
            collect_switch_keys(l, body, true, out)
        }
        Stmt::InitialStep(b) => collect_switch_keys(l, b, cond, out),
        Stmt::Event { body, .. } => collect_switch_keys(l, body, true, out),
        _ => {}
    }
}

/// Collect the canonical keys of branches whose current is probed `I(a,b)` in a
/// statement (recursively), so they can be promoted to explicit unknowns.
pub(super) fn collect_probe_keys(l: &Lower, s: &Stmt, out: &mut Vec<(String, String)>) {
    match s {
        Stmt::Block(ss) => ss.iter().for_each(|s| collect_probe_keys(l, s, out)),
        Stmt::Assign { rhs, .. } => collect_expr_probes(l, rhs, out),
        Stmt::Contribution { rhs, .. } => collect_expr_probes(l, rhs, out),
        Stmt::Indirect { lhs, rhs, .. } => {
            collect_expr_probes(l, lhs, out);
            collect_expr_probes(l, rhs, out);
        }
        Stmt::If { cond, then, els } => {
            collect_expr_probes(l, cond, out);
            collect_probe_keys(l, then, out);
            if let Some(e) = els {
                collect_probe_keys(l, e, out);
            }
        }
        Stmt::Case {
            sel,
            items,
            default,
        } => {
            collect_expr_probes(l, sel, out);
            for (labels, body) in items {
                labels.iter().for_each(|e| collect_expr_probes(l, e, out));
                collect_probe_keys(l, body, out);
            }
            if let Some(d) = default {
                collect_probe_keys(l, d, out);
            }
        }
        Stmt::For {
            init,
            cond,
            step,
            body,
        } => {
            collect_probe_keys(l, init, out);
            collect_expr_probes(l, cond, out);
            collect_probe_keys(l, step, out);
            collect_probe_keys(l, body, out);
        }
        Stmt::While { cond, body } => {
            collect_expr_probes(l, cond, out);
            collect_probe_keys(l, body, out);
        }
        Stmt::InitialStep(b) => collect_probe_keys(l, b, out),
        Stmt::Event { body, .. } => collect_probe_keys(l, body, out),
        Stmt::Call { args, .. } => args.iter().for_each(|a| collect_expr_probes(l, a, out)),
        Stmt::SysTask { args, .. } => args.iter().for_each(|a| collect_expr_probes(l, a, out)),
        Stmt::Empty | Stmt::IgnoredCall => {}
    }
}

pub(super) fn collect_expr_probes(l: &Lower, e: &Expr, out: &mut Vec<(String, String)>) {
    match e {
        Expr::Access { access, hi, lo, .. } => {
            if !is_potential(access) {
                let (hn, ln) = l.resolve_pair(hi, lo);
                let (key, _) = canon(&hn, &ln);
                if !out.contains(&key) {
                    out.push(key);
                }
            }
        }
        Expr::Unary { arg, .. } => collect_expr_probes(l, arg, out),
        Expr::Binary { lhs, rhs, .. } => {
            collect_expr_probes(l, lhs, out);
            collect_expr_probes(l, rhs, out);
        }
        Expr::Ternary {
            cond, then, els, ..
        } => {
            collect_expr_probes(l, cond, out);
            collect_expr_probes(l, then, out);
            collect_expr_probes(l, els, out);
        }
        Expr::Call { args, .. } | Expr::SysFn { args, .. } | Expr::Array(args) => {
            args.iter().for_each(|a| collect_expr_probes(l, a, out))
        }
        Expr::Num(_) | Expr::Str(_) | Expr::Ident(_, _) => {}
    }
}

/// Whether an access function is a potential (V/Temp/...) vs a flow (I/Pwr/...).
/// This is what makes electrical and thermal (and other conservative
/// disciplines) lower uniformly: a potential contribution is a source branch, a
/// flow contribution stamps into the node balance.
/// Whether `s` contributes a potential anywhere, whatever its branches
/// merge to (a short already merged reads as a self-branch).
pub(super) fn contributes_potential(s: &Stmt) -> bool {
    match s {
        Stmt::Contribution { access, .. } => is_potential(access),
        Stmt::Block(ss) => ss.iter().any(contributes_potential),
        Stmt::If { then, els, .. } => {
            contributes_potential(then) || els.as_deref().is_some_and(contributes_potential)
        }
        Stmt::Case { items, default, .. } => {
            items.iter().any(|(_, b)| contributes_potential(b))
                || default.as_deref().is_some_and(contributes_potential)
        }
        Stmt::For { body, .. }
        | Stmt::While { body, .. }
        | Stmt::InitialStep(body)
        | Stmt::Event { body, .. } => contributes_potential(body),
        _ => false,
    }
}
