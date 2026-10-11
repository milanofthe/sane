//! The analog operators (`ddt`, `idt`, `absdelay`, `laplace_nd`, the
//! noise sources) and the elementary functions they fall back on.

use super::*;

impl<'a, 'b> Lower<'a, 'b> {
    /// `ddt(q)`: a placeholder for the time derivative of the charge `q`,
    /// resolved row by row once the rows are whole (see
    /// [`Self::resolve_ddt`]).
    pub(super) fn ddt(&mut self, q: ExprId) -> ExprId {
        let name = format!("{}ddt#{}", self.inst, self.ddts.len());
        let m = self.ctx().sym(&name);
        let s = sym_of(self.lo.ctx(), m).expect("sym() yields a Symbol");
        self.ddts.push((s, q));
        m
    }

    /// `rows` as `i + d/dt q`: per row its current `i`, the placeholders
    /// read as zero, and its charge `q = sum_m (d row / d m) q_m` (empty
    /// without any `ddt`). A row must be affine in the placeholders; `ddt`
    /// inside a nonlinear function has no charge.
    pub(super) fn resolve_ddt(
        &mut self,
        rows: &[ExprId],
    ) -> Result<(Vec<ExprId>, Vec<ExprId>), String> {
        if self.ddts.is_empty() {
            return Ok((rows.to_vec(), Vec::new()));
        }
        let ddts = self.ddts.clone();
        let ctx = self.lo.ctx();
        let zero = ctx.zero();
        let rest: HashMap<SymbolId, ExprId> = ddts.iter().map(|&(m, _)| (m, zero)).collect();
        let markers: HashSet<SymbolId> = ddts.iter().map(|&(m, _)| m).collect();
        let mut charges = Vec::with_capacity(rows.len());
        for &r in rows {
            let free = ctx.free_symbols(r);
            let mut terms = Vec::new();
            for &(m, q) in ddts.iter().filter(|(m, _)| free.contains(m)) {
                let c = differentiate(ctx, r, m);
                if ctx.free_symbols(c).iter().any(|s| markers.contains(s)) {
                    return Err(format!(
                        "module {}: ddt in a nonlinear expression has no charge",
                        self.em.name
                    ));
                }
                terms.push(ctx.mul(c, q));
            }
            charges.push(ctx.reduce(rsdag::ReduceOp::Sum, terms));
        }
        Ok((rsdag::substitute(ctx, rows, &rest), charges))
    }

    /// Scale a flow expression by the parallel multiplicity `m` (no-op for m = 1).
    /// A noise generator of density `psd` (a table's densities `table`) in
    /// the contribution being lowered: its symbol `{inst}#noise{k}`, the
    /// density divided by the contribution's scale (see `noise_scale`).
    pub(super) fn noise_generator(
        &mut self,
        name: &str,
        psd: ExprId,
        flicker_exp: ExprId,
        table: Vec<(ExprId, ExprId)>,
    ) -> Result<ExprId, String> {
        let Some(scale) = self.noise_scale else {
            return Err(format!("{name} outside a contribution statement"));
        };
        let name = noise_symbol_name(&self.inst, self.noise.len());
        let g = self.ctx().sym(&name);
        let input = sym_of(self.lo.ctx(), g).expect("sym() yields a Symbol node");
        let per = |ctx: &mut Graph, e: ExprId| {
            if scale == 1.0 {
                e
            } else {
                let s = ctx.konst_f64(scale);
                ctx.div(e, s)
            }
        };
        let psd = per(self.ctx(), psd);
        let table = (table.into_iter())
            .map(|(f, p)| (f, per(self.ctx(), p)))
            .collect();
        self.noise.push(NoiseSource {
            input,
            psd,
            flicker_exp,
            table,
        });
        Ok(g)
    }

    pub(super) fn scale_m(&mut self, e: ExprId) -> ExprId {
        let mf = self.mfactor;
        if (mf - 1.0).abs() < f64::EPSILON {
            return e;
        }
        let m = self.ctx().konst_f64(mf);
        self.ctx().mul(m, e)
    }

    pub(super) fn call(&mut self, name: &str, args: &[Expr]) -> Result<ExprId, String> {
        let op = AnalogOp::from_name(name);
        if op == Some(AnalogOp::Ddt) {
            if args.is_empty() {
                return Err("ddt expects one argument".into());
            }
            let q = self.expr(&args[0])?;
            return Ok(self.ddt(q));
        }
        if matches!(op, Some(AnalogOp::WhiteNoise | AnalogOp::FlickerNoise)) {
            // A noise generator: a symbol, zero in every evaluation, that
            // enters the residuals where the contribution puts it.
            if args.is_empty() {
                return Err(format!("{name} expects a power-spectral-density argument"));
            }
            let psd = self.expr(&args[0])?;
            let flicker_exp = if op == Some(AnalogOp::FlickerNoise) {
                match args.get(1) {
                    Some(e) => self.expr(e)?,
                    None => self.ctx().one(),
                }
            } else {
                self.ctx().zero()
            };
            // A source on a conditional path is there where the path is.
            let reached = self.reached();
            let psd = self.ctx().mul(psd, reached);
            return self.noise_generator(name, psd, flicker_exp, Vec::new());
        }
        if op == Some(AnalogOp::NoiseTable) {
            // Tabular noise: a flat {f0, p0, f1, p1, ...} coefficient array,
            // chunked into (frequency, psd) points. Large-signal value is zero.
            if args.is_empty() {
                return Err(format!("{name} expects a coefficient array argument"));
            }
            let flat = self.coeffs(&args[0])?;
            let table: Vec<(ExprId, ExprId)> = flat
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| (c[0], c[1]))
                .collect();
            let reached = self.reached();
            let table = table
                .into_iter()
                .map(|(f, p)| (f, self.ctx().mul(p, reached)))
                .collect();
            let (psd, flicker_exp) = (self.ctx().zero(), self.ctx().zero());
            return self.noise_generator(name, psd, flicker_exp, table);
        }
        if op == Some(AnalogOp::LaplaceNd) {
            // Continuous Laplace filter H(s) = N(s)/D(s) with numerator/
            // denominator coefficient vectors -> controllable-canonical state
            // space (one differential state per denominator order).
            if self.cond_depth > 0 {
                return Err("laplace filter inside a conditional is not supported".into());
            }
            if args.len() < 3 {
                return Err("laplace_nd expects (input, num_coeffs, den_coeffs)".into());
            }
            let u = self.expr(&args[0])?;
            let num = self.coeffs(&args[1])?;
            let den = self.coeffs(&args[2])?;
            return self.lower_laplace_nd(u, &num, &den);
        }
        if op == Some(AnalogOp::Ddx) {
            // ddx(f, V(node)) = exact symbolic partial derivative of f w.r.t. the
            // node potential -- a direct hit for SANE's `differentiate`.
            if args.len() != 2 {
                return Err("ddx expects (expr, access)".into());
            }
            let f = self.expr(&args[0])?;
            if let Expr::Access { hi, lo, .. } = &args[1] {
                let (hn, _) = self.resolve_pair(hi, lo);
                let vexpr = self.node_voltage(&hn)?;
                if let Some(sym) = sym_of(self.lo.ctx(), vexpr) {
                    return Ok(differentiate(self.ctx(), f, sym));
                }
                // ground / non-symbol target: derivative is zero.
                return Ok(self.ctx().zero());
            }
            return Err("ddx second argument must be a node access like V(n)".into());
        }
        if matches!(op, Some(AnalogOp::Idt | AnalogOp::Idtmod)) {
            // idt(u[,ic]) -> state s with ds/dt = u (current `-u`, charge `s`);
            // the integral value is s. At DC (at rest) this enforces u = 0,
            // the steady-state condition for the usual NQS-charge integrand.
            // idtmod(u, ic, modulus[, offset]) additionally wraps the output.
            if self.cond_depth > 0 {
                return Err(format!("{name} inside a conditional is not supported"));
            }
            if args.is_empty() {
                return Err(format!("{name} expects at least one argument"));
            }
            // The ic argument: DC enforces the steady-state `u = 0` (matching
            // OpenVAF/OSDI), so ic is not a constraint -- a constant ic is
            // routed as the state's DC Newton seed (`.nodeset` semantics: it
            // breaks symmetry / picks the branch on multi-stable integrands,
            // and the transient then starts from that DC point). A
            // non-constant ic cannot seed a numeric solve; surface it as a
            // captured (catchable) warning instead of silently dropping it
            // (issue #43).
            let mut dc_seed = None;
            if args.len() >= 2 {
                let ic = self.expr(&args[1])?;
                match self.ctx().const_f64(ic) {
                    Some(c) if c != 0.0 => dc_seed = Some(c),
                    Some(_) => {}
                    None => sane_core::log::warn_captured(&format!(
                        "{name} initial condition is not a constant (module {}): it cannot \
                         seed the DC solve and is ignored; DC enforces the steady state u = 0",
                        self.em.name
                    )),
                }
            }
            let u = self.expr(&args[0])?;
            let sname = format!("idt{}", self.lo.extras.len());
            let s = self.lo.unknown_of(&self.inst, &sname);
            if let Some(c) = dc_seed {
                self.lo.extras.last_mut().expect("just minted").dc_seed = Some(c);
            }
            let ds = self.ddt(s);
            let r = self.ctx().sub(ds, u);
            self.branch_resid.push(r);
            if op == Some(AnalogOp::Idtmod) && args.len() >= 3 {
                // offset + ((s - offset) mod modulus)
                let modulus = self.expr(&args[2])?;
                let offset = if args.len() >= 4 {
                    self.expr(&args[3])?
                } else {
                    self.ctx().zero()
                };
                let shifted = self.ctx().sub(s, offset);
                let m = self.binary(BinOp::Mod, shifted, modulus);
                return Ok(self.ctx().add(offset, m));
            }
            return Ok(s);
        }
        if op == Some(AnalogOp::Absdelay) {
            // absdelay(u, td[, maxdelay]) -> transport delay. The source is
            // pinned into an auxiliary algebraic unknown y = u so the
            // integrator's dense output covers it; the delayed value d enters
            // through a history input the transient loop fills with y(t - td)
            // (residual `d - hist = 0`). td must be a constant/parameter
            // expression -- it compiles into the delay tape over the parameter
            // vector and is checked positive at transient setup. maxdelay is
            // redundant under that restriction (the history window adapts to
            // the actual delay) and is intentionally ignored.
            if self.cond_depth > 0 {
                return Err("absdelay inside a conditional is not supported".into());
            }
            if args.len() < 2 {
                return Err("absdelay expects (expr, td)".into());
            }
            let u = self.expr(&args[0])?;
            let td = self.expr(&args[1])?;
            let k = self.lo.delays.len();
            let hist_suffix = format!("dly{k}_hist");
            let hname = format!("{}.{hist_suffix}", self.inst);
            let src_extra = self.lo.extras.len();
            let y = self.lo.unknown_of(&self.inst, &format!("dly{k}_src"));
            let r = self.ctx().sub(y, u);
            self.branch_resid.push(r);
            let out_extra = self.lo.extras.len();
            let d = self.lo.unknown_of(&self.inst, &format!("dly{k}"));
            let hexpr = self.ctx().sym(&hname);
            let hist = sym_of(self.lo.ctx(), hexpr).expect("sym() yields a Symbol node");
            let r = self.ctx().sub(d, hexpr);
            self.branch_resid.push(r);
            self.lo.delays.push(LoweredDelay {
                src_extra,
                out_extra,
                hist,
                hist_suffix,
                tau: td,
            });
            return Ok(d);
        }
        if op == Some(AnalogOp::Analysis) {
            // `analysis("name", ...)` reports whether the active simulator analysis
            // matches. SANE lowers ONE analysis-agnostic symbolic model that all
            // analyses share (noise is extracted separately from the same model),
            // so there is no single "active analysis": this folds to 0. The model
            // keeps its full large-signal behaviour and skips analysis-specific
            // simplifications (e.g. BSIM3's `if (analysis("noise"))` derivative
            // pruning). The string argument is intentionally not lowered.
            return Ok(self.ctx().zero());
        }
        // User analog function: inline.
        if let Some(func) = self.em.functions.get(name).cloned() {
            return self.inline_fn(&func, args);
        }
        let a: Vec<ExprId> = args
            .iter()
            .map(|e| self.expr(e))
            .collect::<Result<_, _>>()?;
        self.builtin(name, &a)
    }

    /// A coefficient vector `{e0, e1, ...}`, each entry lowered.
    pub(super) fn coeffs(&mut self, e: &Expr) -> Result<Vec<ExprId>, String> {
        match e {
            Expr::Array(elems) => elems.iter().map(|x| self.expr(x)).collect(),
            _ => Err("expected a coefficient vector {..}".into()),
        }
    }

    /// Lower H(s)=N(s)/D(s) (coefficient vectors, ascending powers of s) to a
    /// controllable-canonical state space: states s_i = w^(i) where D(s)w = u,
    /// output y = sum_i num_i * w^(i). One differential unknown per denominator
    /// order (its trailing coefficients that are the constant zero dropped);
    /// the integrator chain and the defining equation are residual rows. The
    /// coefficients are expressions, a parameter's included.
    pub(super) fn lower_laplace_nd(
        &mut self,
        u: ExprId,
        num: &[ExprId],
        den: &[ExprId],
    ) -> Result<ExprId, String> {
        let zero_coeff = |l: &mut Self, c: ExprId| l.ctx().const_f64(c) == Some(0.0);
        let Some(k) = (0..den.len()).rev().find(|&i| !zero_coeff(self, den[i])) else {
            return Err("laplace_nd denominator is empty or all zero".into());
        };
        let n_num = (0..num.len())
            .rev()
            .find(|&i| !zero_coeff(self, num[i]))
            .map_or(0, |i| i + 1);
        if n_num > k + 1 {
            return Err(
                "laplace_nd with numerator order > denominator order is not supported".into(),
            );
        }
        let coeff = |i: usize| num.get(i).copied().filter(|_| i < n_num);
        if k == 0 {
            let Some(b0) = coeff(0) else {
                return Ok(self.ctx().zero());
            };
            let g = self.ctx().div(b0, den[0]);
            return Ok(self.ctx().mul(g, u));
        }
        let base = self.lo.extras.len();
        let mut s = Vec::with_capacity(k);
        let mut sdot = Vec::with_capacity(k);
        for i in 0..k {
            let si = self.lo.unknown_of(&self.inst, &format!("lap{base}_{i}"));
            s.push(si);
            let dsi = self.ddt(si);
            sdot.push(dsi);
        }
        for i in 0..k - 1 {
            let r = self.ctx().sub(sdot[i], s[i + 1]);
            self.branch_resid.push(r);
        }
        // a_k * s_{k-1}' + sum_{i<k} a_i s_i - u = 0
        let ctx = self.ctx();
        let mut terms: Vec<ExprId> = (0..k).map(|i| ctx.mul(den[i], s[i])).collect();
        terms.push(ctx.mul(den[k], sdot[k - 1]));
        let lhs = ctx.reduce(rsdag::ReduceOp::Sum, terms);
        let last = ctx.sub(lhs, u);
        self.branch_resid.push(last);
        // output y = sum_{i<k} b_i s_i (+ b_k * s_{k-1}' if deg num == k)
        let ctx = self.lo.ctx();
        let mut terms: Vec<ExprId> = (0..k)
            .filter_map(|i| coeff(i).map(|b| ctx.mul(b, s[i])))
            .collect();
        if let Some(bk) = coeff(k) {
            terms.push(ctx.mul(bk, sdot[k - 1]));
        }
        Ok(ctx.reduce(rsdag::ReduceOp::Sum, terms))
    }

    pub(super) fn builtin(&mut self, name: &str, a: &[ExprId]) -> Result<ExprId, String> {
        let ctx = self.lo.ctx();
        // Verilog-A-specific names first; everything else is the shared
        // elementary-function table in `sane_core::mathfn` (one canonical
        // construction per function, used by every expression frontend).
        Ok(match (AnalogOp::from_name(name), a.len()) {
            (Some(AnalogOp::Limexp), 1) => sane_device::safe_exp(ctx, a[0]),
            // `$limit(x, ...)` is path-only (junction limiting): it shapes Newton
            // convergence, never the converged result, so identity is exact.
            (Some(AnalogOp::Limit), _) if !a.is_empty() => a[0],
            // Time-domain filters change the dynamic response; not supported, so
            // reject rather than silently use the unfiltered signal. (absdelay
            // is supported and handled in `call` before argument lowering.)
            (Some(AnalogOp::TimeFilter), _) => {
                return Err(format!("time-domain filter '{name}' is not supported"))
            }
            (Some(AnalogOp::LaplaceNd | AnalogOp::ZFilter), _) => {
                return Err(format!("Laplace/Z filter '{name}' is not supported"))
            }
            _ => match (MathFn::from_veriloga(name)).filter(|f| f.arity() == a.len()) {
                Some(f) => f.lower(ctx, a),
                None => return Err(format!("function '{name}/{}' not yet lowered", a.len())),
            },
        })
    }
}
