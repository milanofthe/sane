//! Elaboration: turn a parsed `ast::Module` into an `ElaboratedModule` ready for
//! lowering. This is the declaration-collection / name-resolution pass:
//!
//! - the node set (ports + internal electrical nets),
//! - parameters with their defaults constant-folded to `f64` (for
//!   `default_params()`), keeping range constraints for later validation,
//! - branch and analog-function tables.
//!
//! Parameters stay SYMBOLIC for lowering (bound numerically by the netlist at
//! solve time); only their *defaults* are folded here. Loop unrolling and
//! analog-function inlining are done during lowering (WP3), driven by the
//! tables collected here, since they interact with the procedural-assignment
//! environment.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::ast::*;
use crate::error::{Diagnostic, Diagnostics, Span};

#[derive(Clone, Debug)]
pub struct ResolvedParam {
    pub name: String,
    pub ty: VarType,
    pub default: f64,
    pub ranges: Vec<RangeConstraint>,
    /// `(* type="instance" *)`: settable per instance, not only on the card.
    pub is_instance: bool,
    /// `(* units = "..." *)` annotation, if any.
    pub units: Option<String>,
    /// `(* desc = "..." *)` annotation, if any.
    pub desc: Option<String>,
}

/// An operating-point variable: a module variable annotated `(* desc = ... *)`
/// (the compact-model OPP idiom, e.g. gm / vth / ids). Its final lowered value
/// is exported per instance so analyses can report it after a solve.
#[derive(Clone, Debug)]
pub struct OpVarDecl {
    pub name: String,
    pub desc: String,
    pub units: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ElaboratedModule {
    pub name: String,
    pub ports: Vec<String>,
    /// All electrical nodes (ports first, then internal nets), de-duplicated.
    pub nodes: Vec<String>,
    /// Nodes not in `ports` (become internal DAE unknowns).
    pub internal_nodes: Vec<String>,
    pub params: Vec<ResolvedParam>,
    /// String parameters (compile-time model-variant selectors): name ->
    /// default literal. Folded during lowering; no numeric symbol.
    pub string_params: HashMap<String, String>,
    /// Operating-point variables (`(* desc *)`-annotated module variables), in
    /// declaration order.
    pub opvars: Vec<OpVarDecl>,
    /// `aliasparam` table: alias -> target (a parameter name, or a system
    /// function carried with its `$` prefix, e.g. `$mfactor`). Setting the
    /// alias in a deck sets the target; the alias is not itself a parameter.
    pub aliases: HashMap<String, String>,
    pub vars: Vec<(String, VarType)>,
    /// Named branch -> (hi node, lo node).
    pub branches: HashMap<String, (String, String)>,
    pub functions: HashMap<String, AnalogFunction>,
    pub analog: Vec<Stmt>,
    /// Parameter defaults with `&'static` names, for `DeviceModel::default_params`
    /// (the trait wants `&'static str`; a VA module is loaded once, so its param
    /// names are leaked once here -- bounded and amortized).
    pub default_params_static: Vec<(&'static str, f64)>,
    /// Parameter name -> default, for the by-name lookup of
    /// [`DeviceModel::param_default`](sane_device::DeviceModel::param_default).
    pub default_map: HashMap<String, f64>,
    /// Lower-cased deck key (parameter or non-system alias) -> exact parameter
    /// name, for [`DeviceModel::canonical_param`](sane_device::DeviceModel::canonical_param).
    pub canon: HashMap<String, &'static str>,
    /// Lower-cased aliases of system functions (`aliasparam m = $mfactor`).
    pub sysfn_aliases: HashSet<String>,
}

/// Intern a parameter name as a `&'static str` for `DeviceModel::default_params`
/// (which requires `'static`). A process-wide pool deduplicates by content, so
/// the leak is bounded by the number of *distinct* parameter names ever seen --
/// re-elaborating the same model (tests, hot-reload) reuses the interned string
/// instead of leaking a fresh copy each time.
fn intern_static(name: &str) -> &'static str {
    use rustc_hash::FxHashSet as HashSet;
    use std::sync::{Mutex, OnceLock};
    static POOL: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let pool = POOL.get_or_init(|| Mutex::new(HashSet::default()));
    let mut set = pool.lock().unwrap();
    if let Some(&s) = set.get(name) {
        return s;
    }
    let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
    set.insert(leaked);
    leaked
}

pub fn elaborate(m: &Module) -> Result<ElaboratedModule, Diagnostics> {
    let mut errs: Vec<Diagnostic> = Vec::new();

    // Node set: ports first, then any other declared net, de-duplicated.
    let mut nodes: Vec<String> = Vec::new();
    for p in &m.ports {
        if !nodes.contains(p) {
            nodes.push(p.clone());
        }
    }
    for n in &m.nets {
        if !nodes.contains(n) {
            nodes.push(n.clone());
        }
    }
    let internal_nodes: Vec<String> = nodes
        .iter()
        .filter(|n| !m.ports.contains(n))
        .cloned()
        .collect();

    // Parameters: fold each default to a constant, with earlier params in scope.
    let mut env: HashMap<String, f64> = HashMap::default();
    let mut params = Vec::new();
    for pd in &m.params {
        // An unfoldable default (depends on a runtime quantity) is non-fatal: the
        // parameter stays symbolic and is normally overridden by the deck; fall
        // back to 0.0 for the `default_params()` value.
        let default = const_eval(&pd.default, &env).unwrap_or(0.0);
        env.insert(pd.name.clone(), default);
        let attr_str = |key: &str| {
            pd.attrs
                .iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| match v {
                    AttrVal::Str(s) => Some(s.clone()),
                    _ => None,
                })
        };
        params.push(ResolvedParam {
            name: pd.name.clone(),
            ty: pd.ty,
            default,
            ranges: pd.ranges.clone(),
            is_instance: attr_str("type").as_deref() == Some("instance"),
            units: attr_str("units"),
            desc: attr_str("desc"),
        });
    }

    // Alias table: an alias must not shadow a parameter, and its target must be
    // a declared parameter or a supported system function ($mfactor).
    let mut aliases = HashMap::default();
    for a in &m.aliases {
        if m.params.iter().any(|p| p.name == a.alias) {
            errs.push(Diagnostic::new(
                format!(
                    "aliasparam '{}' collides with a parameter of the same name",
                    a.alias
                ),
                a.span,
            ));
            continue;
        }
        let target_ok = if let Some(sys) = a.target.strip_prefix('$') {
            sys == "mfactor"
        } else {
            m.params.iter().any(|p| p.name == a.target)
        };
        if !target_ok {
            errs.push(Diagnostic::new(
                format!(
                    "aliasparam '{}' targets unknown parameter or unsupported system \
                     function '{}'",
                    a.alias, a.target
                ),
                a.span,
            ));
            continue;
        }
        aliases.insert(a.alias.clone(), a.target.clone());
    }

    // Branch table; endpoints must be nodes (or ground "0").
    let mut branches = HashMap::default();
    for b in &m.branches {
        for ep in [&b.hi, &b.lo] {
            if ep != "0" && !nodes.contains(ep) {
                errs.push(Diagnostic::new(
                    format!("branch '{}' references unknown node '{}'", b.name, ep),
                    Span::default(),
                ));
            }
        }
        branches.insert(b.name.clone(), (b.hi.clone(), b.lo.clone()));
    }

    let functions: HashMap<String, AnalogFunction> = m
        .functions
        .iter()
        .map(|f| (f.name.clone(), f.clone()))
        .collect();
    let vars: Vec<(String, VarType)> = m.vars.iter().map(|v| (v.name.clone(), v.ty)).collect();
    // Operating-point variables: any module variable with a `desc` attribute.
    let opvars: Vec<OpVarDecl> = m
        .vars
        .iter()
        .filter_map(|v| {
            let get = |key: &str| {
                v.attrs
                    .iter()
                    .find(|(k, _)| k == key)
                    .and_then(|(_, a)| match a {
                        AttrVal::Str(s) => Some(s.clone()),
                        _ => None,
                    })
            };
            get("desc").map(|desc| OpVarDecl {
                name: v.name.clone(),
                desc,
                units: get("units"),
            })
        })
        .collect();

    if !errs.is_empty() {
        return Err(Diagnostics {
            file: String::new(),
            items: errs,
            map: None,
        });
    }

    let default_params_static: Vec<(&'static str, f64)> = params
        .iter()
        .map(|p| (intern_static(&p.name), p.default))
        .collect();
    let default_map: HashMap<String, f64> =
        params.iter().map(|p| (p.name.clone(), p.default)).collect();
    let mut canon: HashMap<String, &'static str> = params
        .iter()
        .map(|p| (p.name.to_ascii_lowercase(), intern_static(&p.name)))
        .collect();
    let mut sysfn_aliases: HashSet<String> = HashSet::default();
    for (alias, target) in &aliases {
        if target.starts_with('$') {
            sysfn_aliases.insert(alias.to_ascii_lowercase());
        } else if let Some(&c) = canon.get(&target.to_ascii_lowercase()) {
            canon.entry(alias.to_ascii_lowercase()).or_insert(c);
        }
    }

    Ok(ElaboratedModule {
        name: m.name.clone(),
        ports: m.ports.clone(),
        nodes,
        internal_nodes,
        params,
        string_params: m.string_params.iter().cloned().collect(),
        opvars,
        aliases,
        vars,
        branches,
        functions,
        analog: m.analog.clone(),
        default_params_static,
        default_map,
        canon,
        sysfn_aliases,
    })
}

impl ElaboratedModule {
    /// Check bound parameter values against their declared `from`/`exclude`
    /// ranges, returning a message per violation. Compact models (BSIM, VBIC,
    /// PSP) declare these ranges precisely; enforcing them turns a mis-entered
    /// `.model` card into a clear diagnostic instead of a silently invalid
    /// operating point. `values` maps (unscoped) parameter names to their bound
    /// values; a parameter absent from `values` is checked at its default.
    pub fn check_param_ranges(&self, values: &HashMap<String, f64>) -> Vec<String> {
        let mut out = Vec::new();
        for p in &self.params {
            if p.ranges.is_empty() {
                continue;
            }
            let v = values.get(&p.name).copied().unwrap_or(p.default);
            if !range_admits(&p.ranges, v, values) {
                out.push(format!(
                    "parameter '{}' = {v} is outside its declared valid range",
                    p.name
                ));
            }
        }
        out
    }
}

/// Numeric value of a range bound (`inf` -> +/-infinity; a bound expression is
/// const-folded against `env`, falling back to +/-infinity if unfoldable so an
/// unknown bound never spuriously rejects a value).
fn bound_val(b: &Bound, env: &HashMap<String, f64>, upper: bool) -> f64 {
    match b {
        Bound::Inf(pos) => {
            if *pos {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        }
        Bound::Inclusive(e) | Bound::Exclusive(e) => const_eval(e, env).unwrap_or(if upper {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }),
    }
}

/// Whether `v` lies inside a single range (open/closed per its bound kinds).
fn in_range(r: &RangeConstraint, v: f64, env: &HashMap<String, f64>) -> bool {
    let lo = bound_val(&r.lo, env, false);
    let hi = bound_val(&r.hi, env, true);
    let lo_ok = if matches!(r.lo, Bound::Exclusive(_)) {
        v > lo
    } else {
        v >= lo
    };
    let hi_ok = if matches!(r.hi, Bound::Exclusive(_)) {
        v < hi
    } else {
        v <= hi
    };
    lo_ok && hi_ok
}

/// Verilog-A range admittance: a value must lie in the union of the `from`
/// ranges (if any are declared) and in none of the `exclude` ranges.
fn range_admits(ranges: &[RangeConstraint], v: f64, env: &HashMap<String, f64>) -> bool {
    let froms: Vec<&RangeConstraint> = ranges.iter().filter(|r| r.include).collect();
    let in_from = froms.is_empty() || froms.iter().any(|r| in_range(r, v, env));
    let excluded = ranges
        .iter()
        .filter(|r| !r.include)
        .any(|r| in_range(r, v, env));
    in_from && !excluded
}

/// Evaluate a compile-time-constant expression to `f64`, with `env` binding
/// in-scope constants (earlier parameter defaults). Returns `None` for anything
/// that depends on the solution / runtime (`V()`, `$temperature`, unknown ident).
/// Also used by the lowering to fold constant subexpressions.
pub fn const_eval(e: &Expr, env: &HashMap<String, f64>) -> Option<f64> {
    const_eval_with(e, &|name| env.get(name).copied())
}

/// Resolver-based compile-time evaluation: `resolve` maps an identifier to its
/// constant value (parameters and constant-shadowed variables), avoiding the
/// cost of materializing a combined environment per call.
pub(crate) fn const_eval_with(e: &Expr, resolve: &dyn Fn(&str) -> Option<f64>) -> Option<f64> {
    Some(match e {
        Expr::Num(n) => *n,
        Expr::Str(_) | Expr::Array(_) => return None,
        Expr::Ident(name, _) => resolve(name)?,
        Expr::Unary { op, arg, .. } => {
            let v = const_eval_with(arg, resolve)?;
            match op {
                UnOp::Neg => -v,
                UnOp::Not => bool_f64(v == 0.0),
            }
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            let a = const_eval_with(lhs, resolve)?;
            let b = const_eval_with(rhs, resolve)?;
            match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
                BinOp::Mod => a - b * (a / b).trunc(),
                BinOp::Pow => a.powf(b),
                BinOp::Lt => bool_f64(a < b),
                BinOp::Gt => bool_f64(a > b),
                BinOp::Le => bool_f64(a <= b),
                BinOp::Ge => bool_f64(a >= b),
                BinOp::Eq => bool_f64(a == b),
                BinOp::Ne => bool_f64(a != b),
                BinOp::And => bool_f64(a != 0.0 && b != 0.0),
                BinOp::Or => bool_f64(a != 0.0 || b != 0.0),
            }
        }
        Expr::Ternary {
            cond, then, els, ..
        } => {
            if const_eval_with(cond, resolve)? != 0.0 {
                const_eval_with(then, resolve)?
            } else {
                const_eval_with(els, resolve)?
            }
        }
        Expr::Call { name, args, .. } => {
            // `analysis(...)` is a compile-time 0 for SANE's analysis-agnostic
            // model (see the lowering), so a flag like `doNoise = analysis("noise")`
            // folds and its guarded blocks resolve statically.
            if name == "analysis" {
                return Some(0.0);
            }
            let v: Vec<f64> = args
                .iter()
                .map(|a| const_eval_with(a, resolve))
                .collect::<Option<_>>()?;
            return const_builtin(name, &v);
        }
        // Access functions are never compile-time constants.
        Expr::Access { .. } => return None,
        // A simulator-option default is the one system function with a
        // compile-time value. `$temperature` / `$vt` are runtime now (they lower
        // to the global temperature symbol so a sweep can vary them), so they are
        // NOT compile-time constants.
        Expr::SysFn { name, args, .. } => {
            return match name.as_str() {
                "simparam" if args.len() >= 2 => const_eval_with(&args[1], resolve),
                // `$param_given(X)` is a compile-time fact of the instance
                // binding. It is routed through the resolver as the pseudo-name
                // `$given(X)` so contexts that know the given-set (the lowering
                // and the collapse scan) fold it, while parameter-default
                // evaluation (which has no instance) safely stays non-constant.
                "param_given" => match args.first() {
                    Some(Expr::Ident(p, _)) => resolve(&format!("$given({p})")),
                    _ => None,
                },
                _ => None,
            };
        }
    })
}

pub(crate) fn const_builtin(name: &str, a: &[f64]) -> Option<f64> {
    let v = |i: usize| a.get(i).copied();
    Some(match name {
        "exp" | "limexp" => v(0)?.exp(),
        "ln" => v(0)?.ln(),
        "log" => v(0)?.log10(),
        "sqrt" => v(0)?.sqrt(),
        "abs" => v(0)?.abs(),
        "sin" => v(0)?.sin(),
        "cos" => v(0)?.cos(),
        "tan" => v(0)?.tan(),
        "atan" => v(0)?.atan(),
        "tanh" => v(0)?.tanh(),
        "sinh" => v(0)?.sinh(),
        "cosh" => v(0)?.cosh(),
        "floor" => v(0)?.floor(),
        "ceil" => v(0)?.ceil(),
        "pow" => v(0)?.powf(v(1)?),
        "min" => v(0)?.min(v(1)?),
        "max" => v(0)?.max(v(1)?),
        _ => return None,
    })
}

pub(crate) fn bool_f64(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}
