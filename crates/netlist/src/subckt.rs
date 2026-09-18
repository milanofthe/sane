//! Subcircuit flattening: expand `.subckt`/`X` hierarchy into a flat list of
//! primitive element lines before parsing.
//!
//! Each `X` instance is expanded recursively: port nodes map to the caller's
//! nodes, internal nodes and element names are prefixed with the instance path
//! (`X1.`, `X1.X2.`, ...), and subcircuit/instance parameters are folded into a
//! local numeric environment used to evaluate `{...}` value expressions.
//! Subcircuit names are a global namespace (a common simplification); `.model`
//! cards inside a subckt are emitted unprefixed (global).

use rustc_hash::FxHashMap as HashMap;

use crate::expr::{lex, resolve_value, value_expr, Tok};
use crate::preprocess::Line;
use crate::{err, ParseError};

struct Subckt {
    ports: Vec<String>,
    defaults: Vec<(String, String)>, // (param, expr)
    body: Vec<Line>,
}

/// Token layout after the element name: `(number of leading node tokens,
/// positions that are element-NAME references)`. Name references (controlling
/// elements of F/H/W, the inductors of K) must be prefixed with the instance
/// path, just like the elements they point at. `None` for unknown types.
fn element_layout(kind: char) -> Option<(usize, &'static [usize])> {
    match kind {
        'R' | 'C' | 'L' | 'V' | 'I' | 'D' | 'P' => Some((2, &[])),
        'Q' | 'J' | 'Z' => Some((3, &[])), // BJT / JFET / MESFET: 3 terminals
        'E' | 'G' | 'M' | 'S' | 'T' | 'O' | 'U' => Some((4, &[])), // 4-terminal (incl. transmission lines)
        'F' | 'H' | 'W' => Some((2, &[2])), // 2 nodes; token 2 = controller name
        'K' => Some((0, &[0, 1])),          // two inductor name references
        // `N` (Verilog-A, variable port count) and `B` (behavioral, with node /
        // branch references buried in its value expression) have layouts the
        // simple "leading node count" model cannot express; they fall through to
        // the `None` arm, so an instance of either *inside a `.subckt`* is not yet
        // port-remapped. Flag rather than silently mis-wire: see `expand`.
        _ => None,
    }
}

fn is_ground(key: &str) -> bool {
    matches!(key, "0" | "gnd" | "ground")
}

/// Flatten all subcircuit instances. Returns a flat line list with no
/// `.subckt`/`.ends`/`X`.
pub fn flatten(lines: &[Line], global_env: &HashMap<String, f64>) -> Result<Vec<Line>, ParseError> {
    let (subckts, top) = collect(lines)?;
    // `.global <node>...`: these node names keep their identity inside every
    // subcircuit instance (supply rails), instead of being instance-prefixed
    let globals: std::collections::HashSet<String> = top
        .iter()
        .filter(|l| l.tokens[0].eq_ignore_ascii_case(".global"))
        .flat_map(|l| l.tokens[1..].iter().map(|t| t.to_ascii_lowercase()))
        .collect();

    let mut out = Vec::new();
    for line in &top {
        let head = &line.tokens[0];
        if head.eq_ignore_ascii_case(".global") {
            continue; // consumed above
        }
        if head.starts_with('X') || head.starts_with('x') {
            let mut nmap = HashMap::default();
            let inst = parse_instance(line, &mut nmap, "", &globals)?;
            expand(&inst, "", global_env, &subckts, &mut out, 0, &globals)?;
        } else {
            out.push(line.clone());
        }
    }
    Ok(out)
}

/// Split lines into subcircuit definitions (global) and top-level lines.
fn collect(lines: &[Line]) -> Result<(HashMap<String, Subckt>, Vec<Line>), ParseError> {
    let mut subckts = HashMap::default();
    let mut top = Vec::new();
    let mut stack: Vec<(String, Subckt)> = Vec::new();

    for line in lines {
        let head = &line.tokens[0];
        if head.eq_ignore_ascii_case(".subckt") {
            if line.tokens.len() < 2 {
                return Err(err(line.no, ".subckt needs a name"));
            }
            let name = line.tokens[1].to_ascii_lowercase();
            let mut ports = Vec::new();
            let mut defaults = Vec::new();
            for t in &line.tokens[2..] {
                if t.eq_ignore_ascii_case("params:") {
                    continue;
                }
                if let Some((k, v)) = t.split_once('=') {
                    defaults.push((k.to_ascii_lowercase(), v.to_string()));
                } else {
                    ports.push(t.clone());
                }
            }
            stack.push((
                name,
                Subckt {
                    ports,
                    defaults,
                    body: Vec::new(),
                },
            ));
        } else if head.eq_ignore_ascii_case(".ends") {
            let (name, sub) = stack
                .pop()
                .ok_or_else(|| err(line.no, ".ends without .subckt"))?;
            subckts.insert(name, sub);
        } else if let Some((_, sub)) = stack.last_mut() {
            sub.body.push(line.clone());
        } else {
            top.push(line.clone());
        }
    }
    Ok((subckts, top))
}

/// A parsed `X` instance with its connection nodes already resolved to the
/// caller's namespace.
struct Instance {
    inst_name: String,
    subname: String,
    conn: Vec<String>,
    params: Vec<(String, String)>, // (param, expr in caller env)
    line: usize,
}

/// Parse an `X` line, resolving its connection-node tokens through `nmap`
/// (with the caller's `prefix` for internal nodes).
fn parse_instance(
    line: &Line,
    nmap: &mut HashMap<String, String>,
    prefix: &str,
    globals: &std::collections::HashSet<String>,
) -> Result<Instance, ParseError> {
    let tok = &line.tokens;
    let kv_start = tok
        .iter()
        .position(|t| t.contains('='))
        .unwrap_or(tok.len());
    if kv_start < 3 {
        return Err(err(line.no, "X instance needs nodes and a subckt name"));
    }
    let head = &tok[1..kv_start];
    let subname = head.last().unwrap().to_ascii_lowercase();
    let conn: Vec<String> = head[..head.len() - 1]
        .iter()
        .map(|n| resolve_node(n, prefix, nmap, globals))
        .collect();
    let params: Vec<(String, String)> = tok[kv_start..]
        .iter()
        .filter_map(|t| t.split_once('='))
        .map(|(k, v)| (k.to_ascii_lowercase(), v.to_string()))
        .collect();
    Ok(Instance {
        inst_name: tok[0].clone(),
        subname,
        conn,
        params,
        line: line.no,
    })
}

/// Map a subcircuit-internal node token to a concrete node name.
fn resolve_node(
    tok: &str,
    prefix: &str,
    nmap: &mut HashMap<String, String>,
    globals: &std::collections::HashSet<String>,
) -> String {
    let key = tok.to_ascii_lowercase();
    if is_ground(&key) {
        return "0".to_string();
    }
    if globals.contains(&key) {
        return tok.to_string();
    }
    if let Some(v) = nmap.get(&key) {
        return v.clone();
    }
    let v = format!("{prefix}{tok}");
    nmap.insert(key, v.clone());
    v
}

/// Format an f64 as a re-lexable token (integer-valued numbers stay integers so
/// numeric node names inside `V()` round-trip).
fn fmt_num(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x:e}")
    }
}

/// Remap a behavioral value expression (a `B`-source RHS, or an `R`/`C`/`L` value
/// that depends on node voltages) for subckt flattening: node names inside
/// `V(...)` are port-/prefix-remapped, element names inside `I(...)` are
/// instance-prefixed, and every other bare identifier (a parameter) is folded to
/// its numeric value from `env`. Function names, numbers and operators pass
/// through. The result is a re-lexable string with no `{...}` wrapper.
fn remap_behavioral(
    s: &str,
    prefix: &str,
    nmap: &mut HashMap<String, String>,
    env: &HashMap<String, f64>,
    globals: &std::collections::HashSet<String>,
) -> String {
    let toks = match lex(s.trim()) {
        Ok(t) => t,
        Err(_) => return s.to_string(),
    };
    let mut out = String::new();
    let mut i = 0;
    while i < toks.len() {
        match &toks[i] {
            Tok::Ident(name) => {
                let lname = name.to_ascii_lowercase();
                let call = matches!(toks.get(i + 1), Some(Tok::LParen));
                if call && (lname == "v" || lname == "i") {
                    // V(a[,b]) / I(elem): remap the names inside the parens.
                    out.push_str(&lname);
                    out.push('(');
                    i += 2;
                    while i < toks.len() && !matches!(toks[i], Tok::RParen) {
                        match &toks[i] {
                            Tok::Comma => out.push(','),
                            Tok::Ident(n) => out.push_str(&if lname == "v" {
                                resolve_node(n, prefix, nmap, globals)
                            } else {
                                format!("{prefix}{n}")
                            }),
                            Tok::Num(x) => {
                                out.push_str(&resolve_node(&fmt_num(*x), prefix, nmap, globals))
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                    out.push(')');
                    i += 1; // past ')'
                } else if call {
                    out.push_str(&lname); // a function: keep, descend into its args
                    i += 1;
                } else if let Some(v) = env.get(&lname) {
                    out.push_str(&fmt_num(*v)); // a parameter -> its value
                    i += 1;
                } else {
                    out.push_str(name); // unbound name -> leave (resolved later)
                    i += 1;
                }
            }
            Tok::Num(x) => {
                out.push_str(&fmt_num(*x));
                i += 1;
            }
            Tok::Op(c) => {
                out.push(*c);
                i += 1;
            }
            Tok::LParen => {
                out.push('(');
                i += 1;
            }
            Tok::RParen => {
                out.push(')');
                i += 1;
            }
            Tok::Comma => {
                out.push(',');
                i += 1;
            }
            Tok::Cmp(s) => {
                out.push_str(s);
                i += 1;
            }
            Tok::Logic(s) => {
                out.push_str(s);
                i += 1;
            }
            Tok::Not => {
                out.push('!');
                i += 1;
            }
            Tok::Question => {
                out.push('?');
                i += 1;
            }
            Tok::Colon => {
                out.push(':');
                i += 1;
            }
        }
    }
    out
}

/// Whether a value string carries a node-voltage / branch-current reference, so
/// it must be lowered as a behavioral element rather than a constant.
fn is_behavioral_value(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    l.contains("v(") || l.contains("i(")
}

#[allow(clippy::too_many_arguments)]
fn expand(
    inst: &Instance,
    parent_prefix: &str,
    env_caller: &HashMap<String, f64>,
    subckts: &HashMap<String, Subckt>,
    out: &mut Vec<Line>,
    depth: u32,
    globals: &std::collections::HashSet<String>,
) -> Result<(), ParseError> {
    if depth > 64 {
        return Err(err(inst.line, "subcircuit nesting too deep (recursion?)"));
    }
    let sub = subckts
        .get(&inst.subname)
        .ok_or_else(|| err(inst.line, &format!("unknown subcircuit '{}'", inst.subname)))?;
    if inst.conn.len() != sub.ports.len() {
        return Err(err(
            inst.line,
            &format!(
                "instance '{}' has {} nodes, subckt '{}' expects {}",
                inst.inst_name,
                inst.conn.len(),
                inst.subname,
                sub.ports.len()
            ),
        ));
    }

    // Node map: ports -> caller nodes (internal nodes added lazily, prefixed).
    let new_prefix = format!("{}{}.", parent_prefix, inst.inst_name);
    let mut nmap: HashMap<String, String> = HashMap::default();
    for (p, c) in sub.ports.iter().zip(inst.conn.iter()) {
        nmap.insert(p.to_ascii_lowercase(), c.clone());
    }

    // Local env: caller env + subckt defaults + instance params (overriding).
    let mut env = env_caller.clone();
    for (k, vexpr) in &sub.defaults {
        if let Some(v) = resolve_value(vexpr, &env) {
            env.insert(k.clone(), v);
        }
    }
    for (k, vexpr) in &inst.params {
        if let Some(v) = resolve_value(vexpr, env_caller) {
            env.insert(k.clone(), v);
        }
    }

    for line in &sub.body {
        let head = &line.tokens[0];
        if head.eq_ignore_ascii_case(".param") {
            for t in &line.tokens[1..] {
                if let Some((k, v)) = t.split_once('=') {
                    if let Some(val) = resolve_value(v, &env) {
                        env.insert(k.to_ascii_lowercase(), val);
                    }
                }
            }
            continue;
        }
        if head.eq_ignore_ascii_case(".model") {
            out.push(line.clone()); // global model namespace
            continue;
        }
        if head.starts_with('X') || head.starts_with('x') {
            let nested = parse_instance(line, &mut nmap, &new_prefix, globals)?;
            // Instance name carries the path; recurse.
            let nested = Instance {
                inst_name: format!("{new_prefix}{}", nested.inst_name),
                ..nested
            };
            expand(&nested, "", &env, subckts, out, depth + 1, globals)?;
            continue;
        }

        let kind = head.chars().next().unwrap().to_ascii_uppercase();
        let mut newtoks = Vec::with_capacity(line.tokens.len());
        newtoks.push(format!("{new_prefix}{head}"));
        match element_layout(kind) {
            Some((nc, name_refs)) => {
                // An R/C/L whose value depends on a node voltage (a foundry
                // behavioral resistor) carries node references in its value
                // expression; remap and fold the whole expression once so they
                // bind to the right nodes (the value parser lowers it to a
                // behavioral element).
                let val_join: String = line.tokens[1..]
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx >= nc && !name_refs.contains(idx))
                    .map(|(_, t)| t.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                let behavioral = matches!(kind, 'R' | 'C' | 'L') && is_behavioral_value(&val_join);
                for (idx, t) in line.tokens[1..].iter().enumerate() {
                    if idx < nc {
                        newtoks.push(resolve_node(t, &new_prefix, &mut nmap, globals));
                    } else if name_refs.contains(&idx) {
                        // Element-name reference: prefix like its target.
                        newtoks.push(format!("{new_prefix}{t}"));
                    } else if !behavioral {
                        newtoks.push(subst_value(t, &env));
                    }
                }
                if behavioral {
                    // Take the value expression (keyword + balanced braces, drop
                    // any trailing `tc1=`/`tc2=` parameters), then remap.
                    let expr = value_expr(&val_join);
                    let remapped = remap_behavioral(expr, &new_prefix, &mut nmap, &env, globals);
                    newtoks.push(format!("{{{remapped}}}"));
                }
            }
            None if kind == 'B' => {
                // B name n+ n- V=expr | I=expr: remap the two terminal nodes,
                // then remap node / element references inside the value
                // expression so they bind through the subcircuit ports.
                newtoks.push(resolve_node(
                    &line.tokens[1],
                    &new_prefix,
                    &mut nmap,
                    globals,
                ));
                newtoks.push(resolve_node(
                    &line.tokens[2],
                    &new_prefix,
                    &mut nmap,
                    globals,
                ));
                let rest = line.tokens[3..].join(" ");
                let remapped = match rest.split_once('=') {
                    Some((lhs, rhs)) => format!(
                        "{}={}",
                        lhs.trim(),
                        remap_behavioral(rhs, &new_prefix, &mut nmap, &env, globals)
                    ),
                    None => remap_behavioral(&rest, &new_prefix, &mut nmap, &env, globals),
                };
                newtoks.push(remapped);
            }
            None if kind == 'N' => {
                // `N` (Verilog-A instance): the bare tokens are its nodes
                // followed by the module/model name; nodes remap through the
                // subcircuit ports, parameter values fold in the caller's
                // environment (so `w={w*fac}` binds the instance's params).
                let bare_end = line.tokens[1..]
                    .iter()
                    .position(|t| t.contains('='))
                    .map(|p| p + 1)
                    .unwrap_or(line.tokens.len());
                if bare_end < 3 {
                    return Err(err(line.no, "N instance needs nodes and a model name"));
                }
                for t in &line.tokens[1..bare_end - 1] {
                    newtoks.push(resolve_node(t, &new_prefix, &mut nmap, globals));
                }
                newtoks.push(line.tokens[bare_end - 1].clone());
                for t in &line.tokens[bare_end..] {
                    match t.split_once('=') {
                        Some((k, v)) => match resolve_value(v, &env) {
                            Some(x) => newtoks.push(format!("{k}={}", fmt_num(x))),
                            None => newtoks.push(t.clone()),
                        },
                        None => newtoks.push(t.clone()),
                    }
                }
            }
            None => {
                // Unknown element type: prefix the name, leave the rest as-is.
                newtoks.extend(line.tokens[1..].iter().cloned());
            }
        }
        out.push(Line {
            no: line.no,
            col: line.col,
            tokens: newtoks,
        });
    }
    Ok(())
}

/// Substitute a value token: evaluate an expression wrapped in `{...}` (SPICE)
/// or `'...'` (HSPICE inline) against `env`, replacing with a numeric literal,
/// both as a bare value and on the RHS of `key=expr`. Plain numbers, keywords
/// and model names are left untouched.
fn subst_value(tok: &str, env: &HashMap<String, f64>) -> String {
    if is_expr(tok) {
        if let Some(v) = resolve_value(tok, env) {
            return format!("{v}");
        }
        return tok.to_string();
    }
    if let Some((k, v)) = tok.split_once('=') {
        if is_expr(v) {
            if let Some(val) = resolve_value(v, env) {
                return format!("{k}={val}");
            }
        }
    }
    tok.to_string()
}

/// Whether a token is an expression wrapped in `{...}` or `'...'`.
fn is_expr(tok: &str) -> bool {
    (tok.starts_with('{') && tok.ends_with('}'))
        || (tok.starts_with('\'') && tok.ends_with('\'') && tok.len() >= 2)
}
