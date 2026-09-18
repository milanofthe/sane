//! Native Verilog-A frontend for SANE.
//!
//! Parses Verilog-A source and lowers the analog block onto SANE's symbolic
//! DAG via the `DeviceModel::lower_behavioral` path, so a Verilog-A model
//! becomes an ordinary device the whole engine can analyze. Verilog-A is also
//! the source of truth for the built-in device models (diode, MOSFET, BJT,
//! JFET, MESFET, switch, transformer, transmission line): they ship as `.va`
//! source in `builtin/` and lower through exactly this pipeline. Implemented
//! clean-room from the Accellera Verilog-AMS LRM 2.4.0; no GPL code is reused.
//!
//! Pipeline: `lexer` -> `preprocessor` (include/define/ifdef) -> `parser` ->
//! AST -> `elaborate` -> `lower` (with `template` instance caching).

// The Verilog-A front-end is an internal pipeline: lexer -> preprocessor ->
// parser -> AST -> elaborate -> lower. Those stages are crate-private so their
// token / AST / elaborated representations can change without breaking
// downstream crates; the curated entry points below are the public surface.
pub(crate) mod ast;
pub mod builtin;
pub mod device;
pub(crate) mod elaborate;
pub(crate) mod error;
pub(crate) mod lexer;
pub(crate) mod lower;
pub(crate) mod parser;
pub(crate) mod preprocessor;
mod template;
pub(crate) mod token;

// Curated public surface.
pub use ast::Module;
pub use builtin::{builtin_device, builtin_module};
pub use elaborate::{elaborate, ElaboratedModule, ResolvedParam};
pub use error::{Diagnostic, Diagnostics, Span};
pub use lower::lower_analog;

use std::path::PathBuf;

/// Preprocess and parse Verilog-A source into modules. `search_dirs` are
/// searched for `\`include` files (after the built-in standard headers).
pub fn parse_modules(
    src: &str,
    file: &str,
    search_dirs: &[PathBuf],
) -> Result<Vec<ast::Module>, error::Diagnostics> {
    let (toks, map) =
        preprocessor::preprocess_mapped(src, file, search_dirs, &["__VAMS_COMPACT_MODELING__"])?;
    // Parser diagnostics carry the preprocessor's source map, so an error
    // inside a macro body or an included file renders its true source line
    // plus the whole expansion chain.
    parser::parse(&toks, file).map_err(|d| d.with_map(map))
}

#[cfg(test)]
mod tests {
    use super::preprocessor::preprocess_mapped;
    use super::token::Tok;

    /// Render a preprocessed token stream to a compact string for assertions.
    fn pp(src: &str) -> String {
        let (toks, _) = preprocess_mapped(src, "test.va", &[], &[]).expect("preprocess ok");
        let mut out = String::new();
        for t in &toks {
            let s = match &t.tok {
                Tok::Ident(s) => s.clone(),
                Tok::Number(n) => format!("{n}"),
                Tok::LParen => "(".into(),
                Tok::RParen => ")".into(),
                Tok::Plus => "+".into(),
                Tok::Star => "*".into(),
                Tok::Comma => ",".into(),
                Tok::Semi => ";".into(),
                Tok::Assign => "=".into(),
                Tok::Minus => "-".into(),
                Tok::Slash => "/".into(),
                Tok::Eof => continue,
                _ => "?".into(),
            };
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&s);
        }
        out
    }

    #[test]
    fn object_macro_expands() {
        assert_eq!(pp("`define TWO 2\nx = `TWO ;"), "x = 2 ;");
    }

    #[test]
    fn function_macro_expands_with_args() {
        assert_eq!(
            pp("`define SQ(a) ((a)*(a))\ny = `SQ(3) ;"),
            "y = ( ( 3 ) * ( 3 ) ) ;"
        );
    }

    #[test]
    fn nested_macro_in_body_expands() {
        let s = pp("`define A 1\n`define B (`A + `A)\nz = `B ;");
        assert_eq!(s, "z = ( 1 + 1 ) ;");
    }

    #[test]
    fn ifdef_selects_branch() {
        assert_eq!(pp("`define FOO 1\n`ifdef FOO\na\n`else\nb\n`endif"), "a");
        assert_eq!(pp("`ifdef FOO\na\n`else\nb\n`endif"), "b");
        assert_eq!(pp("`ifndef FOO\na\n`else\nb\n`endif"), "a");
    }

    #[test]
    fn continued_define_body_folds_lines() {
        // A `\`-continued define body stays one logical line.
        assert_eq!(pp("`define S a + \\\n b\nx = `S ;"), "x = a + b ;");
    }

    #[test]
    fn builtin_constants_header_resolves() {
        // constants.vams is built-in; M_PI expands to its numeric value.
        let s = pp("`include \"constants.vams\"\nv = `M_PI ;");
        assert!(s.starts_with("v = 3.14159"), "got: {s}");
    }

    use super::ast::Stmt;
    use super::parse_modules;

    fn one_module(src: &str) -> super::ast::Module {
        let mut ms = parse_modules(src, "t.va", &[]).expect("parse ok");
        assert_eq!(ms.len(), 1, "expected one module");
        ms.pop().unwrap()
    }

    #[test]
    fn macro_expansion_chain_in_diagnostics() {
        // A parse error INSIDE a macro body must render the body's own line
        // plus a note pointing at the invocation site.
        let src =
            "`define BAD(x) I(a) <+ (x ;\nmodule m(a); electrical a;\nanalog `BAD(1.0)\nendmodule";
        let e = parse_modules(src, "t.va", &[]).unwrap_err();
        let text = e.to_string();
        assert!(
            text.contains("in expansion of `BAD"),
            "chain names the macro: {text}"
        );
        assert!(
            text.contains("t.va:3"),
            "chain points at the invocation line: {text}"
        );
        let rendered = e.render(src);
        assert!(
            rendered.contains("note: in expansion of `BAD"),
            "rendered note present: {rendered}"
        );
    }

    #[test]
    fn include_chain_in_diagnostics() {
        // A parse error inside an `include`d file must render the INCLUDE's
        // source line (not the root file's) plus an "included from" note.
        let dir = std::env::temp_dir().join("sane_va_inc_test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("broken.inc"), "parameter real P = ;\n").unwrap();
        let src =
            "module m(a); electrical a;\n`include \"broken.inc\"\nanalog I(a) <+ 1.0;\nendmodule";
        let e = parse_modules(src, "t.va", &[dir]).unwrap_err();
        let text = e.to_string();
        assert!(
            text.contains("broken.inc:1"),
            "error located in the include: {text}"
        );
        assert!(
            text.contains("included from t.va:2"),
            "include chain present: {text}"
        );
        let rendered = e.render(src);
        assert!(
            rendered.contains("parameter real P = ;"),
            "snippet shows the include's own source line: {rendered}"
        );
        assert!(
            rendered.contains("note: included from here"),
            "note present: {rendered}"
        );
    }

    #[test]
    fn parses_resistor_module() {
        let src = r#"
            `include "disciplines.vams"
            module res(p, n);
              inout p, n;
              electrical p, n;
              parameter real R = 1k from (0:inf);
              analog
                I(p, n) <+ V(p, n) / R;
            endmodule
        "#;
        let m = one_module(src);
        assert_eq!(m.name, "res");
        assert_eq!(m.ports, vec!["p", "n"]);
        assert_eq!(m.params.len(), 1);
        assert_eq!(m.params[0].name, "R");
        assert_eq!(m.analog.len(), 1);
        assert!(matches!(m.analog[0], Stmt::Contribution { .. }));
    }

    #[test]
    fn parses_diode_with_branch_ddt_if() {
        let src = r#"
            `include "disciplines.vams"
            `include "constants.vams"
            module dio(a, c);
              inout a, c;
              electrical a, c;
              parameter real Is = 1e-14 from (0:inf);
              parameter real Cj = 0.0;
              parameter real N  = 1.0 from (0:inf);
              branch (a,c) b;
              real id, vt;
              analog begin
                vt = $vt;
                id = Is * (limexp(V(b)/(N*vt)) - 1.0);
                if (V(b) > 0.0)
                  I(b) <+ id + ddt(Cj * V(b));
                else
                  I(b) <+ id;
              end
            endmodule
        "#;
        let m = one_module(src);
        assert_eq!(m.name, "dio");
        assert_eq!(m.params.len(), 3);
        assert_eq!(m.branches.len(), 1);
        assert_eq!(m.vars.len(), 2);
        // analog block flattened: one assign, one assign, one if
        assert!(m.analog.iter().any(|s| matches!(s, Stmt::If { .. })));
    }

    #[test]
    fn aliasparam_parses_and_elaborates() {
        let src = r#"
            module m(a);
              electrical a;
              parameter real R = 1k;
              aliasparam res = R;
              aliasparam mm = $mfactor;
              analog I(a) <+ V(a)/R;
            endmodule
        "#;
        let m = one_module(src);
        assert_eq!(m.aliases.len(), 2);
        assert_eq!(m.aliases[0].alias, "res");
        assert_eq!(m.aliases[0].target, "R");
        assert_eq!(m.aliases[1].target, "$mfactor");
        let em = super::elaborate::elaborate(&m).expect("elaborate");
        assert_eq!(em.aliases.get("res"), Some(&"R".to_string()));
        assert_eq!(em.aliases.get("mm"), Some(&"$mfactor".to_string()));
    }

    #[test]
    fn aliasparam_unknown_target_is_rejected() {
        let src = "module m(a); electrical a; parameter real R = 1k; \
                   aliasparam bad = nosuch; analog I(a) <+ V(a)/R; endmodule";
        let m = one_module(src);
        let e = super::elaborate::elaborate(&m).unwrap_err();
        assert!(
            e.to_string().contains("bad"),
            "diagnostic names the alias: {e}"
        );
    }

    #[test]
    fn analog_initial_block_parses_as_initial_step() {
        // `analog initial` is a one-time init block, same semantics as
        // `@(initial_step)` under SANE's single-model lowering.
        let src = r#"
            module m(a);
              electrical a;
              parameter real R = 1k;
              real g;
              analog initial begin
                g = 1.0/R;
              end
              analog I(a) <+ g*V(a);
            endmodule
        "#;
        let m = one_module(src);
        assert!(m.analog.iter().any(|s| matches!(s, Stmt::InitialStep(_))));
        let dev = VerilogADevice::new("X", elab(src));
        dev.validate().expect("analog initial lowers");
    }

    /// Lower a single-instance module and return its fragment.
    fn frag_of(ctx: &mut Graph, src: &str) -> sane_device::BehavioralFragment {
        let em = elab(src);
        let np = em.ports.len();
        let tv: Vec<rsdag::ExprId> = (1..=np).map(|k| ctx.sym(&format!("v{k}"))).collect();
        let tvd: Vec<rsdag::ExprId> = (1..=np).map(|k| ctx.sym(&format!("vdot{k}"))).collect();
        let mut lo = sane_device::Lowerer::new(ctx);
        let pv = std::collections::HashMap::default();
        let given = std::collections::HashSet::default();
        super::lower::lower_analog(&em, "X1", &pv, &given, 1.0, &mut lo, &tv, &tvd).expect("lower")
    }

    #[test]
    fn cross_event_declares_switching_surface() {
        // `@(cross(expr, dir))` and `@(above(expr))` register the surface with
        // its direction; identical surfaces collapse to one; the body may be
        // `$discontinuity`.
        let src = r#"
            module m(a, b);
              electrical a, b;
              parameter real Vt = 1.0;
              analog begin
                I(a, b) <+ (V(a, b) > Vt) ? V(a, b) : 1e-6*V(a, b);
                @(cross(V(a, b) - Vt, +1)) $discontinuity(0);
                @(cross(V(a, b) - Vt, +1));
                @(above(V(a, b) - 2*Vt));
                @(cross(V(a, b) - 3*Vt));
              end
            endmodule
        "#;
        let mut ctx = Graph::new();
        let frag = frag_of(&mut ctx, src);
        let dirs: Vec<i8> = frag.events.iter().map(|e| e.dir).collect();
        assert_eq!(dirs, vec![1, 1, 0], "one surface per distinct (expr, dir)");
        // the surface is a real expression over the terminal voltages
        let syms = ctx.free_symbols(frag.events[0].g);
        assert!(!syms.is_empty());
    }

    #[test]
    fn event_bodies_and_other_controls_are_rejected() {
        let with_body = r#"
            module m(a);
              electrical a;
              real s;
              analog begin
                @(cross(V(a) - 1.0, 0)) s = 1.0;
                I(a) <+ V(a);
              end
            endmodule
        "#;
        let e = VerilogADevice::new("X", elab(with_body))
            .validate()
            .unwrap_err();
        assert!(e.contains("body"), "{e}");
        let timer = r#"
            module m(a);
              electrical a;
              analog begin
                @(timer(0, 1e-6));
                I(a) <+ V(a);
              end
            endmodule
        "#;
        let e = VerilogADevice::new("X", elab(timer))
            .validate()
            .unwrap_err();
        assert!(e.contains("timer"), "{e}");
    }

    #[test]
    fn string_parameter_selects_model_variant() {
        // `parameter string mode` + `if (mode == "fast")`: the comparison folds
        // at lowering, selecting the variant statically (BSIMCMG/PSP pattern).
        let src = r#"
            module m(a,b);
              inout a,b; electrical a,b;
              parameter string mode = "fast";
              parameter real R = 1k;
              analog begin
                if (mode == "fast")
                  I(a,b) <+ 2.0 * V(a,b) / R;
                else
                  I(a,b) <+ V(a,b) / R;
              end
            endmodule
        "#;
        let em = elab(src);
        assert_eq!(em.string_params.get("mode"), Some(&"fast".to_string()));
        VerilogADevice::new("X", em)
            .validate()
            .expect("string variant lowers");

        // A bare string parameter outside ==/!= must error clearly.
        let bad = r#"
            module m(a,b); inout a,b; electrical a,b;
              parameter string mode = "x";
              analog I(a,b) <+ mode * V(a,b);
            endmodule
        "#;
        let e = VerilogADevice::new("X", elab(bad)).validate().unwrap_err();
        assert!(e.contains("string"), "clear diagnostic: {e}");
    }

    /// The OpenVAF changelog regression checklist: every case below was a real
    /// crash or misparse in another Verilog-A compiler. They must all parse
    /// (and, where applicable, lower) here.
    #[test]
    fn changelog_regression_checklist_parses() {
        let cases: &[(&str, &str)] = &[
            ("param without explicit type",
             "module m(a); electrical a; parameter P = 2.0; analog I(a) <+ P*V(a); endmodule"),
            ("ground declaration without discipline",
             "module m(a); electrical a; ground gnd; analog I(a,gnd) <+ V(a,gnd); endmodule"),
            ("ports without direction declaration",
             "module m(a,b); electrical a,b; parameter real R=1k; analog I(a,b) <+ V(a,b)/R; endmodule"),
            ("exclude with an expression bound",
             "module m(a); electrical a; parameter real X = 2.0; \
              parameter real P = 1.0 exclude X; analog I(a) <+ P*V(a); endmodule"),
            ("module with no branch contribution",
             "module m(a,b); electrical a,b; parameter real R=1k; real x; analog x = R; endmodule"),
            ("integer parameter with range",
             "module m(a); electrical a; parameter integer sel = 0 from [0:3]; \
              analog I(a) <+ sel*V(a); endmodule"),
        ];
        for (what, src) in cases {
            let ms = parse_modules(src, "t.va", &[]).unwrap_or_else(|d| panic!("{what}: {d}"));
            assert_eq!(ms.len(), 1, "{what}");
            let em = Arc::new(elaborate(&ms[0]).unwrap_or_else(|d| panic!("{what}: {d}")));
            VerilogADevice::new("X", em)
                .validate()
                .unwrap_or_else(|e| panic!("{what}: lower: {e}"));
        }
    }

    #[test]
    fn static_zero_potential_collapses_nodes() {
        // `if (sw==1) V(mid,b) <+ 0;` with sw=1 (default): mid collapses onto
        // port b -- no internal unknown, no source branch; the module IS a
        // resistor a-b. With sw=0 the else-arm makes it two series resistors.
        let src = r#"
            module m(a,b);
              inout a,b; electrical a,b,mid;
              parameter real R = 1k;
              parameter integer sw = 1;
              analog begin
                I(a,mid) <+ V(a,mid)/R;
                if (sw == 1)
                  V(mid,b) <+ 0;
                else
                  I(mid,b) <+ V(mid,b)/R;
              end
            endmodule
        "#;
        let em = elab(src);

        // Collapsed variant: identical DAE to a plain resistor element.
        let mut ctx = Graph::new();
        let mut cn = Circuit::new();
        cn.voltage_source("V1", 1, 0).resistor("R1", 1, 0);
        let native = assemble_dae(&mut ctx, &cn, &[]);
        let dev = VerilogADevice::new("R1", em.clone());
        let mut cv = Circuit::new();
        cv.voltage_source("V1", 1, 0);
        let devs = vec![DeviceInstance::new(Box::new(dev), vec![1, 0])];
        let va = assemble_dae(&mut ctx, &cv, &devs);
        assert_eq!(
            va.unknowns, native.unknowns,
            "collapse removes the internal node and the source branch"
        );
        let env = env_of(
            &mut ctx,
            &[
                ("V1", 2.0),
                ("R1", 1000.0),
                ("R1.R", 1000.0),
                ("R1.sw", 1.0),
                ("v1", 2.0),
                ("vdot1", 0.0),
                ("i_V1", -0.002),
                ("t", 0.0),
            ],
        );
        assert_dae_match(&ctx, &native, &va, &env);

        // Non-collapsed variant (sw=0): the internal node survives.
        let mut dev2 = VerilogADevice::new("R2", em);
        dev2.params.insert("sw".into(), 0.0);
        let mut cv2 = Circuit::new();
        cv2.voltage_source("V1", 1, 0);
        let devs2 = vec![DeviceInstance::new(Box::new(dev2), vec![1, 0])];
        let va2 = assemble_dae(&mut ctx, &cv2, &devs2);
        assert!(
            va2.unknowns.iter().any(|u| u.contains("mid")),
            "sw=0 keeps the internal node: {:?}",
            va2.unknowns
        );
    }

    #[test]
    fn collapse_chain_and_probe_block() {
        // A chain V(m1,m2)<+0, V(m2,b)<+0 collapses both internals onto the
        // port; a current-probed zero branch must NOT collapse (the probe needs
        // the branch current unknown).
        let chain = r#"
            module m(a,b);
              inout a,b; electrical a,b,m1,m2;
              parameter real R = 1k;
              analog begin
                I(a,m1) <+ V(a,m1)/R;
                V(m1,m2) <+ 0;
                V(m2,b) <+ 0;
              end
            endmodule
        "#;
        let mut ctx = Graph::new();
        let dev = VerilogADevice::new("X", elab(chain));
        let mut cv = Circuit::new();
        cv.voltage_source("V1", 1, 0);
        let va = assemble_dae(
            &mut ctx,
            &cv,
            &[DeviceInstance::new(Box::new(dev), vec![1, 0])],
        );
        assert_eq!(
            va.unknowns,
            vec!["v1", "i_V1"],
            "chain fully collapsed: {:?}",
            va.unknowns
        );

        let probed = r#"
            module m(a,b);
              inout a,b; electrical a,b,mid;
              parameter real R = 1k;
              real imon;
              analog begin
                I(a,mid) <+ V(a,mid)/R;
                V(mid,b) <+ 0;
                imon = I(mid,b);
                I(a,b) <+ 0.001 * imon;
              end
            endmodule
        "#;
        let mut ctx2 = Graph::new();
        let dev2 = VerilogADevice::new("X", elab(probed));
        let mut cv2 = Circuit::new();
        cv2.voltage_source("V1", 1, 0);
        let va2 = assemble_dae(
            &mut ctx2,
            &cv2,
            &[DeviceInstance::new(Box::new(dev2), vec![1, 0])],
        );
        assert!(
            va2.unknowns.iter().any(|u| u.contains("mid")),
            "probed zero branch keeps its unknown: {:?}",
            va2.unknowns
        );
    }

    #[test]
    fn dollar_limit_registers_terminal_limit() {
        use sane_device::{DeviceModel, LimitKind};
        // $limit over a port pair declares a Newton limit; the value is the
        // plain access (identity), so the converged point is unchanged.
        let src = r#"
            module dio(a,c);
              inout a,c; electrical a,c;
              parameter real Is = 1e-14;
              analog I(a,c) <+ Is * (limexp($limit(V(a,c), "pnjlim", 0.025, 0.7)/0.025) - 1.0);
            endmodule
        "#;
        let em = elab(src);
        let dev = VerilogADevice::new("D1", em);
        dev.validate().expect("$limit lowers as identity");
        // The lowered fragment records the limit on the (a, c) node symbols.
        let mut ctx = Graph::new();
        let va = ctx.sym("v1");
        let vc = ctx.sym("v2");
        let vad = ctx.sym("vdot1");
        let vcd = ctx.sym("vdot2");
        let sym = |ctx: &Graph, e| match ctx.node(e) {
            rsdag::Node::Symbol(s) => *s,
            _ => unreachable!(),
        };
        let (sa, sc) = (sym(&ctx, va), sym(&ctx, vc));
        let mut lo = sane_device::Lowerer::new(&mut ctx);
        let frag = dev.lower_behavioral(&mut lo, &[va, vc], &[vad, vcd], &[]);
        assert_eq!(frag.limits.len(), 1, "one recorded limit");
        assert_eq!(frag.limits[0].hi, Some(sa));
        assert_eq!(frag.limits[0].lo, Some(sc));
        assert_eq!(frag.limits[0].kind, LimitKind::PnJunction);
    }

    #[test]
    fn while_loop_bound_analyses_lower() {
        // Pattern 1: the HiSIM2 goto-emulation -- nested same-flag whiles with
        // a counter-capped re-trigger. Must lower via the shared budget.
        let flag_nest = r#"
            module m(a,b); inout a,b; electrical a,b;
              parameter real R = 1k;
              integer SOSL, NNN, MAXL; real x;
              analog begin
                MAXL = 5; NNN = 0; SOSL = 1; x = V(a,b);
                while (SOSL) begin
                  while (SOSL) begin
                    SOSL = 0;
                    x = x - 0.5*(x - V(a,b));
                    if ((x > 0.01) && (NNN < MAXL)) begin
                      NNN = NNN + 1;
                      SOSL = 1;
                    end
                  end
                end
                I(a,b) <+ x/R;
              end
            endmodule
        "#;
        VerilogADevice::new("X", elab(flag_nest))
            .validate()
            .expect("flag nest lowers");

        // Pattern 2: a runtime descent loop bounded by an enclosing guard fact
        // (the HiSIM2 exp-reduction idiom).
        let descent = r#"
            module m(a,b); inout a,b; electrical a,b;
              parameter real R = 1k;
              real T1, y, xx;
              analog begin
                xx = V(a,b);
                if (xx >= 500.0) y = 1.0e18;
                else begin
                  T1 = xx; y = 1.0;
                  while (T1 >= 60.0) begin
                    y = y * 1.0e6;
                    T1 = T1 - 60.0;
                  end
                  y = y * exp(T1);
                end
                I(a,b) <+ 1.0e-18 * y / R;
              end
            endmodule
        "#;
        VerilogADevice::new("X", elab(descent))
            .validate()
            .expect("descent loop lowers");

        // A genuinely unbounded loop must still be rejected with a clear error.
        let unbounded = r#"
            module m(a,b); inout a,b; electrical a,b;
              real t;
              analog begin
                t = V(a,b);
                while (t >= 1.0) t = t - V(a,b);
                I(a,b) <+ t;
              end
            endmodule
        "#;
        let e = VerilogADevice::new("X", elab(unbounded))
            .validate()
            .unwrap_err();
        assert!(
            e.contains("no static iteration bound"),
            "clear diagnostic: {e}"
        );
    }

    #[test]
    fn parses_case_for_and_function() {
        let src = r#"
            module m(a);
              electrical a;
              parameter integer sel = 0;
              analog function real twice;
                input x;
                real x;
                twice = 2.0 * x;
              endfunction
              analog begin : main
                integer k;
                real acc;
                acc = 0.0;
                for (k = 0; k < 3; k = k + 1) acc = acc + twice(k);
                case (sel)
                  0, 1: I(a) <+ acc;
                  default: I(a) <+ 0.0;
                endcase
              end
            endmodule
        "#;
        let m = one_module(src);
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].name, "twice");
        assert!(m.analog.iter().any(|s| matches!(s, Stmt::For { .. })));
        assert!(m.analog.iter().any(|s| matches!(s, Stmt::Case { .. })));
    }

    use super::ast::Module;
    use super::device::VerilogADevice;
    use super::elaborate::elaborate;
    use num_complex::Complex64;
    use rsdag::{eval, Graph, Node, SymbolId};
    use sane_dae::{assemble_dae, Dae, DeviceInstance};
    use sane_mna::Circuit;
    use std::sync::Arc;

    fn elab(src: &str) -> Arc<super::elaborate::ElaboratedModule> {
        let ms: Vec<Module> = parse_modules(src, "x.va", &[]).expect("parse");
        Arc::new(elaborate(&ms[0]).expect("elaborate"))
    }

    fn env_of(
        ctx: &mut Graph,
        vals: &[(&str, f64)],
    ) -> std::collections::HashMap<SymbolId, Complex64> {
        // Native devices read the global `$temp` and per-instance Tnom/Eg/XTI;
        // default them to nominal so temperature scalings are identities.
        let tnom = sane_core::constants::TEMP_NOMINAL_K;
        let mut all: Vec<(String, f64)> =
            vec![(sane_core::constants::TEMP_SYMBOL.to_string(), tnom)];
        let mut insts: std::collections::HashSet<_> = std::collections::HashSet::default();
        for (name, _) in vals {
            if let Some((inst, _)) = name.split_once('.') {
                if insts.insert(inst.to_string()) {
                    all.push((format!("{inst}.Tnom"), tnom));
                    all.push((format!("{inst}.Eg"), 1.11));
                    all.push((format!("{inst}.XTI"), 3.0));
                }
            }
        }
        all.extend(vals.iter().map(|(n, x)| (n.to_string(), *x)));
        let mut env = std::collections::HashMap::default();
        for (name, x) in &all {
            let e = ctx.sym(name);
            if let Node::Symbol(s) = ctx.node(e) {
                env.insert(*s, Complex64::new(*x, 0.0));
            }
        }
        env
    }

    /// Assert two DAEs have the same unknown layout and identical residual values
    /// at the given environment point.
    fn assert_dae_match(
        ctx: &Graph,
        a: &Dae,
        b: &Dae,
        env: &std::collections::HashMap<SymbolId, Complex64>,
    ) {
        assert_eq!(a.unknowns, b.unknowns, "unknown layout differs");
        for (i, (ra, rb)) in a.residuals.iter().zip(&b.residuals).enumerate() {
            let d = (eval(ctx, &[*ra], env)[0] - eval(ctx, &[*rb], env)[0]).norm();
            assert!(d < 1e-9, "residual {i} differs by {d}");
        }
    }

    #[test]
    fn va_resistor_matches_native_element() {
        let mut ctx = Graph::new();
        let mut cn = Circuit::new();
        cn.voltage_source("V1", 1, 0).resistor("R1", 1, 0);
        let native = assemble_dae(&mut ctx, &cn, &[]);

        let src = "module res(p,n); inout p,n; electrical p,n; \
                   parameter real R = 1000.0; analog I(p,n) <+ V(p,n)/R; endmodule";
        let dev = VerilogADevice::new("R1", elab(src));
        let mut cv = Circuit::new();
        cv.voltage_source("V1", 1, 0);
        let devs = vec![DeviceInstance::new(Box::new(dev), vec![1, 0])];
        let va = assemble_dae(&mut ctx, &cv, &devs);

        let env = env_of(
            &mut ctx,
            &[
                ("V1", 2.0),
                ("R1", 1000.0),
                ("R1.R", 1000.0),
                ("v1", 2.0),
                ("vdot1", 0.0),
                ("i_V1", -0.002),
                ("t", 0.0),
            ],
        );
        assert_dae_match(&ctx, &native, &va, &env);
    }

    #[test]
    fn va_diode_matches_native_diode() {
        let mut ctx = Graph::new();
        // native diode D1 across V1, with a series resistor for a real node.
        let mut cn = Circuit::new();
        cn.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
        let nd = vec![DeviceInstance::new(
            Box::new(crate::builtin_device("sane_diode", "D1", &[])),
            vec![2, 0],
        )];
        let native = assemble_dae(&mut ctx, &cn, &nd);

        // The VA diode uses the built-in thermal voltage `$vt = k*T/q`, exactly
        // as the native diode now does, so the two agree (rather than hard-coding
        // a slightly different Vt constant).
        let src = "module diode(a,c); inout a,c; electrical a,c; \
                   parameter real Is = 1e-14; parameter real N = 1.0; \
                   analog I(a,c) <+ Is * (limexp(V(a,c)/(N*$vt)) - 1.0); endmodule";
        let dev = VerilogADevice::new("D1", elab(src));
        let mut cv = Circuit::new();
        cv.voltage_source("V1", 1, 0).resistor("R1", 1, 2);
        let vd = vec![DeviceInstance::new(Box::new(dev), vec![2, 0])];
        let va = assemble_dae(&mut ctx, &cv, &vd);

        let env = env_of(
            &mut ctx,
            &[
                ("V1", 0.7),
                ("R1", 1000.0),
                ("D1.Is", 1e-14),
                ("D1.N", 1.0),
                ("v1", 0.7),
                ("v2", 0.55),
                ("vdot1", 0.0),
                ("vdot2", 0.0),
                ("i_V1", -1.5e-4),
                ("t", 0.0),
            ],
        );
        assert_dae_match(&ctx, &native, &va, &env);
    }

    fn resistor_env(ctx: &mut Graph) -> std::collections::HashMap<SymbolId, Complex64> {
        env_of(
            ctx,
            &[
                ("V1", 2.0),
                ("R1", 1000.0),
                ("R1.R", 1000.0),
                ("v1", 2.0),
                ("vdot1", 0.0),
                ("i_V1", -0.002),
                ("t", 0.0),
            ],
        )
    }

    #[test]
    fn va_function_and_if_resistor_matches_native() {
        let mut ctx = Graph::new();
        let mut cn = Circuit::new();
        cn.voltage_source("V1", 1, 0).resistor("R1", 1, 0);
        let native = assemble_dae(&mut ctx, &cn, &[]);
        let src = "module res(p,n); inout p,n; electrical p,n; parameter real R = 1000.0; \
                   analog function real recip; input x; real x; recip = 1.0/x; endfunction \
                   analog begin real g; g = recip(R); \
                   if (V(p,n) >= 0.0) I(p,n) <+ g*V(p,n); else I(p,n) <+ g*V(p,n); end \
                   endmodule";
        let dev = VerilogADevice::new("R1", elab(src));
        let mut cv = Circuit::new();
        cv.voltage_source("V1", 1, 0);
        let devs = vec![DeviceInstance::new(Box::new(dev), vec![1, 0])];
        let va = assemble_dae(&mut ctx, &cv, &devs);
        let env = resistor_env(&mut ctx);
        assert_dae_match(&ctx, &native, &va, &env);
    }

    #[test]
    fn va_for_loop_resistor_matches_native() {
        let mut ctx = Graph::new();
        let mut cn = Circuit::new();
        cn.voltage_source("V1", 1, 0).resistor("R1", 1, 0);
        let native = assemble_dae(&mut ctx, &cn, &[]);
        let src = "module res(p,n); inout p,n; electrical p,n; parameter real R = 1000.0; \
                   analog begin real g; integer k; g = 0.0; \
                   for (k = 0; k < 4; k = k + 1) g = g + 1.0/(4.0*R); \
                   I(p,n) <+ g*V(p,n); end endmodule";
        let dev = VerilogADevice::new("R1", elab(src));
        let mut cv = Circuit::new();
        cv.voltage_source("V1", 1, 0);
        let devs = vec![DeviceInstance::new(Box::new(dev), vec![1, 0])];
        let va = assemble_dae(&mut ctx, &cv, &devs);
        let env = resistor_env(&mut ctx);
        assert_dae_match(&ctx, &native, &va, &env);
    }

    #[test]
    fn va_extended_math_and_simparam_lower() {
        let src = "module m(a,b); inout a,b; electrical a,b; \
                   analog I(a,b) <+ (asin(0.5) + acos(0.5) + atan2(1.0,2.0) + acosh(2.0) \
                   + atanh(0.25) + log2(2.0) + $simparam(\"gmin\")) * V(a,b); endmodule";
        let dev = VerilogADevice::new("X1", elab(src));
        dev.validate().expect("extended math + $simparam lower");
    }

    #[test]
    fn va_b3_constructs_lower() {
        // analog function with an output argument (call statement, inlined).
        let f = "module m(a,b); inout a,b; electrical a,b; parameter real R=1k; \
                 analog function real dbl; input x; output y; real x; begin y = 2.0*x; dbl = x; end endfunction \
                 analog begin real h; h = 0.0; dbl(V(a,b), h); I(a,b) <+ h/R; end endmodule";
        VerilogADevice::new("X", elab(f))
            .validate()
            .expect("output-arg function");
        // idtmod
        let g = "module m(a,b); inout a,b; electrical a,b; \
                 analog V(a,b) <+ idtmod(V(a,b), 0.0, 1.0); endmodule";
        VerilogADevice::new("X", elab(g))
            .validate()
            .expect("idtmod");
        // indirect contribution (implicit equation)
        let h = "module m(a,b); inout a,b; electrical a,b; \
                 analog V(a,b) : V(a,b) == 1.0; endmodule";
        VerilogADevice::new("X", elab(h))
            .validate()
            .expect("indirect");
        // tabular noise (parse + lower; large-signal zero)
        let nt = "module m(a,b); inout a,b; electrical a,b; parameter real R=1k; \
                  analog begin I(a,b) <+ V(a,b)/R; I(a,b) <+ noise_table({1.0, 1e-18, 1e9, 1e-18}); end endmodule";
        VerilogADevice::new("X", elab(nt))
            .validate()
            .expect("noise_table");
    }

    #[test]
    fn empty_arg_builtins_error_instead_of_panicking() {
        // `ddt()`, `idt()`, `white_noise()`, `noise_table()` with no argument used
        // to index `args[0]` and panic; they must return a clean error. (The test
        // completing at all proves there is no panic.)
        let bodies = [
            "I(a,b) <+ ddt();",
            "V(a,b) <+ idt();",
            "I(a,b) <+ white_noise();",
            "I(a,b) <+ noise_table();",
        ];
        for body in bodies {
            let src = format!(
                "module m(a,b); inout a,b; electrical a,b; analog begin {body} end endmodule"
            );
            let r = VerilogADevice::new("X", elab(&src)).validate();
            assert!(r.is_err(), "expected an error for `{body}`, got Ok");
        }
    }

    #[test]
    fn const_fold_and_lowering_math_tables_agree() {
        // Every math function the compile-time folder (`const_builtin`) accepts
        // must also be lowerable at runtime (`builtin`); otherwise a constant
        // argument folds while a voltage-dependent one fails. Exercise the folder
        // set via a parameter default (const args) AND the lowering set via a
        // contribution (runtime arg `V(a,b)`); both must succeed.
        let src = "module m(a,b); inout a,b; electrical a,b; \
            parameter real p = exp(0.1)+ln(2.0)+log(2.0)+sqrt(2.0)+abs(-1.0) \
              +sin(0.1)+cos(0.1)+tan(0.1)+atan(0.1)+tanh(0.1)+sinh(0.1)+cosh(0.1) \
              +floor(1.5)+ceil(1.5)+pow(2.0,3.0)+min(1.0,2.0)+max(1.0,2.0); \
            analog I(a,b) <+ p * ( exp(V(a,b))+ln(V(a,b))+log(V(a,b))+sqrt(V(a,b)) \
              +abs(V(a,b))+sin(V(a,b))+cos(V(a,b))+tan(V(a,b))+atan(V(a,b)) \
              +tanh(V(a,b))+sinh(V(a,b))+cosh(V(a,b))+floor(V(a,b))+ceil(V(a,b)) \
              +pow(V(a,b),2.0)+min(V(a,b),1.0)+max(V(a,b),1.0) ); endmodule";
        VerilogADevice::new("X", elab(src))
            .validate()
            .expect("const-folder and lowering math tables must agree");
    }

    #[test]
    fn analysis_system_function_folds_to_zero() {
        // `analysis("noise")` folds to 0 (SANE's analysis-agnostic model), so a
        // `doNoise`-gated block resolves statically and the model lowers -- the
        // BSIM3 pattern. The string argument must not be lowered as a value.
        let src = "module m(a,b); inout a,b; electrical a,b; \
                   parameter real g = 1e-3; integer doNoise; \
                   analog begin doNoise = analysis(\"noise\"); \
                   I(a,b) <+ g*V(a,b); \
                   if (doNoise) I(a,b) <+ white_noise(1e-12, \"n\"); end endmodule";
        VerilogADevice::new("X", elab(src))
            .validate()
            .expect("analysis() folds and the model lowers");
    }

    #[test]
    fn variable_read_before_assignment_defaults_to_zero() {
        // `flag` is read in an `if` condition before its only (guarded)
        // assignment. Verilog-A defaults an unassigned variable to 0, so this
        // must lower (folding the `if` to its else arm) rather than raising
        // "unknown identifier" -- the BSIM3 / MVSG compact-model pattern.
        let src = "module m(a,b); inout a,b; electrical a,b; \
                   parameter real sw = 0.0; \
                   integer flag; \
                   analog begin \
                   if (flag) I(a,b) <+ 2.0*V(a,b); else I(a,b) <+ V(a,b); \
                   if (sw > 0.5) flag = 1; end endmodule";
        VerilogADevice::new("X", elab(src))
            .validate()
            .expect("read-before-assign defaults to 0");
    }

    #[test]
    fn parameter_ranges_are_enforced() {
        // `from (0:inf)` and `exclude` must actually reject out-of-range values
        // (the hardening for compact models, which declare these precisely).
        let src = r#"
            module m(a, c);
              inout a, c; electrical a, c;
              parameter real Is = 1e-14 from (0:inf);
              parameter real N = 1.0 from [1:2];
              parameter real sel = 1.0 exclude 0;
              analog I(a,c) <+ Is*V(a,c)/N + sel*0.0;
            endmodule"#;
        let m = elab(src);
        let mut v = std::collections::HashMap::default();
        // All in range -> no violations.
        v.insert("Is".to_string(), 1e-12);
        v.insert("N".to_string(), 1.5);
        v.insert("sel".to_string(), 3.0);
        assert!(
            m.check_param_ranges(&v).is_empty(),
            "valid params should pass"
        );
        // Is <= 0 violates from(0:inf); N = 3 violates [1:2]; sel = 0 is excluded.
        v.insert("Is".to_string(), -1.0);
        v.insert("N".to_string(), 3.0);
        v.insert("sel".to_string(), 0.0);
        let viol = m.check_param_ranges(&v);
        assert_eq!(viol.len(), 3, "expected 3 range violations, got: {viol:?}");
        assert!(viol.iter().any(|s| s.contains("'Is'")));
        assert!(viol.iter().any(|s| s.contains("'N'")));
        assert!(viol.iter().any(|s| s.contains("'sel'")));
    }

    #[test]
    fn elaborates_diode_nodes_params_branches() {
        let src = r#"
            `include "disciplines.vams"
            `include "constants.vams"
            module dio(a, c);
              inout a, c;
              electrical a, c, mid;
              parameter real Is = 1e-14 from (0:inf);
              parameter real area = 2.0;
              parameter real Isarea = Is * area;
              branch (a,mid) bd;
              real id;
              analog I(bd) <+ Is * area * (limexp(V(bd)/0.025) - 1.0);
            endmodule
        "#;
        let m = one_module(src);
        let em = elaborate(&m).expect("elaborate ok");
        // nodes = ports a,c plus internal mid; internal = [mid]
        assert_eq!(em.nodes, vec!["a", "c", "mid"]);
        assert_eq!(em.internal_nodes, vec!["mid"]);
        // param defaults const-folded, incl. one referencing earlier params
        assert_eq!(em.params.len(), 3);
        assert_eq!(em.params[1].default, 2.0);
        assert!(
            (em.params[2].default - 2e-14).abs() < 1e-25,
            "Isarea={}",
            em.params[2].default
        );
        // branch resolved
        assert_eq!(
            em.branches.get("bd"),
            Some(&("a".to_string(), "mid".to_string()))
        );
        assert_eq!(em.vars.len(), 1);
    }

    #[test]
    fn const_eval_folds_builtins() {
        use super::elaborate::const_eval;
        use rustc_hash::FxHashMap as HashMap;
        let env = HashMap::default();
        let src = "module m(a); electrical a; parameter real p = sqrt(4.0) + pow(2.0,3.0); analog I(a)<+0.0; endmodule";
        let m = one_module(src);
        // p default = 2 + 8 = 10
        assert_eq!(m.params[0].name, "p");
        assert_eq!(const_eval(&m.params[0].default, &env), Some(10.0));
    }

    // Corpus lowering probe: parse+elaborate+lower real ECL-2.0 models directly
    // (bypassing assemble_dae), reporting which construct, if any, is unhandled.
    // Ignored (TEMP path); run with `--ignored`.
    #[test]
    #[ignore]
    fn lowers_corpus_models() {
        use super::lower::lower_analog;
        use rsdag::ExprId;
        use sane_device::Lowerer;
        let base = match corpus_base() {
            Some(b) => b,
            None => {
                println!("corpus: SKIP (set SANE_VA_CORPUS to a compact-model checkout)");
                return;
            }
        };
        // Layout: OpenVAF `integration_tests` (one dir per model). Model-local
        // `.include`/`.inc` resolve via the file's own dir; standard headers
        // (constants.vams / discipline.h) are built in.
        let cases = [
            ("DIODE/diode.va", "diode"),
            ("DIODE_CMC/diode_cmc.va", "diode_cmc"),
            ("RESISTOR/resistor.va", "resistor"),
            ("EKV/ekv.va", "ekv"),
            ("BSIM3/bsim3.va", "bsim3"),
            ("BSIM4/bsim4.va", "bsim4"),
            ("BSIM6/bsim6.va", "bsim6"),
            ("BSIMBULK/bsimbulk.va", "bsimbulk"),
            ("BSIMCMG/bsimcmg.va", "bsimcmg"),
            ("BSIMIMG/bsimimg.va", "bsimimg"),
            ("BSIMSOI/bsimsoi.va", "bsimsoi"),
            ("MEXTRAM/mextram.va", "mextram"),
            ("HICUML2/hicuml2.va", "hicuml2"),
            ("HiSIM2/hisim2.va", "hisim2"),
            ("HiSIMHV/hisimhv.va", "hisimhv"),
            ("HiSIMSOTB/hisimsotb.va", "hisimsotb"),
            ("MVSG_CMC/mvsg_cmc.va", "mvsg_cmc"),
            ("PSP102/psp102.va", "psp102"),
            ("PSP103/psp103.va", "psp103"),
            ("ASMHEMT/asmhemt.va", "asmhemt"),
        ];
        for (rel, label) in cases {
            let path = base.join(rel);
            let src = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => {
                    println!("{label}: SKIP (not present)");
                    continue;
                }
            };
            let dir = path.parent().unwrap().to_path_buf();
            let ms = match parse_modules(&src, rel, &[dir]) {
                Ok(m) => m,
                Err(d) => {
                    println!("{label}: PARSE FAIL: {}", first_line(&d.to_string()));
                    continue;
                }
            };
            let em = match elaborate(&ms[0]) {
                Ok(e) => std::sync::Arc::new(e),
                Err(d) => {
                    println!("{label}: ELAB FAIL: {}", first_line(&d.to_string()));
                    continue;
                }
            };
            let np = em.ports.len();
            let mut ctx = Graph::new();
            let term_v: Vec<ExprId> = (1..=np).map(|k| ctx.sym(&format!("v{k}"))).collect();
            let term_vdot: Vec<ExprId> = (1..=np).map(|k| ctx.sym(&format!("vdot{k}"))).collect();
            let mut lo = Lowerer::new(&mut ctx);
            let pv = std::collections::HashMap::default();
            let given = std::collections::HashSet::default();
            match lower_analog(&em, "X1", &pv, &given, 1.0, &mut lo, &term_v, &term_vdot) {
                Ok(frag) => println!(
                    "{label}: LOWERED ok ({} terminals, {} residuals, {} extras)",
                    frag.terminal_currents.len(),
                    frag.residuals.len(),
                    lo.extras.len()
                ),
                Err(e) => println!("{label}: LOWER GAP: {e}"),
            }
        }
    }

    fn first_line(s: &str) -> String {
        s.lines().next().unwrap_or("").to_string()
    }

    // Diagnostic: report compact-model parameters whose folded default is
    // non-finite (NaN/Inf). A non-finite default is silently baked into the
    // residual as a bias-independent constant, poisoning every Newton step
    // (PSP102/PSP103 fail to converge for exactly this reason). Run with
    // `--ignored` and SANE_VA_CORPUS set.
    #[test]
    #[ignore]
    fn nonfinite_param_defaults() {
        use super::elaborate::elaborate;
        let base = match corpus_base() {
            Some(b) => b,
            None => {
                println!("corpus: SKIP (set SANE_VA_CORPUS)");
                return;
            }
        };
        for (rel, label) in [
            ("PSP103/psp103.va", "psp103"),
            ("PSP102/psp102.va", "psp102"),
        ] {
            let path = base.join(rel);
            let src = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => {
                    println!("{label}: SKIP (not present)");
                    continue;
                }
            };
            let dir = path.parent().unwrap().to_path_buf();
            let ms = parse_modules(&src, rel, &[dir]).expect("parse");
            let em = elaborate(&ms[0]).expect("elaborate");
            let bad: Vec<_> = em
                .params
                .iter()
                .filter(|p| !p.default.is_finite())
                .map(|p| format!("{}={}", p.name, p.default))
                .collect();
            println!(
                "{label}: {} params, {} non-finite default(s): {:?}",
                em.params.len(),
                bad.len(),
                bad
            );
        }
    }

    // Trace the origin of the bias-independent NaN that stops PSP102/PSP103 from
    // converging: lower the model with `SANE_VA_TRACE_NAN=1` set and watch for
    // the first `VA non-finite const: <var> = NaN` warning. Run with `--ignored`,
    // SANE_VA_CORPUS and SANE_VA_TRACE_NAN set.
    #[test]
    #[ignore]
    fn trace_nan_lowering() {
        use super::elaborate::elaborate;
        use super::lower::lower_analog;
        use rsdag::ExprId;
        use sane_device::Lowerer;
        sane_core::log::set_level(sane_core::LogLevel::Warning);
        let base = match corpus_base() {
            Some(b) => b,
            None => {
                println!("corpus: SKIP (set SANE_VA_CORPUS)");
                return;
            }
        };
        for (rel, label) in [("PSP103/psp103.va", "psp103")] {
            let path = base.join(rel);
            let src = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => {
                    println!("{label}: SKIP (not present)");
                    continue;
                }
            };
            let dir = path.parent().unwrap().to_path_buf();
            let ms = parse_modules(&src, rel, &[dir]).expect("parse");
            let em = std::sync::Arc::new(elaborate(&ms[0]).expect("elaborate"));
            let np = em.ports.len();
            let mut ctx = Graph::new();
            let tv: Vec<ExprId> = (1..=np).map(|k| ctx.sym(&format!("v{k}"))).collect();
            let tvd: Vec<ExprId> = (1..=np).map(|k| ctx.sym(&format!("vdot{k}"))).collect();
            let mut lo = Lowerer::new(&mut ctx);
            let pv = std::collections::HashMap::default();
            let given = std::collections::HashSet::default();
            let _ = lower_analog(&em, "X1", &pv, &given, 1.0, &mut lo, &tv, &tvd);
            println!("{label}: lowered (see VA non-finite const warnings above)");
        }
    }

    // Pinpoint the first non-finite node in a lowered compact model's residual
    // DAG: bind every symbol (params -> card/default, unknowns -> a test bias),
    // evaluate the whole DAG, and report the lowest-index NaN/Inf node with its
    // operands. Nodes are interned bottom-up, so that node is the NaN *origin*
    // (its operands have smaller indices and are finite). Run with `--ignored`
    // and SANE_VA_CORPUS set.
    #[test]
    #[ignore]
    fn find_nan_origin() {
        use super::elaborate::elaborate;
        use super::lower::lower_analog;
        use rsdag::ExprId;
        /// Every node of the graph over `env`, by id.
        fn eval_all(ctx: &Graph, env: &std::collections::HashMap<SymbolId, f64>) -> Vec<f64> {
            let all: Vec<ExprId> = (0..ctx.len() as u32).map(ExprId).collect();
            rsdag::eval(ctx, &all, env)
        }
        use sane_device::Lowerer;
        let base = match corpus_base() {
            Some(b) => b,
            None => {
                println!("corpus: SKIP (set SANE_VA_CORPUS)");
                return;
            }
        };
        let (rel, label) = ("PSP103/psp103.va", "psp103");
        let path = base.join(rel);
        let src = std::fs::read_to_string(&path).expect("read");
        let dir = path.parent().unwrap().to_path_buf();
        let ms = parse_modules(&src, rel, &[dir]).expect("parse");
        let em = std::sync::Arc::new(elaborate(&ms[0]).expect("elaborate"));
        let pnames: std::collections::HashSet<&str> =
            em.params.iter().map(|p| p.name.as_str()).collect();
        let pdefault: std::collections::HashMap<&str, f64> = em
            .params
            .iter()
            .map(|p| (p.name.as_str(), p.default))
            .collect();
        let np = em.ports.len();
        let mut ctx = Graph::new();
        let tv: Vec<ExprId> = (1..=np).map(|k| ctx.sym(&format!("v{k}"))).collect();
        let tvd: Vec<ExprId> = (1..=np).map(|k| ctx.sym(&format!("vdot{k}"))).collect();
        let mut lo = Lowerer::new(&mut ctx);
        let pv = std::collections::HashMap::default();
        let given = std::collections::HashSet::default();
        let frag = lower_analog(&em, "X1", &pv, &given, 1.0, &mut lo, &tv, &tvd).expect("lower");
        let resids = frag.residuals.clone();
        drop(lo);

        // Bind every symbol: param -> default, vdot -> 0, everything else -> 0.3.
        let mut env = std::collections::HashMap::default();
        for i in 0..ctx.len() {
            if let Node::Symbol(s) = ctx.node(ExprId(i as u32)) {
                let s = *s;
                let name = ctx.symbol_name(s).to_string();
                let suffix = name.rsplit('.').next().unwrap_or(&name);
                let v = if name == sane_core::constants::TEMP_SYMBOL {
                    sane_core::constants::TEMP_NOMINAL_K
                } else if pnames.contains(suffix) {
                    pdefault[suffix]
                } else if name.starts_with("vdot") {
                    0.0
                } else {
                    0.0
                };
                env.insert(s, v);
            }
        }

        let w = eval_all(&ctx, &env);
        // First non-finite node overall = the NaN origin.
        let origin = (0..ctx.len()).find(|&i| !w[i].is_finite());
        match origin {
            None => println!("{label}: no non-finite node (residuals finite)"),
            Some(i) => {
                let kids = |id: ExprId| -> Vec<ExprId> {
                    match ctx.node(id) {
                        Node::Add(a, b) | Node::Mul(a, b) => vec![*a, *b],
                        Node::Neg(a) | Node::Pow(a, _) | Node::Unary(_, a) => vec![*a],
                        Node::Select(c, t, e) => vec![*c, *t, *e],
                        _ => vec![],
                    }
                };
                // Recursively render the origin's expression tree (to `depth`),
                // annotating each node with its evaluated value and symbol names,
                // so the offending physical quantity is identifiable in source.
                fn render(
                    ctx: &Graph,
                    w: &[f64],
                    kids: &dyn Fn(ExprId) -> Vec<ExprId>,
                    id: ExprId,
                    depth: usize,
                    pad: usize,
                    out: &mut String,
                ) {
                    let v = w[id.0 as usize];
                    let label = match ctx.node(id) {
                        Node::Symbol(s) => ctx.symbol_name(*s).to_string(),
                        Node::Const(_) => "const".to_string(),
                        Node::Pow(_, k) => format!("pow(_,{k})"),
                        Node::Unary(op, _) => format!("{op:?}"),
                        Node::Mul(..) => "Mul".to_string(),
                        Node::Add(..) => "Add".to_string(),
                        Node::Neg(..) => "Neg".to_string(),
                        Node::Select(..) => "Select".to_string(),
                        other => format!("{other:?}").chars().take(16).collect(),
                    };
                    out.push_str(&format!("{:pad$}#{} {label} = {v}\n", "", id.0, pad = pad));
                    if depth > 0 {
                        for k in kids(id) {
                            render(ctx, w, kids, k, depth - 1, pad + 2, out);
                        }
                    }
                }
                let mut tree = String::new();
                render(&ctx, &w, &kids, ExprId(i as u32), 4, 0, &mut tree);
                println!("{label}: NAN ORIGIN at node #{i}:\n{tree}");
                // How many residuals does this poison?
                let nbad = eval_all(&ctx, &env);
                let poisoned = resids
                    .iter()
                    .filter(|r| !nbad[r.0 as usize].is_finite())
                    .count();
                println!("{label}: {poisoned}/{} residuals non-finite", resids.len());
            }
        }
    }

    /// Root of the Verilog-A validation corpus. Path-agnostic and CI-neutral:
    /// read from `SANE_VA_CORPUS` (point it at a checkout of real compact models,
    /// e.g. OpenVAF's `integration_tests`). `None` -> the corpus tests skip.
    fn corpus_base() -> Option<std::path::PathBuf> {
        std::env::var_os("SANE_VA_CORPUS").map(std::path::PathBuf::from)
    }

    // Corpus parse smoke test (the WP1 gate): real ECL-2.0 models from the
    // reference checkout in TEMP. Ignored by default (environment-specific path,
    // and we test locally not in CI); run with `--ignored`.
    #[test]
    #[ignore]
    fn parses_corpus_models() {
        let base = match corpus_base() {
            Some(b) => b,
            None => {
                println!("corpus: SKIP (set SANE_VA_CORPUS to a compact-model checkout)");
                return;
            }
        };
        let cases = [
            ("EKV/ekv.va", "ekv"),
            ("PSP103/psp103.va", "psp103"),
            ("ASMHEMT/asmhemt.va", "asmhemt"),
        ];
        for (rel, label) in cases {
            let path = base.join(rel);
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|_| panic!("read {}", path.display()));
            let dir = path.parent().unwrap().to_path_buf();
            match parse_modules(&src, rel, &[dir]) {
                Ok(ms) => {
                    assert!(!ms.is_empty(), "{label}: no modules");
                    let em =
                        elaborate(&ms[0]).unwrap_or_else(|d| panic!("{label} elaborate:\n{d}"));
                    println!(
                        "{label}: {} module(s); first: {} nodes ({} internal), {} params, {} branches, {} fns, {} opvars",
                        ms.len(), em.nodes.len(), em.internal_nodes.len(),
                        em.params.len(), em.branches.len(), em.functions.len(), em.opvars.len()
                    );
                }
                Err(d) => panic!("{label} parse failed:\n{}", d.render(&src)),
            }
        }
    }

    /// `(* ... *)` attribute instances on parameter / variable declarations
    /// land in the AST and elaborate into metadata: `type="instance"`,
    /// `units`, `desc` on parameters; `desc`-annotated variables become the
    /// module's operating-point variable list.
    #[test]
    fn attributes_elaborate_to_metadata_and_opvars() {
        let src = r#"
`include "disciplines.vams"
module attres(a, b);
  inout a, b; electrical a, b;
  (* type="instance", units="Ohm", desc="series resistance" *) parameter real R = 1000.0;
  parameter real Tnom = 300.0;
  (* desc="branch current", units="A" *) real ival;
  (* desc="dissipated power" *) real pwr;
  real scratch;
  analog begin
    ival = V(a,b) / R;
    pwr = V(a,b) * ival;
    scratch = 2.0 * pwr;
    I(a,b) <+ ival;
  end
endmodule
"#;
        let ms = parse_modules(src, "attres.va", &[]).expect("parse");
        let m = &ms[0];
        let r = m.params.iter().find(|p| p.name == "R").expect("param R");
        assert_eq!(r.attrs.len(), 3, "three attributes on R");
        assert!(matches!(
            r.attrs.iter().find(|(k, _)| k == "type"),
            Some((_, super::ast::AttrVal::Str(s))) if s == "instance"
        ));
        let em = elaborate(m).expect("elaborate");
        let rp = em.params.iter().find(|p| p.name == "R").unwrap();
        assert!(rp.is_instance, "R is an instance parameter");
        assert_eq!(rp.units.as_deref(), Some("Ohm"));
        assert_eq!(rp.desc.as_deref(), Some("series resistance"));
        let tnom = em.params.iter().find(|p| p.name == "Tnom").unwrap();
        assert!(!tnom.is_instance && tnom.units.is_none() && tnom.desc.is_none());
        // Only desc-annotated variables are op-vars, in declaration order.
        assert_eq!(em.opvars.len(), 2);
        assert_eq!(em.opvars[0].name, "ival");
        assert_eq!(em.opvars[0].units.as_deref(), Some("A"));
        assert_eq!(em.opvars[1].name, "pwr");
        assert_eq!(em.opvars[1].desc, "dissipated power");
        assert!(em.opvars[1].units.is_none());
    }
}
