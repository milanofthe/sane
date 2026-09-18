//! `.model` card parsing and the binned model library (`nch.1`, `nch.2`, ...:
//! bin selection by instance geometry, SPICE-style half-open windows).

use rustc_hash::FxHashMap as HashMap;

use crate::expr::resolve_value;
use crate::preprocess::Line;

/// The geometric validity window of a binned model card: a device instance is
/// served by this bin when its length and width fall in `[lmin,lmax]` and
/// `[wmin,wmax]` (half-open at the top, matching SPICE binning).
#[derive(Clone, Copy, Debug)]
pub(crate) struct BinRange {
    pub lmin: f64,
    pub lmax: f64,
    pub wmin: f64,
    pub wmax: f64,
}

impl BinRange {
    pub(crate) fn contains(&self, l: f64, w: f64) -> bool {
        l >= self.lmin && l < self.lmax && w >= self.wmin && w < self.wmax
    }
}

/// A parsed `.model` card: its type keyword (e.g. `NMOS`, `PMOS`, `NPN`, `D`),
/// its numeric parameters, and an optional binning window (set when the card
/// declares `lmin/lmax/wmin/wmax`).
#[derive(Clone, Default)]
pub(crate) struct ModelCard {
    pub mtype: String,
    pub params: Vec<(String, f64)>,
    pub bin: Option<BinRange>,
}

/// A model library: base model name -> its card(s). Most names map to a single
/// card; a binned model (`nch.1`, `nch.2`, ...) maps to several cards selected
/// by the instance geometry.
#[derive(Default)]
pub(crate) struct ModelLib {
    bins: HashMap<String, Vec<ModelCard>>,
}

impl ModelLib {
    /// Select the card for model `name` given an instance length/width. With one
    /// card (the common case) geometry is ignored; with several, the bin whose
    /// window contains `(l, w)` wins, else the first un-binned / first card.
    pub(crate) fn select(&self, name: &str, l: Option<f64>, w: Option<f64>) -> Option<&ModelCard> {
        let cards = self.bins.get(&name.to_ascii_lowercase())?;
        if cards.len() == 1 {
            return Some(&cards[0]);
        }
        if let (Some(l), Some(w)) = (l, w) {
            if let Some(c) = cards
                .iter()
                .find(|c| c.bin.is_some_and(|b| b.contains(l, w)))
            {
                return Some(c);
            }
        }
        cards
            .iter()
            .find(|c| c.bin.is_none())
            .or_else(|| cards.first())
    }
}

/// Strip a trailing numeric bin suffix (`nch.1` -> `nch`); names without a
/// `.<digits>` tail are returned unchanged.
pub(crate) fn model_base_name(name: &str) -> &str {
    match name.rsplit_once('.') {
        Some((base, idx)) if !idx.is_empty() && idx.bytes().all(|b| b.is_ascii_digit()) => base,
        _ => name,
    }
}

/// Tokenise a `.model` card body. Whitespace, parentheses, and commas all
/// separate tokens (so both the `nmos(p=v, ...)` and the `nmos p=v ...` forms
/// parse) -- EXCEPT inside a `{...}` or `'...'` value expression, where they are
/// part of the expression and must be preserved (e.g. a SKY130 statistical card
/// param `toxe={4.148e-9+AGAUSS(0,1,1)*...}`).
pub(crate) fn split_card_tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut brace = 0i32;
    let mut in_quote = false;
    for c in s.chars() {
        if in_quote {
            cur.push(c);
            if c == '\'' {
                in_quote = false;
            }
            continue;
        }
        if brace > 0 {
            match c {
                '{' => brace += 1,
                '}' => brace -= 1,
                _ => {}
            }
            cur.push(c);
            continue;
        }
        match c {
            '\'' => {
                in_quote = true;
                cur.push(c);
            }
            '{' => {
                brace += 1;
                cur.push(c);
            }
            ' ' | '\t' | '(' | ')' | ',' => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Collect `.model NAME TYPE(p=v ...)` cards into a [`ModelLib`].
///
/// Parentheses, commas and whitespace all separate tokens; any token of the
/// form `key=value` is a parameter. Model names are matched case-insensitively.
/// Parameter keys are kept verbatim and must match the device model's symbol
/// suffixes (e.g. `Is`, `N`, `Vt` for a diode). A card carrying
/// `lmin/lmax/wmin/wmax` is registered as a geometry bin under its base name.
pub(crate) fn parse_model_cards(lines: &[Line], env: &HashMap<String, f64>) -> ModelLib {
    let mut lib = ModelLib::default();
    for line in lines {
        if !line.tokens[0].eq_ignore_ascii_case(".model") {
            continue;
        }
        // Rejoin (continuations already merged) and split on whitespace, parens,
        // and commas -- but only OUTSIDE `{...}` / `'...'` value expressions, so a
        // statistical card param like `toxe={4.148e-9+AGAUSS(0,1,1)*...}` keeps its
        // function calls and nested parens intact instead of being shredded.
        let flat = line.tokens.join(" ");
        let tok = split_card_tokens(&flat);
        if tok.len() < 2 {
            continue;
        }
        let name = tok[1].to_ascii_lowercase();
        // The type is the first non-`key=value` token after the model name.
        let mtype = tok
            .get(2..)
            .and_then(|rest| rest.iter().find(|t| !t.contains('=')))
            .map(|s| s.to_string())
            .unwrap_or_default();
        let params: Vec<(String, f64)> = tok
            .iter()
            .filter_map(|t| t.split_once('='))
            .filter_map(|(k, v)| resolve_value(v, env).map(|val| (k.to_string(), val)))
            .collect();
        // A complete `lmin/lmax/wmin/wmax` set marks this card as a geometry bin.
        let p = |key: &str| {
            params
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(key))
                .map(|(_, v)| *v)
        };
        let bin = match (p("lmin"), p("lmax"), p("wmin"), p("wmax")) {
            (Some(lmin), Some(lmax), Some(wmin), Some(wmax)) => Some(BinRange {
                lmin,
                lmax,
                wmin,
                wmax,
            }),
            _ => None,
        };
        // The binning bounds are metadata, not device parameters: drop them so
        // they are not bound onto the instance (nor flagged as unknown later).
        let params: Vec<(String, f64)> = params
            .into_iter()
            .filter(|(k, _)| {
                !matches!(
                    k.to_ascii_lowercase().as_str(),
                    "lmin" | "lmax" | "wmin" | "wmax"
                )
            })
            .collect();
        lib.bins
            .entry(model_base_name(&name).to_string())
            .or_default()
            .push(ModelCard { mtype, params, bin });
    }
    lib
}
