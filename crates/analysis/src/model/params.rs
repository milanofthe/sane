//! The named parameter store of a model: bound values and defaults, the
//! hierarchical name resolution, the parameter vector.

use super::*;

/// Named parameter store: bound values, defaults, and the hierarchical name
/// resolution / `p`-vector assembly that used to live in the Python wrapper.
pub(super) struct ParamStore {
    /// Parameter names in the engine's column order.
    pub(super) pnames: Vec<String>,
    /// Fast membership for column names.
    pub(super) pset: HashSet<String>,
    /// Intermediate group prefixes ("X1", "X1.D1" for "X1.D1.Is").
    pub(super) prefixes: HashSet<String>,
    /// Current bound values (canonical name -> value), behind a lock so `set`
    /// works through `&self` and `Model` stays `Send + Sync`.
    pub(super) values: Mutex<HashMap<String, f64>>,
    /// Construction-time snapshot for `reset`.
    pub(super) defaults: HashMap<String, f64>,
}

/// The module default of every parameter column, by symbol (see
/// [`sane_dae::Dae::param_defaults`]): the value an unstated parameter takes.
pub(super) fn column_defaults(cdc: &CompiledDc) -> Vec<Option<f64>> {
    let defaults = &cdc.dae().param_defaults;
    cdc.param_syms()
        .iter()
        .map(|s| defaults.get(s).copied())
        .collect()
}

impl ParamStore {
    pub(super) fn new(
        pnames: Vec<String>,
        bound: HashMap<String, f64>,
        defaults: Vec<Option<f64>>,
    ) -> Self {
        let pset: HashSet<String> = pnames.iter().cloned().collect();
        let mut prefixes = HashSet::new();
        for p in &pnames {
            let parts: Vec<&str> = p.split('.').collect();
            for i in 1..parts.len() {
                prefixes.insert(parts[..i].join("."));
            }
        }
        // Keep only bound values that are actually parameters (canonicalised).
        let mut known = HashMap::new();
        for (k, v) in bound {
            if let Some(c) = Self::resolve_in(&pset, &k) {
                known.insert(c, v);
            }
        }
        // Parameters the deck left unspecified take the device's default.
        for (p, d) in pnames.iter().zip(defaults) {
            if let (false, Some(v)) = (known.contains_key(p), d) {
                known.insert(p.clone(), v);
            }
        }
        ParamStore {
            pnames,
            pset,
            prefixes,
            defaults: known.clone(),
            values: Mutex::new(known),
        }
    }

    pub(super) fn resolve_in(pset: &HashSet<String>, name: &str) -> Option<String> {
        if pset.contains(name) {
            return Some(name.to_string());
        }
        let r = sane_circuit::value_symbol_name(name);
        if pset.contains(&r) {
            Some(r)
        } else {
            None
        }
    }

    pub(super) fn resolve(&self, name: &str) -> Option<String> {
        Self::resolve_in(&self.pset, name)
    }

    pub(super) fn get(&self, name: &str) -> Option<f64> {
        let c = self.resolve(name)?;
        Some(self.values.lock().unwrap().get(&c).copied().unwrap_or(0.0))
    }

    pub(super) fn set(&self, name: &str, value: f64) -> Result<(), ModelError> {
        let c = self
            .resolve(name)
            .ok_or_else(|| ModelError::UnknownParam(name.to_string()))?;
        self.values.lock().unwrap().insert(c, value);
        Ok(())
    }

    pub(super) fn reset(&self) {
        *self.values.lock().unwrap() = self.defaults.clone();
    }

    pub(super) fn values_map(&self) -> HashMap<String, f64> {
        self.values.lock().unwrap().clone()
    }

    /// Parameter symbols that appear in the residual but have no bound value, so a
    /// numeric analysis silently uses `0` for them (a symbolic analysis keeps them
    /// free). Typically a foundry `.param`/global the deck never defines (e.g.
    /// `mc_mm_switch`), whose `0` default then propagates through the model card.
    /// Sorted for a stable message.
    pub(super) fn unbound(&self) -> Vec<String> {
        let vals = self.values.lock().unwrap();
        let mut u: Vec<String> = self
            .pnames
            .iter()
            .filter(|n| !vals.contains_key(*n))
            .cloned()
            .collect();
        u.sort();
        u
    }

    /// The parameter vector in column order, from the bound values plus per-call
    /// overrides (canonicalised; unknown override keys are ignored).
    pub(super) fn pvec(&self, overrides: &[(&str, f64)]) -> Vec<f64> {
        let vals = self.values.lock().unwrap();
        if overrides.is_empty() {
            return self
                .pnames
                .iter()
                .map(|n| vals.get(n).copied().unwrap_or(0.0))
                .collect();
        }
        let mut ov = HashMap::new();
        for (k, v) in overrides {
            if let Some(c) = self.resolve(k) {
                ov.insert(c, *v);
            }
        }
        self.pnames
            .iter()
            .map(|n| ov.get(n).or_else(|| vals.get(n)).copied().unwrap_or(0.0))
            .collect()
    }
}
