# Netlist test fixtures: sources and licenses

A diverse set of small SPICE-like netlists used to exercise the parser and the
DAE extraction across circuit topologies. Adapted to SANE's element set
(R/C/L/V/I, E/G controlled sources, D/M/Q devices, `.model`), with nodes
renumbered to be contiguous and device-model parameters renamed to SANE's
symbol suffixes (`Is/N/Vt`, `Kp/W/L/Vth`, `Is/Vt/betaF/betaR`).

## Sources

- **Tony R. Kuphaldt, "Lessons in Electric Circuits"** — released under the
  Design Science License (DSL), a free-content copyleft license. Safe to use
  with attribution. https://www.ibiblio.org/kuphaldt/electricCircuits/
  (DSL: https://www.ibiblio.org/kuphaldt/electricCircuits/Devel/dsl.html)
  Fixtures: series_rlc, parallel_tank, rc_timeconst, bridge_rectifier,
  diode_clipper, biased_clipper, zener_clipper, bjt_ce_min,
  bjt_emitter_follower, nmos_curve.

- **S. Dusausay, "Colpitts"** — MIT license, with attribution.
  https://github.com/nimisbert/Colpitts
  Fixtures: colpitts_oscillator.

- **Own / synthesized** (standard textbook topologies, re-authored with our own
  node numbering and component values; unencumbered):
  voltage_divider, rc_lowpass, bjt_current_mirror, cmos_inverter,
  mos_current_mirror, op_inverting, sallen_key_lp.

Topologies for op-amps and common-source/CE amplifiers were inspired by
eCircuitCenter.com and the McGill SPICE decks, but those sources carry no open
license, so the fixtures here are re-authored derivatives with our own values
rather than verbatim copies.

- **G. W. Roberts (McGill), companion SPICE decks to Sedra & Smith,
  "Microelectronic Circuits"** — educational courseware, no explicit open-source
  license; reproduced here for interoperability testing with attribution. Single
  soft-wrapped lines were rejoined and the leading title line marked as a comment;
  device connectivity and the standard Sedra-Smith (Gray & Meyer) model
  parameters are otherwise verbatim.
  http://www.ece.mcgill.ca/~grober4/SPICE/
  Fixtures: ua741 (Fig. 10.2; model parameters cross-checked against the
  MIT-licensed mirror github.com/sina96n/ua741-Design-and-Simulation),
  cmos_diffpair_ota (Fig. 6.25), multistage_bjt_opamp (Example 6.2).
