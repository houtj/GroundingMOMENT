# GroundingMOMENT demonstrations

These silent screen recordings illustrate the research interface described in
the EMNLP 2026 paper. They are included as visual examples only; the interface,
underlying model, checkpoints, and source time-series files are not distributed
in this release.

## Clips

- [`synthetic_pattern_grounding.mp4`](synthetic_pattern_grounding.mp4)
  (6.8 seconds) — a compositional synthetic query over repeated temporal
  patterns.
- [`drilling_cycle_grounding.mp4`](drilling_cycle_grounding.mp4)
  (4.5 seconds) — a drilling-position query expressed as a quick step followed
  by a prolonged decrease.
- [`ecg_heartbeat_query.mp4`](ecg_heartbeat_query.mp4)
  (5.3 seconds) — an ECG signal and language-guided heartbeat query.
- [`ecg_event_grounding.mp4`](ecg_event_grounding.mp4)
  (11.9 seconds) — joint heartbeat-event and primitive-span visualization on
  an ECG signal.

An edited, research-focused narration is provided in
[`transcript.md`](transcript.md).

## Interpretation

Colored bands represent predicted temporal spans. Event-level spans cover the
composite occurrence; primitive-level spans show the lower-level waveform
structure associated with phrases in the query. The clips demonstrate the
interaction concept and should not be interpreted as independent clinical,
industrial, or production validation.

The underlying drilling and ECG series are not included here. Refer to the
published paper for data sources, evaluation protocols, results, and
limitations.
