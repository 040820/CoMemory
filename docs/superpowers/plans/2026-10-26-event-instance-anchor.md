# Event Instance Anchor Implementation Plan

> **For agentic workers:** Implement inline in this session. The user explicitly requested no validation run.

**Goal:** Give each historical activity an upper-layer event identity, attach source facts and Triple snapshots, and record corrections to the same event's date without replacing independent activities.

**Architecture:** Keep the original Chunk storage and middle NetworkX graph. Extend the independent `event_state_layer` payload with event instances and time assertions. Build those records during the existing event-state writer pass and consult the upper correction index in the new structural-semantic retrieval path.

**Tech Stack:** Python, existing pickle-backed CogniGraph, local LLM manager.

**Spec:** Conversation design agreed on 2026-10-26.

## Constraints

- Preserve original Chunk text, graph edges, original LiCoMemory retrieval, and prior experiment files.
- Store structured Triple occurrence references including Chunk ID; never rely on the surviving DiGraph edge alone.
- A correction changes the event's canonical date while retaining the old assertion and its evidence.
- Distinct activities with the same actor and verb stay separate.
- Do not run tests, sample construction, or evaluation in this task, as requested by the user.

## Tasks

- [x] Add event instance and time assertion fields, indexes, and methods to `coregraph/event_state_layer.py`.
- [x] Extract event facts and normalized time provenance, then match possible same-event candidates one by one in `coregraph/event_state_writer.py`.
- [x] Exclude corrected date Triple occurrences from the new retrieval path and expose canonical event evidence in `experiments/integrated_event_state/query.py`.
- [ ] Document the persisted schema, commit the code to GitHub, and sync changed code to the server without running it.
