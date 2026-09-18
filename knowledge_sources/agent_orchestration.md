# Agent Orchestration And Tool Contracts

Scope: current YieldMind StateGraph and Function Calling boundaries.

## Domain StateGraph

The domain workflow uses explicit nodes for preparation, data handling,
requirements, search, candidate generation, model generation, pre-execution,
operation, review, and finish. Conditional edges control bounded revision and
cancellation rather than relying on an unbounded conversational loop.

## Multi-agent roles

The original requirement, search, candidate, model, operation, and review
responsibilities remain distinct even though LangGraph now owns the shared
state and transitions. Multi-agent value comes from role separation and
verification boundaries, not from the number of model calls.

## Tool Registry

The Tool Registry publishes Pydantic JSON schemas and validates arguments
before invoking data profiling, baseline evaluation, candidate benchmarking,
knowledge retrieval, evidence validation, reporting, or sandbox execution.
Unknown tool names and invalid parameters must fail explicitly.

## Offline and live modes

The offline rule planner is deterministic and records zero model calls. Real
Function Calling requires `allow_live_llm=true` and must report actual call
counts separately. Simulated adapters and skipped live tests are not evidence
of real model behavior.

## Retry and termination budgets

Workflow repair, revision, Tool Calling, and anomaly drill-down need explicit
limits. A failed verifier can route to a bounded local repair or replan, while
cancellation routes to a terminal state and should not silently continue work.
