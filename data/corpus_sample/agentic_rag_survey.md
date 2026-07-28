---
title: "Agentic RAG: Planning, Iteration, and Control"
year: 2025
tags: [agentic-rag, planning, decomposition, stopping-rules]
---

# Agentic RAG: Planning, Iteration, and Control

## From pipeline to agent

Standard RAG is a fixed pipeline: retrieve once, generate once. Agentic RAG
reframes retrieval as a task an agent plans and iterates on: analyze the
question, decompose it into subquestions, retrieve evidence per
subquestion, judge whether the evidence suffices, reformulate and search
again where it does not, then synthesize an answer with verified citations.
The defining property is a feedback loop — retrieval outcomes influence
subsequent retrieval decisions.

## Core components

Surveyed systems converge on a recognizable set of roles: a planner that
produces subquestions and evidence requirements; a retriever invoked per
subquestion; an evidence assessor that detects insufficient or off-topic
results; a query rewriter for weak searches; a contradiction checker for
conflicting sources; a synthesizer; and a citation verifier. Implementations
range from prompted single-model loops to multi-agent systems, but the
control flow is broadly shared.

## Where iteration pays

Reported gains concentrate on complex tasks: multi-hop questions whose
evidence must be assembled across documents, comparative analyses, and
research synthesis requiring broad source coverage. On simple factoid
lookups the agentic machinery mostly adds latency and cost — one good
retrieval pass already suffices — so several systems route easy queries
past the loop entirely, using task-complexity classification at intake.

## Control and safety of the loop

Uncontrolled iteration is the characteristic failure: an agent that keeps
reformulating a hopeless query burns budget without converging, or loops
between similar queries indefinitely. Deployed systems therefore impose
hard bounds — maximum retrieval rounds, maximum subquestions, token and
cost budgets, wall-clock timeouts — and deterministic stopping rules that
fire when evidence is judged sufficient or when successive rounds stop
adding new information. Recording the stop reason with each run is reported
as essential for debugging and for honest cost accounting.
