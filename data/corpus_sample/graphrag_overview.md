---
title: "GraphRAG: Knowledge-Graph Retrieval for Global Sensemaking"
year: 2024
tags: [graphrag, knowledge-graph, community-summaries]
---

# GraphRAG: Knowledge-Graph Retrieval for Global Sensemaking

## Main contribution

GraphRAG's main contribution is an indexing pipeline that uses a language
model to extract an entity-and-relationship knowledge graph from the corpus,
then partitions the graph into communities and writes a summary for each
community. Instead of retrieving raw text chunks, global questions are
answered by map-reducing over community summaries: each summary produces a
partial answer, and the partial answers are combined into a final response.

## Retrieval strategy

GraphRAG distinguishes local search from global search. Local search starts
from entities mentioned in the query and gathers their neighborhoods —
related entities, relationships, and the source chunks that mention them.
Global search ignores individual chunks and instead consults the hierarchy
of community summaries, which makes it suitable for corpus-wide questions
like "what are the main themes across these documents" that classic
similarity retrieval handles poorly.

## Strengths

The approach excels at sensemaking questions whose answers are not localized
in any single passage. Community summaries give it a structured, corpus-wide
view, and entity linking makes multi-hop connections explicit rather than
hoping nearest-neighbor search stumbles onto them. Provenance is preserved
because graph elements point back to their source text.

## Limitations

Index construction is substantially more expensive than plain vector
indexing: entity extraction and community summarization require many model
calls over the whole corpus. The quality of the graph depends on extraction
accuracy, and sparse or noisy corpora yield fragmented communities. For
narrow factoid lookups, the graph machinery adds cost without improving over
straightforward dense retrieval of chunks.
