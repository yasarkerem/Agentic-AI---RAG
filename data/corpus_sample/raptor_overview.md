---
title: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
year: 2024
tags: [raptor, hierarchical-retrieval, summarization]
---

# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

## Main contribution

RAPTOR's main contribution is a hierarchical retrieval index built by
recursively clustering text chunks and summarizing each cluster with a
language model. The process repeats bottom-up until a single root summary
remains, producing a tree whose leaves are raw chunks and whose internal
nodes are increasingly abstract summaries. At query time the system can
retrieve from any level of the tree, so a question about fine-grained
details matches leaf chunks while a question about themes or overall
arguments matches higher-level summary nodes.

## Retrieval strategy

Two traversal modes are described. Tree traversal walks from the root
downward, selecting the most relevant children at each level. Collapsed-tree
retrieval flattens all nodes into a single pool and ranks them together by
similarity, which the authors found performed better in practice. Because
summary nodes compress information from many chunks, collapsed-tree
retrieval lets a single retrieved node stand in for evidence that would
otherwise require many separate chunks.

## Strengths

RAPTOR helps most on questions that require integrating information
scattered across a long document, such as narrative comprehension or
multi-step reasoning over a book chapter. Summary nodes give the generator
compact, thematically coherent context. The method is index-time heavy but
query-time cheap: clustering and summarization happen once during ingestion.

## Limitations

The tree is expensive to build because every internal node requires a
summarization call, and the index must be rebuilt or patched when documents
change. Summaries can also blur or omit details, so purely factoid questions
sometimes do better with plain chunk retrieval. Cluster quality depends on
the embedding model, and errors in early clustering propagate upward into
every ancestor summary.
