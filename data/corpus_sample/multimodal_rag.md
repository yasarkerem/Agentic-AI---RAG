---
title: "Multimodal RAG: Retrieval Beyond Text"
year: 2025
tags: [multimodal, images, tables, retrieval]
---

# Multimodal RAG: Retrieval Beyond Text

## Scope

Multimodal RAG extends retrieval-augmented generation to corpora containing
images, tables, charts, and document layouts alongside prose. Scientific
papers are a canonical case: figures and tables often carry the exact
result a question asks about, and text-only ingestion silently discards
them.

## Approaches

Two ingestion strategies dominate. Captioning pipelines convert non-text
elements into text — a model describes each figure and serializes each
table — after which standard text retrieval applies; this is simple and
compatible with existing indexes but loses visual detail the caption writer
did not anticipate needing. Native multimodal embedding maps images and
text into a shared vector space so a text query can retrieve an image
directly; this preserves more information but requires multimodal
embedding models and complicates chunking, since a figure's meaning depends
on its caption and surrounding prose.

## Challenges

Reported challenges include evaluation — gold relevance labels for
image regions and table cells are far more expensive to author than for
passages; grounding — citing "figure 3" is only verifiable if the pipeline
preserves stable references from retrieved elements back to source
locations; and layout-aware chunking, because tables split across chunk
boundaries become unrecoverable.

## Relation to text-only benchmarks

Text-only benchmarks remain the foundation because their gold labels are
cheap and their failure analysis is tractable, but conclusions drawn from
them understate retrieval difficulty on real technical corpora. Studies
recommend treating multimodal support as a staged extension: stabilize the
text pipeline and its evaluation first, then add captioned figures and
tables as new document elements with the same provenance discipline.
