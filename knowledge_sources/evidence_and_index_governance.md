# Evidence And Index Governance

Scope: evidence rules implemented by the YieldMind knowledge and safety tools.

## Stable evidence identity

Each indexed chunk has a stable `chunk_id`, `text_hash`, `document_id`,
`document_version`, and `index_version`. Reports should retain these fields
together with the source path so a cited passage can be checked later.

## Traceability versus semantic support

A valid chunk ID proves that an indexed chunk exists and matches its stored
hash. It does not prove that the chunk semantically supports the generated
claim. Citation accuracy therefore needs labeled relevance checks or human
review in addition to identifier validation.

## Embedding profile isolation

An index profile locks provider, model ID, revision, vector dimension,
normalization, metric, and query/document instructions. Different embedding
dimensions or model profiles must not share one Chroma collection. Re-indexing
uses a new index version or isolated collection.

## Document lifecycle

Ingestion stores the document hash and version. Repeating the same source hash
and index version is idempotent. Deleting a document marks its metadata as
deleted and removes its vectors so normal retrieval does not cite it.

## Retrieval evaluation

Recall at k measures whether a labeled relevant source appears in the first k
results. Mean reciprocal rank rewards placing the first relevant source near
the top. Citation accuracy at one measures whether the first returned source
matches the labeled source. Latency must be reported with the retrieval mode.
