# YieldMind Literature Corpus

`manifest.json` is the source of truth for the curated open-literature corpus.
Downloaded documents are intentionally excluded from Git and must be reproduced
with:

```bash
/opt/anaconda3/envs/amla/bin/python scripts/download_yieldmind_literature.py
```

After verification, ingest both corpora with:

```bash
/opt/anaconda3/envs/amla/bin/python scripts/ingest_yieldmind_corpora.py
```

Project rules use `700/80` character chunks. Full papers use `1800/180`
because the smaller project-document setting produced excessive fragments on
the initial eight-paper corpus. Both values are encoded in `split_version`.

Every entry records its landing page, direct HTTPS download, DOI, reuse terms,
topics, and expected SHA-256. The downloader rejects private-network addresses,
unexpected content types, files over 100 MiB, and hash changes. A changed remote
file requires manual review and a manifest update; the hash must not be refreshed
automatically.

The corpus is supporting evidence, not ground truth. Equations, experimental
conditions, material systems, and measurement protocols must be cited with the
returned page number and checked for applicability before they are used in a
YieldMind model.
