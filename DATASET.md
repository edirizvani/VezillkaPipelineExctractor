# Vezilka MK↔SQ Parallel Corpus

Sentence-level Macedonian ↔ Albanian parallel data, extracted from bilingual issues of *Службен
весник* (Official Gazette of North Macedonia). Two corpora are released: the current **v2** corpus
and the earlier **v1** export it superseded.

## v2 — current corpus

`pipeline_v2/data/output/` · TSV, UTF-8, tab-separated, one header row.

| File | Rows | Size |
|---|---|---|
| `vezilka_v2_corpus.tsv` | 81,351 | 26 MB |
| `vezilka_v2_train.tsv` | 65,080 | 21 MB |
| `vezilka_v2_val.tsv` | 8,135 | 2.7 MB |
| `vezilka_v2_test.tsv` | 8,136 | 2.7 MB |

`corpus` is the union of the three splits — use the splits for training, `corpus` if you want to
re-split yourself.

### Columns

| Column | Meaning |
|---|---|
| `mk`, `sq` | The aligned Macedonian and Albanian sentences |
| `pdf_id` | Source gazette issue |
| `item_number`, `article_number` | Position within the issue |
| `layout_type` | Layout the issue was extracted with (two-column / sequential / OCR) |
| `alignment_strategy` | Which aligner produced the pair (structural / Gale-Church / dense retrieval) |
| `tier_reached` | How far up the validation cascade the pair got |
| `labse_score`, `laser3_score` | Cross-lingual sentence-embedding similarity |
| `comet_qe_score` | Reference-free COMET-QE translation quality estimate |
| `back_translation_score` | Agreement after round-trip translation |
| `length_ratio_score` | MK/SQ length-ratio plausibility |
| `blended_confidence` | Combined confidence used for the final filter |
| `mk_word_count`, `sq_word_count` | Token counts |

The per-pair scores are kept in the release deliberately: they let you re-filter the corpus at a
stricter confidence threshold than the one used here, without re-running the pipeline.

## v1 — superseded export

`pipeline_v1/data/export/` · TSV, **stored in Git LFS** (both files exceed GitHub's 100 MB limit).

| File | Rows | Size |
|---|---|---|
| `vezilka_mk_sq_full_meta.tsv` | 100,898 | 120 MB |
| `vezilka_mk_sq_cleaned.tsv` | 83,785 | 75 MB |

Columns: `mk`, `sq`, `source`, `article_id`, `confidence`, `method`. `cleaned` is `full_meta` after
the semantic filter.

v1 has *more* rows than v2 but a weaker alignment and validation stack — the higher count reflects
looser filtering, not more usable data. **Prefer v2** unless you specifically want to compare the
two generations.

### Fetching the v1 files

They are LFS pointers, not real files, in a plain `git clone`:

```bash
git lfs install
git clone https://github.com/edirizvani/VezillkaPipelineExctractor.git
git lfs pull        # if you cloned before installing LFS
```

## Provenance and licence

Derived from public-domain legal texts published by the Official Gazette of North Macedonia
(https://slvesnik.com.mk). The alignment is automatic and **not** human-verified: pairs carry
confidence scores rather than guarantees. Expect some misalignment, particularly in OCR-extracted
issues.
