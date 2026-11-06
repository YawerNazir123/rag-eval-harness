# rag-eval-harness

A small, reusable evaluation harness for RAG systems. It runs a golden
QA dataset through any RAG system, scores retrieval and answers with
standard metrics, and writes JSON and Markdown reports.

It was built to evaluate
[document-qa-rag](https://github.com/YawerNazir123/document-qa-rag), a
TF-IDF retrieval system over 744 Kashmir bird species, but it is not
tied to that project: any system can be plugged in through one small
adapter interface.

## Why evals matter for production RAG

A RAG demo that answers five friendly questions in a row proves
nothing. What breaks in production is the long tail: paraphrased
questions that retrieve the wrong chunk, confident answers about
things that are not in the corpus, and citations that point at the
wrong document. Without a fixed golden set and fixed metrics, every
change to the retriever, the chunking, or the prompt is a guess. This
harness makes those changes measurable: run the same 36 questions
before and after, and the numbers tell you whether you improved the
system or just changed which questions it gets wrong.

## What is measured

Retrieval (per question, over the top-k retrieved chunks):

- Recall@k: fraction of the relevant chunks found in the top k.
- MRR: reciprocal rank of the first relevant chunk.
- nDCG@k: rank-aware gain, normalised by the ideal ranking.
- Hit rate@k: whether any relevant chunk is in the top k.

Answers (compared with the expected answer string):

- Token-level F1, exact match, and ROUGE-L (implemented here with a
  plain longest-common-subsequence, so no heavy dependency is needed).
- Citation accuracy: whether the cited species matches the expected
  species, plus citation coverage (how often a citation was given).
- Abstention accuracy: whether the system abstained exactly when it
  should have, on questions about things outside the corpus.
- For out-of-scope questions, the answer metrics are 1.0 when the
  system correctly abstains and 0.0 when it answers; the aggregate
  answer metrics exclude out-of-scope questions, so those 1.0s and
  0.0s never enter the headline numbers.

## Architecture

```
src/rag_eval/
  __init__.py  Public API: re-exports the metrics, runner and systems
               modules, plus __version__.
  metrics.py   Pure functions: recall@k, MRR, nDCG@k, hit rate,
               token F1, exact match, ROUGE-L, citation and
               abstention correctness. No I/O, fully unit-tested.
  systems.py   The adapter interface: SystemResult and Citation
               dataclasses, the RagSystem protocol, plus three
               adapters (DocQARagSystem, HttpRagSystem, StubSystem).
  runner.py    Loads the golden dataset, runs it through a system,
               aggregates metrics overall and per category, and
               writes JSON and Markdown reports.
  __main__.py  CLI: python -m rag_eval run --dataset ... --system ...
data/golden.json   36 QA pairs over the Kashmir Birds corpus.
tests/             pytest: metric unit tests with hand-computed
                   values, dataset validity tests, CLI end-to-end test.
results/           Example reports from a real run (committed on purpose).
```

The runner only talks to the `RagSystem` protocol, so the metrics and
the reporting never change when a new system is added.

## The golden dataset

`data/golden.json` has 36 questions, all grounded in the real
document-qa-rag index (`index/chunks.json`, 744 chunks). Every
expected answer is a substring that appears verbatim in the cited
chunk text, verified programmatically (see `tests/test_dataset.py`).
No fact in the dataset was invented.

- 23 factoid questions: Kashmiri name, length, family, conservation
  status, and migratory status of named species.
- 7 paraphrase questions: the same facts asked in different words,
  including two that never name the species, to expose the known
  TF-IDF paraphrase gap honestly.
- 6 out-of-scope questions: birds that are not in the corpus
  (African Penguin, Emperor Penguin, Ostrich, Kiwi, Toucan, Bald
  Eagle), verified absent from the chunk metadata. A correct system
  must abstain on these.

One quirk worth knowing: document-qa-rag stores `chunk_id: 0` in the
metadata of every one of the 744 chunks, so that field cannot
identify a chunk. The harness therefore uses each chunk's position in
the index list as its id, and the golden dataset's
`relevant_chunk_ids` use the same numbering. If the index is rebuilt
with real chunk ids, the dataset will need regenerating.

The dataset-grounding tests (`tests/test_dataset.py`) run against the
real index when it is present, and otherwise against a committed
fixture, `tests/fixtures/chunks.json`: the same 744-slot index with
only the 15 chunks the golden set cites carrying real text. That keeps
CI verifying grounding even though the real index is not checked out
there.

## Quickstart

The harness itself needs only the standard library.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
python -m pytest tests/ -q
```

Evaluate the stub system (smoke test, no dependencies):

```bash
python -m rag_eval run --dataset data/golden.json --system stub --out results/
```

Evaluate a live HTTP RAG service:

```bash
python -m rag_eval run --dataset data/golden.json --system http \
  --endpoint http://localhost:8000/ask --out results/
```

Evaluate document-qa-rag (needs its own dependencies, numpy and
scikit-learn, because the adapter imports that project):

```bash
# reuse document-qa-rag's virtual environment, or:
pip install numpy scikit-learn
python -m rag_eval run --dataset data/golden.json --system doc-qa-rag --out results/
```

The adapter finds document-qa-rag at
`~/workspace/goals/get-hired-as-an-ai-ml-engineer/hidden_files/document-qa-rag`
by default. Point it elsewhere with `--doc-qa-rag-path PATH` or the
`RAG_EVAL_DOC_QA_RAG_PATH` environment variable. The default path is
resolved once when `rag_eval.systems` is imported, so set the variable
before importing (or pass `--doc-qa-rag-path`).

Each run prints a summary table and writes `report.json` and
`report.md` into the output directory.

### Docker

```bash
docker compose run rag-eval
```

The image as built runs the stub adapter (its `PYTHONPATH` points at
`/app/src`, so the `python -m rag_eval` entrypoint works). The compose
file mounts the document-qa-rag checkout and sets
`RAG_EVAL_DOC_QA_RAG_PATH`, but the image has no numpy or
scikit-learn, so the doc-qa-rag adapter cannot run in it as built: to
run it, install those two packages in the image and change the compose
command to `--system doc-qa-rag`. The HTTP adapter works in the image
as built (it needs only the standard library).

## How to add a new system adapter

Write a class with an `ask` method that returns a `SystemResult`:

```python
from rag_eval.systems import Citation, SystemResult

class MyRagSystem:
    name = "my-rag"

    def ask(self, question: str, k: int = 5) -> SystemResult:
        docs = my_retriever.search(question, top_k=k)
        answer, species = my_answerer.answer(question, docs)
        return SystemResult(
            answer=answer,
            citations=[Citation(species=species)],
            abstained=False,
            retrieved_chunk_ids=[d.chunk_id for d in docs],
        )
```

Then pass it to `run_evaluation` from `rag_eval.runner`, or wire it
into the CLI in `src/rag_eval/__main__.py`. The `retrieved_chunk_ids`
must use the same numbering as the dataset's `relevant_chunk_ids`.

## Results: document-qa-rag (TF-IDF) on the golden set

Run on 2026-11-06 with the command in the quickstart above, k=5,
36 questions. Full reports: `results/report.json` and
`results/report.md`.

| Metric | Value |
| --- | --- |
| Recall@5 | 0.933 |
| MRR | 0.883 |
| nDCG@5 | 0.896 |
| Hit rate@5 | 0.933 |
| Answer token F1 | 0.688 |
| Answer exact match | 0.000 |
| Answer ROUGE-L | 0.688 |
| Citation accuracy | 0.933 |
| Citation coverage | 1.000 |
| Abstention accuracy | 0.917 |

Split by category:

| Category | n | Recall@5 | MRR | Answer F1 | Abstention |
| --- | --- | --- | --- | --- | --- |
| factoid | 23 | 1.000 | 0.957 | 0.750 | 1.000 |
| paraphrase | 7 | 0.714 | 0.643 | 0.483 | 1.000 |
| out_of_scope | 6 | n/a | n/a | n/a | 0.500 |

### Honest interpretation

- Factoid questions are essentially solved: recall@5 is 1.000 and
  every citation names the right species. When the question names a
  species and a field, TF-IDF finds the right chunk.
- The paraphrase gap is real and measurable. On reworded questions
  recall@5 drops to 0.714 and answer F1 to 0.483. The two descriptive
  paraphrases that never name the species both fail: "the owl that
  hunts at night in barns" retrieves Black-crowned Night Heron
  instead of Barn Owl, and "bird of prey with a forked tail" retrieves
  Little Forktail instead of Black Kite. That is a retrieval problem,
  not an answering problem, and it is exactly the case dense
  embeddings are meant to fix.
- Abstention is 3 out of 6. The system correctly abstains on the
  two-word names (African Penguin, Emperor Penguin, Bald Eagle) but
  answers the single-word ones instead: Ostrich gets answered as
  Hoopoe, Kiwi as Himalayan Flameback, Toucan as Himalayan Flameback.
  Its unknown-species check only looks for capitalised phrases of two
  or more words, so a one-word species name slips through. That is a
  concrete, fixable bug this harness caught.
- Exact match is 0.000 even though answers are mostly right. That is
  a metric artefact, not a system failure: the system answers
  "Barn Owl: Kashmiri name: Rata Mogul." while the expected answer is
  the bare field "Kashmiri name: Rata Mogul", so the strings never
  match exactly. Token F1 (0.688) is the meaningful answer metric
  here. If exact match is wanted as a headline number, the expected
  answers should be normalised to the system's answer format first.
- Retrieval metrics are n/a for out-of-scope questions because those
  items have no relevant chunks by definition; abstention accuracy is
  the only meaningful number for them.

## Limitations

- No LLM-as-judge yet. Answer scoring is lexical (token F1, exact
  match, ROUGE-L), so a correct answer phrased differently from the
  expected string scores less than it deserves. An LLM judge that
  grades semantic correctness is a documented future extension, not
  something this harness claims to do. It needs an API key and a
  judge prompt, and it is not implemented here.
- The golden set is small (36 questions) and covers one corpus. It
  is a regression suite, not a benchmark: scores are comparable
  across runs of the same system, not across different corpora.
- The doc-qa-rag adapter depends on that project's index layout and
  its `chunk_id` quirk described above.
- Answer scoring normalises text by stripping non-ASCII characters,
  so words outside the Latin alphabet lose their tokens before token
  F1 and ROUGE-L are computed.
- The HTTP adapter assumes a simple JSON `/ask` contract and cannot
  score retrieval unless the service returns chunk ids.

## Development

```bash
pip install -r requirements-dev.txt
pip install -e .
python -m pytest tests/ -q
```

Continuous integration is not set up yet: the workflow file
(`.github/workflows/ci.yml`, which runs the same command on Python
3.12 for every push) has to be added through the GitHub web UI.

## License

MIT. See `LICENSE`.
