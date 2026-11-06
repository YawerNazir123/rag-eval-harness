# RAG evaluation report

System: `doc-qa-rag`, k=5, questions: 36

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

## By category

| Category | n | Recall@5 | MRR | Answer F1 | Abstention |
| --- | --- | --- | --- | --- | --- |
| factoid | 23 | 1.000 | 0.957 | 0.750 | 1.000 |
| paraphrase | 7 | 0.714 | 0.643 | 0.483 | 1.000 |
| out_of_scope | 6 | n/a | n/a | n/a | 0.500 |

## Per-question results

- [OK] q01 (factoid): What is the Kashmiri name of the Greylag Goose? -> Greylag Goose: Kashmiri name: Ans.
- [OK] q02 (factoid): What is the Kashmiri name of the Alpine Accentor? -> Alpine Accentor: Kashmiri name: Zanda tsar.
- [OK] q03 (factoid): What is the Kashmiri name of the Common Kingfisher? -> Common Kingfisher: Kashmiri name: Kola tonch.
- [OK] q04 (factoid): What is the Kashmiri name of the Eurasian Hoopoe? -> Eurasian Hoopoe: Kashmiri name: Satut.
- [OK] q05 (factoid): What is the Kashmiri name of the Black Kite? -> Black Kite: Kashmiri name: Gont.
- [OK] q06 (factoid): What is the Kashmiri name of the Himalayan Monal? -> Himalayan Monal: Kashmiri name: Sunal, Suna Murg, Haum.
- [OK] q07 (factoid): What is the Kashmiri name of the Barn Owl? -> Barn Owl: Kashmiri name: Rata Mogul.
- [OK] q08 (factoid): What is the length of the Alexandrine Parakeet? -> Alexandrine Parakeet: Length: 56–62 cm.
- [OK] q09 (factoid): What is the length of the Common Kingfisher? -> Common Kingfisher: Length: 16 cm.
- [OK] q10 (factoid): What is the length of the Black Kite? -> Black Kite: Length: 55–60 cm.
- [OK] q11 (factoid): What is the length of the Barn Owl? -> Barn Owl: Length: 33–39 cm.
- [OK] q12 (factoid): Which family does the Himalayan Monal belong to? -> Himalayan Monal: Family: Phasianidae.
- [OK] q13 (factoid): Which family does the Eurasian Hoopoe belong to? -> Eurasian Hoopoe: Family: Upupidae.
- [OK] q14 (factoid): Which family does the Common Myna belong to? -> Common Myna: Family: Sturnidae.
- [OK] q15 (factoid): Which family does the Alpine Accentor belong to? -> Alpine Accentor: Family: Prunellidae.
- [OK] q16 (factoid): Which family does the Kashmir Flycatcher belong to? -> Kashmir Flycatcher: Family: Muscicapidae.
- [OK] q17 (factoid): What is the conservation status of the Asian Paradise Flycatcher? -> Asian Paradise Flycatcher: Conservation status: Least Concern.
- [OK] q18 (factoid): What is the conservation status of the Black-hooded Oriole? -> Black-hooded Oriole: Conservation status: Least Concern.
- [OK] q19 (factoid): What is the conservation status of the Black-naped Monarch? -> Black-naped Monarch: Conservation status: Least Concern.
- [OK] q20 (factoid): What is the conservation status of the Blossom-headed Parakeet? -> Blossom-headed Parakeet: Conservation status: Near Threatened.
- [OK] q21 (factoid): What is the migratory status of the Greylag Goose? -> Greylag Goose: Migratory status: Winter Migratory.
- [OK] q22 (factoid): What is the migratory status of the Amur Falcon? -> Amur Falcon: Migratory status: Rare Visitor.
- [OK] q23 (factoid): What is the migratory status of the Common Kingfisher? -> Common Kingfisher: Migratory status: Resident.
- [OK] q24 (paraphrase): What do people in Kashmir call the Greylag Goose? -> Greylag Goose: Kashmiri name: Ans.
- [OK] q25 (paraphrase): What do people in Kashmir call the Alpine Accentor? -> Alpine Accentor: Kashmiri name: Zanda tsar.
- [OK] q26 (paraphrase): What do people in Kashmir call the Common Kingfisher? -> Common Kingfisher: Kashmiri name: Kola tonch.
- [CHECK] q27 (paraphrase): How big is the Alexandrine Parakeet from beak to tail? -> Alexandrine Parakeet: Family: Psittaculidae.
- [OK] q28 (paraphrase): Which bird family is the Himalayan Monal part of? -> Himalayan Monal: Family: Phasianidae.
- [CHECK] q29 (paraphrase): What do locals call the owl that hunts at night in barns? -> Black-crowned Night Heron: Kashmiri name: Bor.
- [CHECK] q30 (paraphrase): Which bird of prey with a forked tail soars over Kashmir towns? -> Little Forktail: Smallest forktail; black-and-white bird of rocky mountain streams and waterfalls, wags and flicks its short forked tail, sometimes plunges underwater dipper-style.
- [OK] q31 (out_of_scope): What is the Kashmiri name of the African Penguin? -> I don't know - not in the documents
- [OK] q32 (out_of_scope): What is the length of the Emperor Penguin? -> I don't know - not in the documents
- [CHECK] q33 (out_of_scope): Which family does the Ostrich belong to? -> Hoopoe: Category: Other.
- [CHECK] q34 (out_of_scope): What is the Kashmiri name of the Kiwi bird? -> Himalayan Flameback: Family: Picidae.
- [CHECK] q35 (out_of_scope): What is the conservation status of the Toucan? -> Himalayan Flameback: Migratory status: Resident.
- [OK] q36 (out_of_scope): What is the Kashmiri name of the Bald Eagle? -> I don't know - not in the documents
