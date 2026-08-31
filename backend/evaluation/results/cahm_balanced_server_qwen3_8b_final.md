# CAHM Recommended Balanced Evaluation

- Generated: 2026-08-30T11:51:47.663452+00:00
- Gold Set: 40 cases (20 relation, 20 retrieval)
- Embedding: paraphrase-multilingual-MiniLM-L12-v2

## Relation judgement

| Evaluated | Operation accuracy | Operation macro-F1 | Target accuracy | Status accuracy | Processed |
|---|---:|---:|---:|---:|---:|
| True | 0.9000 | 0.9125 | 0.8889 | 0.8571 | 20/20 |

## Retrieval comparison

| Variant | R@1 | R@5 | MRR | Wrong injection | Pending leak | Superseded leak | Retracted leak | Evidence | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_hybrid | 0.5938 | 0.7500 | 0.6875 | 0.3158 | 0.0526 | 0.1053 | 0.0526 | 0.0000 | 9.07 |
| balanced_default | 0.9688 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 7.97 |

Leakage rates use each selected row's original lifecycle status. Evidence completeness requires both evidence text and a source message ID; a missed required claim is incomplete.

## Failure cases

### Relation (2 total)

- `b_e05` gold=COEXIST/103; predicted=NOOP/-
- `b_e06` gold=PENDING/-; predicted=NOOP/-

### legacy_hybrid retrieval (9 total)

- `b_r01` gold=['202']; selected=['201', '202']; wrong=['201']; leaked={'pending': [], 'superseded': ['201'], 'retracted': [], 'archived': []}; missing_evidence=[]
- `b_r02` gold=['201']; selected=['201', '202']; wrong=['202']; leaked={'pending': [], 'superseded': [], 'retracted': [], 'archived': []}; missing_evidence=[]
- `b_r03` gold=['203']; selected=['204']; wrong=['204']; leaked={'pending': ['204'], 'superseded': [], 'retracted': [], 'archived': []}; missing_evidence=[]
- `b_r04` gold=['206']; selected=['205', '206']; wrong=['205']; leaked={'pending': [], 'superseded': [], 'retracted': ['205'], 'archived': []}; missing_evidence=[]
- `b_r06` gold=['209']; selected=[]; wrong=[]; leaked={'pending': [], 'superseded': [], 'retracted': [], 'archived': []}; missing_evidence=[]
- `b_r07` gold=['211']; selected=['212']; wrong=['212']; leaked={'pending': [], 'superseded': [], 'retracted': [], 'archived': []}; missing_evidence=[]
- `b_r13` gold=['217']; selected=['217']; wrong=[]; leaked={'pending': [], 'superseded': [], 'retracted': [], 'archived': []}; missing_evidence=['217']
- `b_r16` gold=['223']; selected=['223', '222']; wrong=['222']; leaked={'pending': [], 'superseded': ['222'], 'retracted': [], 'archived': []}; missing_evidence=[]
- `b_r17` gold=['224']; selected=[]; wrong=[]; leaked={'pending': [], 'superseded': [], 'retracted': [], 'archived': []}; missing_evidence=[]

### balanced_default retrieval (0 total)

No retrieval failures under the Gold definition.

## Effective ablation controls

- `legacy_hybrid`: {"candidate_limit": 100, "evidence_enabled": false, "gate_enabled": true, "include_pending": false, "min_hybrid_score": 0.35, "query_expansion_enabled": false, "rrf_enabled": false, "semantic_enabled": true, "version_filter_enabled": false}; unsupported=[]; historical_query_control_supported=True; embedding_failures=0
- `balanced_default`: {"candidate_limit": 100, "evidence_enabled": true, "gate_enabled": true, "include_pending": false, "min_hybrid_score": 0.35, "query_expansion_enabled": true, "rrf_enabled": true, "semantic_enabled": true, "version_filter_enabled": true}; unsupported=[]; historical_query_control_supported=True; embedding_failures=0
