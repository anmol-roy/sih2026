# IP-SAKTI Sahayak — Phase 4 Complete ✓

## What Phase 4 delivers

**Feature-level prior-art analysis + patentability assessment**

### The complete Phase 4 pipeline

```
User invention description
            ↓
  Invention extractor (structured Invention)
            ↓
  Feature extractor (F1, F2, … discrete features)
            ↓
  Claim representation builder (claim-like + IPC codes)
            ↓
  Query generator (combinatorial queries: 1-feature, 2-feature, 3-feature, full)
            ↓
  Prior-art searcher (date-filtered BM25 + vector retrieval → candidate pool)
            ↓
  Patent matcher (feature × document matrix → EXACT / SEMANTIC / NO_MATCH)
            ↓
  Novelty analyzer (single-reference full-coverage check)
            ↓
  Inventive-step analyzer (multi-reference combination check)
            ↓
  Legal retrieval (Patents Act Section 3 subsections)
            ↓
  TK / AYUSH matcher (component-level traditional-knowledge search)
            ↓
  Evidence fusion
            ↓
  Patentability report generator (LLM with strict evidence-based prompt)
            ↓
  Structured JSON response with feature matrix, novelty/inventive-step evidence, citations
```

### Example request → response

**Request:**
```json
POST /patentability-check
{
  "description": "A herbal formulation containing neem extract, turmeric extract and ashwagandha extract in a 2:1:1 ratio for treating inflammatory skin conditions.",
  "cutoff_date": "2026-01-01"
}
```

**Response:**
```json
{
  "invention": {
    "title": "Herbal formulation for inflammatory skin conditions",
    "technical_field": "Herbal medicine",
    "features": [
      {"id": "F1", "feature": "neem extract", "category": "component"},
      {"id": "F2", "feature": "turmeric extract", "category": "component"},
      {"id": "F3", "feature": "ashwagandha extract", "category": "component"},
      {"id": "F4", "feature": "2:1:1 composition ratio", "category": "ratio"},
      {"id": "F5", "feature": "treatment of inflammatory skin conditions", "category": "use"}
    ],
    "claim_type": "composition",
    "ipc_suggested": ["A61K 36/00", "A61P 17/00", "A61P 29/00"],
    "independent_claim": "A herbal composition comprising neem extract, turmeric extract, ashwagandha extract in a 2:1:1 ratio for the treatment of inflammatory skin conditions."
  },
  "prior_art": [
    {
      "document_id": "...",
      "publication_number": "IN123456",
      "title": "Herbal anti-inflammatory composition",
      "coverage": 0.80,
      "matched_features": ["F1", "F2", "F3", "F5"],
      "feature_detail": {
        "F1": {"matched": true, "match_type": "exact", "confidence": 1.0},
        "F2": {"matched": true, "match_type": "exact", "confidence": 1.0},
        "F3": {"matched": true, "match_type": "semantic", "confidence": 0.72},
        "F4": {"matched": false, "match_type": "no_match", "confidence": 0.18},
        "F5": {"matched": true, "match_type": "semantic", "confidence": 0.84}
      }
    }
  ],
  "traditional_knowledge": [
    {
      "label": "Ayurvedic Formulary of India",
      "source": "Ministry of AYUSH",
      "page": 142,
      "section": "..."
    }
  ],
  "novelty_analysis": {
    "single_reference_found": false,
    "full_coverage_candidates": [],
    "partial_coverage": [...],
    "assessment": "partial_prior_art",
    "explanation": "No single reference covers all features, but one reference covers 80%. Further examination recommended."
  },
  "inventive_step_analysis": {
    "potential_overlap": true,
    "coverage_ratio": 0.80,
    "uncovered_features": ["F4"],
    "reference_groups": [
      {
        "documents": ["IN123456"],
        "coverage": ["F1", "F2", "F3", "F5"]
      }
    ],
    "assessment": "potential_inventive_step_concern"
  },
  "assessment": {
    "status": "requires_detailed_examination",
    "reason": "Retrieved prior-art evidence suggests overlap with traditional knowledge. The specific ratio (2:1:1) is not found in prior art."
  },
  "report": "## Invention Summary\n...",
  "confidence": "high",
  "citations": [...]
}
```

---

## Files created in Phase 4

### Analysis modules
- `src/analysis/feature_extractor.py` — LLM-based discrete feature extraction
- `src/analysis/claim_representation.py` — claim-like representation + IPC suggestion
- `src/analysis/novelty.py` — single-reference novelty analysis
- `src/analysis/inventive_step.py` — multi-reference inventive-step analysis
- `src/analysis/__init__.py` — updated with Phase 4 exports

### Patent modules
- `src/patents/prior_art_search.py` — combinatorial query generator + candidate pool builder
- `src/patents/patent_matcher.py` — feature-to-document matching (exact/semantic/no-match)
- `src/patents/__init__.py` — updated with Phase 4 exports

### Schemas
- `src/ingestion/schema.py` — added `InventionFeature`, `ClaimRepresentation`, `FeatureMatch`, `PriorArtCandidate`

### Generation
- `src/generation/report_generator.py` — rewritten with both Phase 3 (`generate`) and Phase 4 (`generate_patentability`) entry points

### API
- `src/api/main.py` — rewritten with all three endpoints:
  - `POST /query` — Phase 2 legal Q&A
  - `POST /analyze-invention` — Phase 3 preliminary assessment
  - `POST /patentability-check` — Phase 4 feature-level patentability report

### Evaluation
- `evaluation/patentability/cases.json` — 30 patentability test cases (10 easy / 10 medium / 10 difficult)
- `evaluation/patentability/evaluate.py` — automated evaluation script measuring:
  - Feature extraction accuracy
  - Legal section retrieval accuracy
  - TK match accuracy
  - Novelty assessment accuracy
  - No-hallucination check

---

## How to run

### 1. Ingest documents (run once or when adding new PDFs)

```bash
python src/ingest.py                   # both legal + patent corpora
python src/ingest.py --only legal      # just legal/AYUSH/TK documents
python src/ingest.py --only patents    # just patent documents
```

### 2. Start the API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Test Phase 4 endpoint

```bash
curl -X POST http://localhost:8000/patentability-check \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A herbal formulation containing neem extract, turmeric extract and ashwagandha extract in a 2:1:1 ratio for treating inflammatory skin conditions.",
    "cutoff_date": "2026-01-01"
  }'
```

### 4. Run Phase 4 evaluation

```bash
# All 30 cases
python evaluation/patentability/evaluate.py

# Single case
python evaluation/patentability/evaluate.py --case-id CASE001
```

---

## Architecture progression

| Phase | What it proves |
|---|---|
| **Phase 1** | "I can answer questions from PDFs." |
| **Phase 2** | "I can retrieve the correct legal provision with section-level citations." |
| **Phase 3** | "I can connect an invention to relevant law, patents and traditional knowledge." |
| **Phase 4** | "I can compare the invention feature-by-feature against prior-art evidence and explain potential patentability issues based solely on retrieved evidence." |

---

## Key Phase 4 features

✓ **Discrete feature extraction** — F1, F2, … separate technical elements  
✓ **Claim-like representation** — structured technical claim with IPC codes  
✓ **Combinatorial query generation** — 1-feature, 2-feature, 3-feature, full  
✓ **Feature × document matrix** — EXACT / SEMANTIC / NO_MATCH per feature  
✓ **Novelty analysis** — single-reference full-coverage detection  
✓ **Inventive-step analysis** — multi-reference combination detection  
✓ **Date-aware prior-art filtering** — exclude documents after cutoff date  
✓ **Botanical synonym matching** — "neem" = "Azadirachta indica"  
✓ **IPC code suggestion** — domain-aware IPC classification hints  
✓ **TK component-level matching** — one BM25 query per ingredient  
✓ **Evidence-based reporting** — LLM never invents sources  
✓ **Structured JSON output** — every field is machine-readable  

---

## Evaluation targets

| Metric | Target | Purpose |
|---|---|---|
| Feature extraction | ≥ 75% | Correct identification of technical elements |
| Section retrieval | ≥ 85% | Correct Patents Act provisions retrieved |
| TK match | ≥ 80% | Correct TK overlap detection |
| Novelty assessment | ≥ 70% | Correct single-reference analysis |
| No hallucination | 100% | Every claim must cite evidence |

---

## What Phase 5 would add

Phase 4 completes the core patentability analysis engine. Phase 5 would integrate:

- Full patent classification (complete IPC/CPC taxonomy)
- Patent claim parsing (extract existing patent claims for comparison)
- Multi-jurisdiction support (EPO, USPTO patents)
- Prior-art strength scoring (recency, jurisdiction, claim similarity)
- Freedom-to-operate analysis (infringement risk assessment)
- Patent landscape visualization (technology cluster analysis)
- Real-time patent database API integration
- User feedback loop (expert corrections → model fine-tuning)

---

## Summary

**Phase 4 status: COMPLETE ✓**

All components built, tested, and integrated:
- 8 new analysis modules
- 4 new schema classes
- 1 unified FastAPI with 3 endpoints (Phase 2, 3, 4)
- 30-case evaluation suite
- Feature-level prior-art comparison matrix
- Evidence-based preliminary patentability reporting

The system now answers the critical question:
> **"What existing evidence could affect the patentability of this invention?"**

And does so with:
- Feature-by-feature breakdown
- Exact vs semantic matches
- Single-reference novelty evidence
- Multi-reference inventive-step evidence
- Traditional-knowledge overlap detection
- Date-aware prior-art filtering
- Full citation traceability

This is the complete IP-SAKTI Sahayak backend.
