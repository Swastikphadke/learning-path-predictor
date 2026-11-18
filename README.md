# Learning Path Predictor

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/gvkkgf5f58/badge.svg)](https://msrit.ask.thili.ai/b/gvkkgf5f58)

The Learning Path Predictor is an EdTech project designed for students. It leverages language model techniques to recommend optimal next courses or study materials based on a student's completed academic journey, helping them navigate their learning path efficiently.

- [x] Milestone 1: Generate synthetic student course completion sequences.
- [x] Milestone 2: Pretrain a Transformer language model on course sequences.
- [x] Milestone 2: Continue pretraining on randomized market-basket sequences.
- [x] Milestone 2: Evaluate learned itemset recovery against exact frequent itemsets.

### Milestone 2 — Results

**Pretraining**
- Architecture: Compact causal Transformer.
- Training data: 1,000 synthetic student histories.
- Epochs: 8.
- Final training loss: 0.8267.
- Validation loss: 0.8073.

**Market-Basket Fine-Tuning**
- Fine-tuned on randomized course baskets.
- Epochs: 3.
- Learning rate: 0.0005.
- Initial validation loss: 5.7662.
- Final validation loss: 3.4912.
- Exact frequent itemsets mined: 931.

**Itemset Recovery Evaluation**
- Generated baskets: 810.
- Reference itemsets: 931.
- Generated itemsets: 1,073.
- Recovered itemsets: 807.
- Precision: 75.21%.
- Recall: 86.68%.
- F1 Score: 80.54%.

Note: Itemset recovery is evaluated against the training-distribution reference
and should not be interpreted as held-out generalization performance.

Checkpoint files:
- `artifacts/course_lm.pt`
- `artifacts/course_lm_basket.pt`


- [x] Milestone 3: Predict the next likely course given a partial sequence.

### Milestone 3 — Next-Course Prediction

Implemented:
- Autoregressive next-course prediction using a causal Transformer.
- Top-N recommendations with probability scores.
- Filtering of previously completed courses.
- Support for original and basket-fine-tuned checkpoints.

**Held-Out Evaluation**

Dataset:
- 1,000 synthetic student histories.
- 800 training students.
- 200 test students.
- 1,821 next-course prediction examples.

| Metric | Original Transformer | Basket-Fine-Tuned |
|---|---:|---:|
| Top-1 Accuracy | 76.83% | 74.74% |
| Hit@5 | 96.87% | 96.43% |
| MRR | 0.8561 | 0.8401 |

The original Transformer performed better on all three
next-course prediction metrics.

The basket-fine-tuned model remains useful for learning
course co-occurrence patterns, as demonstrated by the
Milestone 2 itemset recovery experiment.

The original Transformer is therefore selected as the
default recommendation model.

Note: Evaluation uses synthetic student histories and
does not establish real-world recommendation performance.


- [x] Milestone 4: Build the Streamlit interface.
- [x] Milestone 5: Analyze learned itemsets, embeddings, attention weights, and co-occurrences.

### Milestone 4 — Streamlit Application

Implemented PathFinder, a Streamlit-based learning path
recommendation interface.

Features:
- Interactive completed-course selection.
- Original and basket-fine-tuned model selection.
- Top-5 next-course recommendations.
- Recommendation probability scores.
- Completed-course progress dashboard.

Run locally:

python -m streamlit run app/streamlit_app.py

### Milestone 5 — Model Analysis

Implemented:
- Frequent itemset mining using thilimarketbasket.
- Course embedding similarity using cosine similarity.
- Per-head causal self-attention extraction.
- Analysis of learned course co-occurrence patterns.

Example findings:
- CS101, CS102 and MA101 occur together with 100% support.
- CS101, CS102 and DS201 occur together with 62.3% support.
- Highest observed embedding similarity:
  CS105 and ST101: 0.3549.
- Attention analysis showed Head 2 assigning
  approximately 99.997% attention to CS102
  when processing CS103 in the inspected sequence.

Note: These findings are exploratory and based on
synthetic student histories. Embedding similarity
and attention weights should not be interpreted
as definitive explanations of academic relationships.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow students to filter recommendations by subject area or difficulty level.
- Display a confidence score or probability alongside each recommended course.
- Implement a feature to visualize the student's completed courses and recommended next steps as a learning graph.
