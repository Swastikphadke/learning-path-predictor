# Learning Path Predictor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/gvkkgf5f58/badge.svg)](https://ask.thili.ai/b/gvkkgf5f58)

The Learning Path Predictor is an EdTech project designed for students. It leverages language model techniques to recommend optimal next courses or study materials based on a student's completed academic journey, helping them navigate their learning path efficiently.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Generate mock student course completion data as sequences of course IDs.**
      Implemented reproducible synthetic student histories using Python's `random` module, with four learning paths and cross-disciplinary variation.

- [ ] **2. Pretrain a small language model on these course ID sequences.**
      Implemented a compact causal Transformer using PyTorch, with course-ID tokenization, next-token prediction, and checkpoint saving. Trained on 1,000 synthetic student histories for 8 epochs. Training loss: 0.827; validation loss: 0.807. Integration of the `lfretail-lfmarketbasket` fine-tuning technique remains to be completed.

- [ ] **3. Implement a function to predict the next likely course given a partial sequence.**
      Adapt the language model's prediction capabilities, similar to predicting the next item in a sequence.

- [ ] **4. Create a Streamlit interface for students to input their completed courses and view recommendations.**
      Utilize Streamlit components like `st.multiselect` and `st.write` to build the user interface.

- [ ] **5. Analyze which course "itemsets" the model has learned.**
      Inspect the model's embeddings or attention weights to identify frequently co-occurring courses, similar to analyzing market basket associations.

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
