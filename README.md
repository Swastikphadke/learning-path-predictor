# Learning Path Predictor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/gvkkgf5f58/badge.svg)](https://ask.thili.ai/b/gvkkgf5f58)

The Learning Path Predictor is an EdTech project designed for students. It leverages language model techniques to recommend optimal next courses or study materials based on a student's completed academic journey, helping them navigate their learning path efficiently.

## What you are building

**Goal.** You will build a Learning Path Predictor that recommends next courses or study materials based on a student's completed courses, optimizing their learning paths.

**Done when.** The project is done when the model suggests relevant next courses, proven by displaying top N recommendations for a user's chosen completed courses.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Generate mock student course completion data as sequences of course IDs.**  
      Hint: Use Python's `random` module to create synthetic data, mimicking the 'shopping baskets' from the drill.
- [ ] **2. Pretrain a small language model on these course ID sequences.**  
      Hint: Apply the `lfretail-lfmarketbasket` fine-tuning technique from the drill, treating course IDs as SKUs.
- [ ] **3. Implement a function to predict the next likely course given a partial sequence.**  
      Hint: Adapt the language model's prediction capabilities, similar to predicting the next item in a sequence.
- [ ] **4. Create a Streamlit interface for students to input their completed courses and view recommendations.**  
      Hint: Utilize Streamlit components like `st.multiselect` and `st.write` to build the user interface.
- [ ] **5. Analyze which course "itemsets" the model has learned.**  
      Hint: Inspect the model's embeddings or attention weights to identify frequently co-occurring courses, similar to analyzing market basket associations.

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
