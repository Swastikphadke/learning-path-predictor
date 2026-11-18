
"""PathFinder - Streamlit interface for the Learning Path Predictor."""

import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is available for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core import COURSE_CATALOG
from app.recommendations import recommend_from_checkpoint


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PathFinder | Learning Path Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINTS = {
    "Original Transformer": BASE_DIR / "artifacts" / "course_lm.pt",
    "Basket-Fine-Tuned Transformer": (
        BASE_DIR / "artifacts" / "course_lm_basket.pt"
    ),
}


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 1.8rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #18283c, #172033);
        border: 1px solid #293d58;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #f3f6fc;
        font-size: 2.4rem;
        margin-bottom: 0.3rem;
    }

    .hero p {
        color: #aabbd0;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .course-card {
        padding: 1rem;
        border-radius: 12px;
        background: #171e2b;
        border: 1px solid #293449;
        margin-bottom: 0.8rem;
    }

    .course-id {
        color: #8db8ff;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .course-name {
        color: #f2f5fa;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    .muted {
        color: #9ba9bb;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        background: #171e2b;
        border: 1px solid #293449;
        padding: 1rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🎓 PathFinder</h1>
        <p>
            Your AI-powered learning path predictor.
            Discover what to learn next based on your completed courses.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    available_models = [
        name for name, path in CHECKPOINTS.items()
        if path.exists()
    ]

    if not available_models:
        st.error("No trained model checkpoints found.")
        st.info(
            "Run Milestone 2 training before launching the application."
        )
        st.stop()

    default_model = (
        "Original Transformer"
        if "Original Transformer" in available_models
        else available_models[0]
    )

    selected_model = st.selectbox(
        "Recommendation model",
        available_models,
        index=available_models.index(default_model),
    )

    st.divider()

    st.markdown("### About PathFinder")
    st.caption(
        "PathFinder uses a Transformer language model trained on "
        "synthetic student course histories."
    )

    st.caption("Project: Learning Path Predictor")
    st.caption("Model: PyTorch Transformer")


# --------------------------------------------------
# STUDENT INPUT
# --------------------------------------------------

st.subheader("Your Learning Profile")

st.write(
    "Select the courses you have already completed. "
    "PathFinder will recommend what you could study next."
)

course_ids = list(COURSE_CATALOG.keys())

completed_courses = st.multiselect(
    "Completed courses",
    options=course_ids,
    format_func=lambda course_id: (
        f"{course_id} — {COURSE_CATALOG[course_id]}"
    ),
    placeholder="Search and select your completed courses...",
)

# Progress and selection summary

total_courses = len(COURSE_CATALOG)
completed_count = len(completed_courses)
remaining_count = total_courses - completed_count

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Completed courses", completed_count)
metric2.metric("Available courses", remaining_count)
metric3.metric("Total catalog", total_courses)

if completed_courses:
    st.progress(completed_count / total_courses)

st.divider()


# --------------------------------------------------
# GENERATE RECOMMENDATIONS
# --------------------------------------------------

left, right = st.columns([3, 1])

with left:
    st.subheader("Discover Your Next Courses")
    st.caption(
        f"Selected model: {selected_model}"
    )

with right:
    generate_button = st.button(
        "✨ Generate Learning Path",
        type="primary",
        use_container_width=True,
    )


if generate_button:
    if not completed_courses:
        st.warning(
            "Please select at least one completed course first."
        )

    elif remaining_count == 0:
        st.success(
            "You've completed every course in the current catalog!"
        )

    else:
        checkpoint = CHECKPOINTS[selected_model]

        try:
            with st.spinner("Analyzing your learning history..."):
                recommendations = recommend_from_checkpoint(
                    completed_courses,
                    checkpoint_path=checkpoint,
                    top_n=5,
                )

            st.session_state["recommendations"] = recommendations
            st.session_state["recommendation_model"] = selected_model
            st.session_state["recommendation_history"] = list(
                completed_courses
            )

        except Exception as error:
            st.error(f"Unable to generate recommendations: {error}")


# --------------------------------------------------
# DISPLAY RECOMMENDATIONS
# --------------------------------------------------

if "recommendations" in st.session_state:
    st.divider()

    st.subheader("🚀 Your Recommended Learning Path")

    st.caption(
        f"Generated using {st.session_state['recommendation_model']}"
    )

    previous_history = st.session_state["recommendation_history"]

    if previous_history != completed_courses:
        st.info(
            "Your selected courses have changed. Generate a new learning "
            "path to update these recommendations."
        )

    recommendations = st.session_state["recommendations"]

    if not recommendations:
        st.info("No eligible recommendations are available.")

    else:
        for rank, recommendation in enumerate(recommendations, start=1):
            course_id = recommendation["course_id"]
            course_name = recommendation["course_name"]
            probability = recommendation["probability"]

            with st.container(border=True):
                col_rank, col_course, col_score = st.columns(
                    [0.7, 3, 1.2]
                )

                with col_rank:
                    st.markdown(f"### #{rank}")

                with col_course:
                    st.markdown(f"**{course_name}**")
                    st.caption(course_id)

                with col_score:
                    st.metric(
                        "Model score",
                        f"{probability * 100:.2f}%",
                    )

                st.progress(
                    max(0.0, min(1.0, probability))
                )

        st.caption(
            "Scores represent the model's relative next-course "
            "probabilities among eligible courses. They are not "
            "guarantees of academic suitability."
        )


# --------------------------------------------------
# COMPLETED COURSES
# --------------------------------------------------

with st.expander("View your completed courses"):
    if completed_courses:
        for course_id in completed_courses:
            st.markdown(
                f"✅ **{course_id}** — {COURSE_CATALOG[course_id]}"
            )
    else:
        st.caption("No courses selected yet.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PathFinder | Learning Path Predictor | "
    "Powered by PyTorch and Streamlit"
)
