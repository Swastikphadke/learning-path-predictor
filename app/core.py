"""Synthetic student course histories for the Learning Path Predictor."""

from __future__ import annotations

import random

# Stable course IDs act like the SKU tokens from the market-basket drill.
COURSE_CATALOG = {
    "CS101": "Programming Fundamentals",
    "CS102": "Data Structures",
    "CS103": "Algorithms",
    "CS104": "Database Systems",
    "CS105": "Operating Systems",
    "CS106": "Computer Networks",
    "MA101": "Discrete Mathematics",
    "MA102": "Linear Algebra",
    "ST101": "Probability and Statistics",
    "AI201": "Introduction to Artificial Intelligence",
    "AI202": "Machine Learning",
    "AI203": "Deep Learning",
    "AI204": "Natural Language Processing",
    "AI205": "Computer Vision",
    "DS201": "Data Analysis",
    "DS202": "Data Visualization",
    "DS203": "Big Data Systems",
    "WD201": "Web Development",
    "WD202": "Backend Engineering",
    "WD203": "Cloud Computing",
    "SE201": "Software Engineering",
    "SE202": "DevOps Fundamentals",
    "CY201": "Cybersecurity",
    "CY202": "Network Security",
}

LEARNING_PATHS = {
    "software": ["CS101", "MA101", "CS102", "CS103", "CS104", "CS105", "CS106", "SE201", "WD201", "WD202", "SE202", "WD203"],
    "ai": ["CS101", "MA101", "MA102", "ST101", "CS102", "DS201", "AI201", "AI202", "AI203", "AI204", "AI205", "DS202"],
    "data": ["CS101", "MA101", "MA102", "ST101", "CS102", "CS104", "DS201", "DS202", "AI202", "DS203", "WD202", "WD203"],
    "security": ["CS101", "MA101", "CS102", "CS103", "CS105", "CS106", "CY201", "CY202", "SE201", "WD202", "SE202"],
}

CROSS_DISCIPLINARY = ["CS104", "CS106", "AI201", "DS201", "WD201", "SE201", "CY201", "DS202", "WD203"]


def generate_student_sequences(n_students: int = 1_000, seed: int = 42) -> list[list[str]]:
    """Create reproducible, varied course-completion sequences for synthetic students."""
    if n_students < 1:
        raise ValueError("n_students must be at least 1")

    rng = random.Random(seed)
    path_names = list(LEARNING_PATHS)
    path_weights = [0.30, 0.30, 0.25, 0.15]
    sequences: list[list[str]] = []

    for _ in range(n_students):
        path_name = rng.choices(path_names, weights=path_weights, k=1)[0]
        path = LEARNING_PATHS[path_name]
        foundation_count = rng.randint(4, 6)
        advanced_count = rng.randint(2, max(2, len(path) - foundation_count))
        completed = path[:foundation_count] + path[foundation_count:][:advanced_count]

        available = [course for course in CROSS_DISCIPLINARY if course not in completed]
        if available and rng.random() < 0.45:
            completed.append(rng.choice(available))
        available = [course for course in CROSS_DISCIPLINARY if course not in completed]
        if available and rng.random() < 0.15:
            completed.append(rng.choice(available))

        # Minor local variation prevents the model from seeing one rigid order.
        for index in range(0, len(completed) - 1, 2):
            if rng.random() < 0.20:
                completed[index], completed[index + 1] = completed[index + 1], completed[index]
        sequences.append(completed)

    return sequences


def milestone_1(n_students: int = 1_000, seed: int = 42) -> list[list[str]]:
    """Milestone 1: generate mock student histories as course-ID sequences."""
    return generate_student_sequences(n_students=n_students, seed=seed)


# --- Milestone 2: Pretrain a small language model on these course ID sequences.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Implement a function to predict the next likely course given a partial sequence.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Create a Streamlit interface for students to input their completed courses and view recommendations.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Analyze which course itemsets the model has learned.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
