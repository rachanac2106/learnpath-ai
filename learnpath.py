"""
LearnPath AI — Personalized Learning Path Generator
Adaptive learning using knowledge graph + collaborative filtering
Author: Rachana C (rachanac2106)
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from collections import defaultdict
from enum import Enum


class Difficulty(Enum):
    BEGINNER     = 1
    INTERMEDIATE = 2
    ADVANCED     = 3
    EXPERT       = 4


@dataclass
class Topic:
    id: str
    name: str
    difficulty: Difficulty
    prerequisites: List[str] = field(default_factory=list)
    category: str = ""
    estimated_hours: float = 2.0
    tags: List[str] = field(default_factory=list)

@dataclass
class LearnerProfile:
    user_id: str
    name: str
    known_topics: Set[str] = field(default_factory=set)
    in_progress: Dict[str, float] = field(default_factory=dict)   # topic_id → progress %
    learning_speed: float = 1.0    # 1.0 = average
    preferred_difficulty: Difficulty = Difficulty.INTERMEDIATE
    goal: str = ""
    completed_hours: float = 0.0


# ─────────────────────────────────────────────
# Knowledge Graph
# ─────────────────────────────────────────────

class KnowledgeGraph:
    """DAG of topics with prerequisite relationships."""

    def __init__(self):
        self.topics: Dict[str, Topic] = {}
        self._build_cs_curriculum()

    def _build_cs_curriculum(self):
        topics = [
            Topic("py_basics",    "Python Basics",         Difficulty.BEGINNER,     [],                          "Programming", 5),
            Topic("data_struct",  "Data Structures",       Difficulty.INTERMEDIATE, ["py_basics"],               "CS Fundamentals", 8),
            Topic("algorithms",   "Algorithms",            Difficulty.INTERMEDIATE, ["data_struct"],             "CS Fundamentals", 10),
            Topic("oop",          "OOP Concepts",          Difficulty.INTERMEDIATE, ["py_basics"],               "Programming", 6),
            Topic("ml_basics",    "ML Fundamentals",       Difficulty.INTERMEDIATE, ["py_basics", "algorithms"], "AI/ML", 8),
            Topic("numpy",        "NumPy & Pandas",        Difficulty.INTERMEDIATE, ["py_basics"],               "Data Science", 4),
            Topic("ml_models",    "ML Models (sklearn)",   Difficulty.ADVANCED,     ["ml_basics", "numpy"],      "AI/ML", 10),
            Topic("dl_basics",    "Deep Learning Basics",  Difficulty.ADVANCED,     ["ml_models"],               "AI/ML", 12),
            Topic("nlp",          "NLP & Transformers",    Difficulty.ADVANCED,     ["dl_basics"],               "AI/ML", 12),
            Topic("fastapi",      "FastAPI Backend",       Difficulty.INTERMEDIATE, ["py_basics", "oop"],        "Backend", 6),
            Topic("react",        "React.js Frontend",     Difficulty.INTERMEDIATE, [],                          "Frontend", 8),
            Topic("sql",          "SQL & Databases",       Difficulty.BEGINNER,     [],                          "Data", 5),
            Topic("docker",       "Docker & Containers",   Difficulty.INTERMEDIATE, [],                          "DevOps", 4),
            Topic("git",          "Git & Version Control", Difficulty.BEGINNER,     [],                          "Tools", 3),
            Topic("system_design","System Design",         Difficulty.EXPERT,       ["fastapi", "docker", "sql"],"Architecture", 15),
            Topic("langchain",    "LangChain & LLM Apps",  Difficulty.ADVANCED,     ["nlp", "fastapi"],          "AI/ML", 10),
        ]
        for t in topics:
            self.topics[t.id] = t

    def get_prerequisites(self, topic_id: str) -> List[Topic]:
        t = self.topics.get(topic_id)
        if not t:
            return []
        return [self.topics[p] for p in t.prerequisites if p in self.topics]

    def get_unlocked_topics(self, known: Set[str]) -> List[Topic]:
        """Topics where all prerequisites are met."""
        unlocked = []
        for t in self.topics.values():
            if t.id in known:
                continue
            if all(p in known for p in t.prerequisites):
                unlocked.append(t)
        return unlocked

    def topological_sort(self, target_ids: List[str]) -> List[Topic]:
        """Return topics in learning order (prerequisites first)."""
        visited = set()
        order = []

        def visit(tid):
            if tid in visited or tid not in self.topics:
                return
            visited.add(tid)
            for prereq in self.topics[tid].prerequisites:
                visit(prereq)
            order.append(self.topics[tid])

        for tid in target_ids:
            visit(tid)
        return order


# ─────────────────────────────────────────────
# Collaborative Filtering Recommender
# ─────────────────────────────────────────────

class CollaborativeFilter:
    """
    User-based collaborative filtering for topic recommendations.
    Finds similar learners and recommends what they studied next.
    """

    def __init__(self):
        # Simulated user-topic interaction matrix
        self.user_topics = {
            "alice":   {"py_basics", "data_struct", "algorithms", "ml_basics", "ml_models"},
            "bob":     {"py_basics", "oop", "fastapi", "docker", "sql"},
            "carol":   {"py_basics", "numpy", "ml_basics", "ml_models", "dl_basics", "nlp"},
            "dave":    {"git", "react", "sql", "fastapi", "docker", "system_design"},
            "eve":     {"py_basics", "ml_basics", "numpy", "ml_models", "langchain"},
        }

    def _jaccard_similarity(self, a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    def recommend(self, learner: LearnerProfile, top_n: int = 5) -> List[str]:
        """Find similar users → recommend topics they've studied that learner hasn't."""
        known = learner.known_topics
        similarities = []
        for user, topics in self.user_topics.items():
            sim = self._jaccard_similarity(known, topics)
            similarities.append((user, sim, topics))
        similarities.sort(key=lambda x: -x[1])

        # Gather candidate topics from top-3 similar users
        candidates = defaultdict(float)
        for user, sim, topics in similarities[:3]:
            for t in topics:
                if t not in known:
                    candidates[t] += sim

        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        return [t for t, _ in sorted_candidates[:top_n]]


# ─────────────────────────────────────────────
# Adaptive Learning Path Generator
# ─────────────────────────────────────────────

class LearnPathAI:
    """
    Generates personalized, adaptive learning paths.
    Combines knowledge graph + collaborative filtering + difficulty adaptation.
    """

    def __init__(self):
        self.graph = KnowledgeGraph()
        self.cf = CollaborativeFilter()

    def generate_path(self, learner: LearnerProfile, goal_topics: List[str]) -> Dict:
        print(f"\n  [LearnPath AI] Generating path for {learner.name}...")
        print(f"  Goal: {learner.goal or ', '.join(goal_topics)}")
        print(f"  Known topics: {len(learner.known_topics)}")

        # Step 1: Topological sort of goal topics
        full_path = self.graph.topological_sort(goal_topics)

        # Step 2: Filter out already-known topics
        to_learn = [t for t in full_path if t.id not in learner.known_topics]

        # Step 3: Add CF recommendations
        cf_recs = self.cf.recommend(learner, top_n=3)
        for rec_id in cf_recs:
            rec = self.graph.topics.get(rec_id)
            if rec and rec not in to_learn and rec.id not in learner.known_topics:
                to_learn.append(rec)

        # Step 4: Estimate timeline
        total_hours = sum(
            t.estimated_hours / learner.learning_speed for t in to_learn
        )

        # Step 5: Build weekly schedule
        schedule = self._build_schedule(to_learn, learner.learning_speed)

        return {
            "learner": learner.name,
            "goal": learner.goal,
            "path": to_learn,
            "total_topics": len(to_learn),
            "total_hours": round(total_hours, 1),
            "estimated_weeks": math.ceil(total_hours / 10),
            "schedule": schedule,
            "cf_recommendations": cf_recs,
        }

    def _build_schedule(self, topics: List[Topic], speed: float) -> List[Dict]:
        """Assign topics to weeks (10 hours/week study plan)."""
        schedule = []
        week = 1
        week_hours = 0.0
        week_topics = []

        for t in topics:
            hours = t.estimated_hours / speed
            if week_hours + hours > 10 and week_topics:
                schedule.append({"week": week, "topics": week_topics, "hours": round(week_hours, 1)})
                week += 1
                week_hours = 0.0
                week_topics = []
            week_topics.append(t)
            week_hours += hours

        if week_topics:
            schedule.append({"week": week, "topics": week_topics, "hours": round(week_hours, 1)})

        return schedule

    def print_path(self, result: Dict):
        print("\n" + "=" * 60)
        print(f"  📚 LearnPath AI — Personalized Plan for {result['learner']}")
        print("=" * 60)
        print(f"  Goal            : {result['goal']}")
        print(f"  Topics to learn : {result['total_topics']}")
        print(f"  Total hours     : {result['total_hours']}h")
        print(f"  Est. duration   : {result['estimated_weeks']} weeks (10h/week)")
        print("-" * 60)

        for week_plan in result["schedule"]:
            topics_str = ", ".join(t.name for t in week_plan["topics"])
            print(f"  Week {week_plan['week']:2d} ({week_plan['hours']}h): {topics_str}")

        print(f"\n  🤝 Also recommended (based on similar learners):")
        for rec in result["cf_recommendations"]:
            t = self.graph.topics.get(rec)
            if t:
                print(f"     • {t.name} ({t.difficulty.name})")

        print("=" * 60)


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ai = LearnPathAI()

    # Learner 1: Rachana — wants to become AI Engineer
    rachana = LearnerProfile(
        user_id="rachana_001",
        name="Rachana",
        known_topics={"py_basics", "git", "sql", "react"},
        learning_speed=1.2,  # slightly faster than average
        goal="Become an AI/ML Engineer",
    )
    result = ai.generate_path(rachana, ["langchain", "system_design", "dl_basics"])
    ai.print_path(result)

    print()

    # Learner 2: Backend-focused learner
    arjun = LearnerProfile(
        user_id="arjun_002",
        name="Arjun",
        known_topics={"py_basics"},
        learning_speed=0.9,
        goal="Full Stack Backend Developer",
    )
    result2 = ai.generate_path(arjun, ["fastapi", "docker", "system_design"])
    ai.print_path(result2)
