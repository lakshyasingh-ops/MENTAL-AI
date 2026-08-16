import random
import io
import re

import streamlit as st
import pandas as pd

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recommendation import (
    load_courses,
    create_vectors,
    recommend_topics
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MENTAL AI - Intelligent Recommendation System",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

courses = load_courses()

vectorizer, vectors = create_vectors(courses)

questions = pd.read_csv("data/questions.csv")


# =========================================================
# PDF FUNCTIONS
# =========================================================

@st.cache_data
def extract_pdf_text(pdf_bytes):

    reader = PdfReader(
        io.BytesIO(pdf_bytes)
    )

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def create_text_chunks(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if len(current_chunk) + len(sentence) < 700:

            current_chunk += " " + sentence

        else:

            if current_chunk:
                chunks.append(
                    current_chunk.strip()
                )

            current_chunk = sentence

    if current_chunk:
        chunks.append(
            current_chunk.strip()
        )

    return chunks


def search_document(
    question,
    chunks,
    number_of_results=3
):

    if not chunks:
        return []

    documents = chunks + [question]

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = tfidf.fit_transform(
        documents
    )

    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    )[0]

    ranked_indexes = similarities.argsort()[::-1]

    results = []

    for index in ranked_indexes[:number_of_results]:

        score = similarities[index]

        if score <= 0:
            continue

        results.append({
            "text": chunks[index],
            "score": score * 100
        })

    return results


# =========================================================
# PERSONALIZED QUESTION GENERATION
# =========================================================

STOP_WORDS = {
    "about", "after", "again", "against", "also", "because",
    "before", "being", "between", "could", "does", "during",
    "each", "from", "have", "into", "more", "most", "other",
    "over", "such", "than", "that", "their", "there", "these",
    "they", "this", "those", "through", "under", "using", "were",
    "which", "while", "with", "would", "your", "what", "when",
    "where", "whose", "will", "shall", "should", "must", "been",
    "very", "only", "some", "many", "much", "used", "use",
    "make", "made", "like", "then", "them", "the", "and",
    "for", "are", "was", "has", "had", "its", "our", "out",
    "not", "but", "can", "may", "one", "two", "three", "first",
    "second", "new"
}


def extract_candidate_terms(text, limit=40):
    """
    Find likely study concepts from the document.

    We prefer terms that:
    - occur more than once
    - contain useful alphabetic words
    - are not common English stop words
    """

    cleaned = re.sub(r"\s+", " ", text).strip()

    words = re.findall(
        r"\b[A-Za-z][A-Za-z-]{3,}\b",
        cleaned.lower()
    )

    frequencies = {}

    for word in words:

        if word in STOP_WORDS:
            continue

        frequencies[word] = (
            frequencies.get(word, 0) + 1
        )

    ranked = sorted(
        frequencies.items(),
        key=lambda item: (-item[1], item[0])
    )

    return [
        word
        for word, _ in ranked[:limit]
    ]


def analyze_document_topics(
    document_text,
    chunks,
    detected_concepts,
    limit=8
):
    """
    Build a document-specific topic profile.

    Topic coverage is calculated from the uploaded PDF itself:
    percentage of extracted text sections that mention a concept.
    We also count total mentions so repeated concepts can be ranked.

    We prefer concepts that came from explicit definition statements,
    then fall back to the existing NLP keyword detector.
    """
    if not document_text or not chunks:
        return []

    definition_pairs = extract_definition_pairs(document_text)
    use_pairs = extract_use_pairs(document_text)

    candidates = []

    # Definitions are stronger evidence of an actual study topic than
    # a raw frequency keyword, so put them first.
    for item in definition_pairs:
        candidates.append(item["concept"])

    for item in use_pairs:
        candidates.append(item["concept"])

    candidates.extend(detected_concepts)

    unique_candidates = []
    seen = set()

    for candidate in candidates:
        candidate = normalize_concept(candidate)
        key = concept_key(candidate)

        if not concept_is_valid(candidate):
            continue

        if key in seen:
            continue

        seen.add(key)
        unique_candidates.append(candidate)

    results = []
    total_sections = max(len(chunks), 1)
    lowered_text = document_text.lower()

    for concept in unique_candidates:
        pattern = re.compile(
            r"(?<![A-Za-z0-9])"
            + re.escape(concept.lower())
            + r"(?![A-Za-z0-9])",
            re.IGNORECASE
        )

        mentions = len(pattern.findall(lowered_text))

        if mentions == 0:
            continue

        section_count = 0

        for chunk in chunks:
            if pattern.search(chunk):
                section_count += 1

        coverage = (
            section_count / total_sections
        ) * 100

        # Coverage is the main signal. Mentions provide a small tie-breaker.
        ranking_score = coverage + min(mentions, 50) * 0.15

        results.append({
            "topic": concept.title(),
            "mentions": mentions,
            "sections": section_count,
            "coverage": coverage,
            "ranking_score": ranking_score
        })

    results.sort(
        key=lambda item: (
            -item["ranking_score"],
            -item["mentions"],
            item["topic"].lower()
        )
    )

    return results[:limit]


def get_document_content_profile(document_text):
    """
    Give a lightweight profile of the uploaded material.

    This is deliberately descriptive rather than pretending to know
    the academic difficulty of the document.
    """
    text_lower = document_text.lower()

    technical_signals = sum([
        len(re.findall(r"\bclass\b", text_lower)),
        len(re.findall(r"\bobject[- ]oriented\b", text_lower)),
        len(re.findall(r"\bmethod\b", text_lower)),
        len(re.findall(r"\balgorithm\b", text_lower)),
        len(re.findall(r"\bprogramming\b", text_lower)),
        len(re.findall(r"\bfunction\b", text_lower)),
        len(re.findall(r"\bdata structure\b", text_lower)),
    ])

    definition_signals = len(re.findall(
        r"\b(?:is|are|refers to|means|defined as)\b",
        text_lower
    ))

    code_signals = len(re.findall(
        r"\b(?:public|private|protected|static|void|int|string|import)\b",
        text_lower
    ))

    if code_signals >= 12 or technical_signals >= 20:
        content_type = "Technical / programming-heavy"
    elif definition_signals >= 20:
        content_type = "Concept and theory-heavy"
    else:
        content_type = "Mixed study material"

    return {
        "content_type": content_type,
        "definition_signals": definition_signals,
        "code_signals": code_signals
    }


def clean_question_text(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"^[•●▪◦\-–—]+\s*", "", text)
    return text


def normalize_concept(text):
    """
    Turn a possible PDF heading/phrase into a short study concept.

    PDF extraction often produces things such as:
    "INTRODUCTION TO JAVA Java"
    or
    "Java Java"

    Those are not useful quiz concepts, so we clean them here.
    """
    text = clean_question_text(text)

    # Remove common heading prefixes accidentally attached to a concept.
    text = re.sub(
        r"^(?:introduction\s+to|chapter\s+\d+|unit\s+\d+|"
        r"topic\s*[:\-]?|definition\s*[:\-]?)\s+",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^[\s,:;.\-–—]+|[\s,:;.\-–—]+$",
        "",
        text
    )

    # Remove repeated adjacent words caused by PDF extraction.
    words = text.split()
    cleaned_words = []

    for word in words:
        if (
            cleaned_words
            and re.sub(r"[^a-z0-9]+", "", cleaned_words[-1].lower())
            == re.sub(r"[^a-z0-9]+", "", word.lower())
        ):
            continue

        cleaned_words.append(word)

    text = " ".join(cleaned_words)

    # A concept should be short enough to be a real topic name.
    return text.strip()


def concept_is_valid(concept):
    """
    Conservative validation.

    MENTAL AI must prefer fewer good questions over questions made from
    arbitrary PDF lines.
    """
    concept = normalize_concept(concept)

    if not concept:
        return False

    words = concept.split()

    if len(words) > 4 or len(words) < 1:
        return False

    if len(concept) > 45:
        return False

    if not re.search(r"[A-Za-z]", concept):
        return False

    bad_starts = {
        "what", "which", "this", "that", "these", "those",
        "it", "they", "he", "she", "we", "you", "there",
        "since", "because", "when", "where", "how",
        "can", "could", "will", "would", "should",
        "has", "have", "had", "also", "then", "thus",
        "a", "an", "the", "is", "are"
    }

    if words[0].lower() in bad_starts:
        return False

    bad_phrases = {
        "introduction to",
        "according to",
        "for example",
        "as follows",
        "in this",
        "it is",
        "there is",
        "there are",
        "can be",
        "is also",
        "are also",
        "has its",
        "have its"
    }

    lowered = concept.lower()

    if any(phrase in lowered for phrase in bad_phrases):
        return False

    # A concept containing too many function words is probably a sentence
    # fragment rather than a topic.
    function_words = {
        "the", "a", "an", "is", "are", "was", "were", "of",
        "to", "and", "or", "for", "in", "on", "with", "by",
        "from", "that", "this"
    }

    function_count = sum(
        1 for word in words if word.lower() in function_words
    )

    if len(words) >= 3 and function_count >= 2:
        return False

    return True


def extract_sentences(document_text):
    """
    Split PDF text into reasonably complete sentences.

    We deliberately do not treat every PDF line as a question.
    """
    cleaned = re.sub(r"\s+", " ", document_text).strip()

    if not cleaned:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned
    )

    result = []

    for sentence in sentences:
        sentence = clean_question_text(sentence)

        if 35 <= len(sentence) <= 320:
            result.append(sentence)

    return result


def _dedupe_key(text):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        clean_question_text(text).lower()
    ).strip()


def _clean_definition(definition):
    definition = clean_question_text(definition)

    # Remove leading article.
    definition = re.sub(
        r"^(?:a|an|the)\s+",
        "",
        definition,
        flags=re.IGNORECASE
    )

    return definition.strip()


def extract_definition_pairs(document_text):
    """
    Extract only explicit concept -> definition statements.

    Examples:
        Java is a high level programming language.
        A class is a blueprint for creating objects.
        Encapsulation refers to ...

    If the PDF does not contain a sufficiently clear definition,
    MENTAL AI skips it rather than inventing a question.
    """
    sentences = extract_sentences(document_text)
    pairs = []

    connector_pattern = (
        r"is|are|refers\s+to|means|is\s+known\s+as|"
        r"can\s+be\s+defined\s+as|is\s+defined\s+as"
    )

    pattern = re.compile(
        rf"^(?P<concept>[A-Za-z][A-Za-z0-9&()/+\- ]{{0,65}}?)"
        rf"\s+(?P<connector>{connector_pattern})\s+"
        rf"(?P<definition>[^.!?]{{25,260}})[.!?]?$",
        re.IGNORECASE
    )

    for sentence in sentences:
        match = pattern.match(sentence)

        if not match:
            continue

        raw_concept = match.group("concept")
        definition = _clean_definition(
            match.group("definition")
        )

        # If the PDF extraction contains a heading immediately before
        # the real concept, try the last 4 words as the concept.
        concept = normalize_concept(raw_concept)

        if not concept_is_valid(concept):
            raw_words = raw_concept.split()

            for size in range(
                min(4, len(raw_words)),
                0,
                -1
            ):
                candidate = normalize_concept(
                    " ".join(raw_words[-size:])
                )

                if concept_is_valid(candidate):
                    concept = candidate
                    break

        if not concept_is_valid(concept):
            continue

        definition = clean_question_text(definition)

        if len(definition.split()) < 5:
            continue

        if len(definition.split()) > 45:
            continue

        if "?" in definition:
            continue

        # A definition should contain meaningful words, not just a
        # repeated copy of the concept.
        if _dedupe_key(definition) == _dedupe_key(concept):
            continue

        pairs.append({
            "concept": concept,
            "definition": definition,
            "sentence": sentence
        })

    # Keep the first useful definition for each concept.
    unique = []
    seen = set()

    for pair in pairs:
        key = _dedupe_key(pair["concept"])

        if not key or key in seen:
            continue

        seen.add(key)
        unique.append(pair)

    return unique


def extract_use_pairs(document_text):
    """
    Extract explicit statements such as:
        X is used to ...
        X is used for ...
    """
    sentences = extract_sentences(document_text)
    pairs = []

    pattern = re.compile(
        r"^(?P<concept>[A-Za-z][A-Za-z0-9&()/+\- ]{0,65}?)"
        r"\s+is\s+used\s+(?P<purpose>to|for)\s+"
        r"(?P<explanation>[^.!?]{20,240})[.!?]?$",
        re.IGNORECASE
    )

    for sentence in sentences:
        match = pattern.match(sentence)

        if not match:
            continue

        concept = normalize_concept(
            match.group("concept")
        )

        explanation = clean_question_text(
            match.group("explanation")
        )

        if not concept_is_valid(concept):
            continue

        if len(explanation.split()) < 5:
            continue

        pairs.append({
            "concept": concept,
            "purpose": (
                match.group("purpose") + " " + explanation
            ).strip(),
            "sentence": sentence
        })

    unique = []
    seen = set()

    for item in pairs:
        key = _dedupe_key(item["concept"])

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def concept_key(text):
    return _dedupe_key(text)


def _similarity_key(text):
    """
    Used only to avoid nearly identical answer choices.
    """
    words = [
        word for word in re.findall(
            r"[a-zA-Z0-9]+",
            text.lower()
        )
        if word not in STOP_WORDS
    ]

    return " ".join(words)


def select_concept_distractors(
    correct_concept,
    pairs,
    used_option_sets
):
    """
    Pick other real concepts from the PDF.

    We never use random high-frequency words as concept options.
    """
    correct_key = concept_key(correct_concept)
    candidates = []
    seen = {correct_key}

    for pair in pairs:
        concept = normalize_concept(pair["concept"])
        key = concept_key(concept)

        if not concept_is_valid(concept):
            continue

        if key in seen:
            continue

        seen.add(key)
        candidates.append(concept)

    # Randomize instead of always selecting the same first concepts.
    random.shuffle(candidates)

    for _ in range(3):
        if len(candidates) < 3:
            break

        distractors = candidates[:3]

        option_set = tuple(sorted(
            [correct_key]
            + [concept_key(item) for item in distractors]
        ))

        if option_set not in used_option_sets:
            return distractors

        candidates = candidates[1:] + candidates[:1]

    return []


def select_definition_distractors(
    correct_pair,
    pairs,
    used_answer_sets
):
    """
    Pick other definitions from the PDF.

    We prefer definitions that are reasonably similar in length.
    """
    correct = clean_question_text(
        correct_pair["definition"]
    )

    candidates = []

    for pair in pairs:
        definition = clean_question_text(
            pair["definition"]
        )

        if not definition:
            continue

        if definition.lower() == correct.lower():
            continue

        if definition.lower() in [
            item.lower() for item in candidates
        ]:
            continue

        candidates.append(definition)

    random.shuffle(candidates)

    candidates.sort(
        key=lambda value: abs(
            len(value.split()) - len(correct.split())
        )
    )

    for start in range(
        0,
        max(1, len(candidates) - 2)
    ):
        selected = candidates[start:start + 3]

        if len(selected) != 3:
            continue

        answer_set = tuple(sorted(
            [_similarity_key(correct)]
            + [_similarity_key(item) for item in selected]
        ))

        if answer_set not in used_answer_sets:
            return selected

    return []


def make_mcq(
    question_text,
    correct_answer,
    distractors,
    topic
):
    """Create a clean four-option MCQ."""
    correct_answer = clean_question_text(
        correct_answer
    )

    options = [correct_answer]

    for distractor in distractors:
        distractor = clean_question_text(
            distractor
        )

        if not distractor:
            continue

        if distractor.lower() == correct_answer.lower():
            continue

        if distractor.lower() in [
            item.lower() for item in options
        ]:
            continue

        options.append(distractor)

        if len(options) == 4:
            break

    if len(options) != 4:
        return None

    random.shuffle(options)

    return {
        "question": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "topic": topic
    }


def generate_personalized_questions(
    document_text,
    number_of_questions=5
):
    """
    Generate sensible PDF-grounded questions.

    IMPORTANT:
    This version does NOT use arbitrary PDF lines as fallback questions.

    It generates questions only when MENTAL AI can identify a clear:
        concept -> definition
    or:
        concept -> purpose

    This is intentionally conservative. Five sensible questions are
    better than ten meaningless questions.
    """
    definitions = extract_definition_pairs(
        document_text
    )

    uses = extract_use_pairs(
        document_text
    )

    questions_generated = []
    used_topics = set()
    used_option_sets = set()
    used_answer_sets = set()

    # ---------------------------------------------------------
    # Style 1: "What is X?"
    # ---------------------------------------------------------
    # This is the preferred style because it is easy to understand
    # and keeps the question directly connected to the PDF.
    # ---------------------------------------------------------
    definition_candidates = definitions.copy()
    random.shuffle(definition_candidates)

    for pair in definition_candidates:
        topic = pair["concept"]
        topic_key = concept_key(topic)

        if topic_key in used_topics:
            continue

        distractors = select_definition_distractors(
            pair,
            definitions,
            used_answer_sets
        )

        if len(distractors) != 3:
            continue

        question = make_mcq(
            (
                "According to your uploaded study material, "
                f"what is **{topic}**?"
            ),
            pair["definition"],
            distractors,
            topic.title()
        )

        if not question:
            continue

        answer_set = tuple(sorted(
            _similarity_key(value)
            for value in question["options"]
        ))

        if answer_set in used_answer_sets:
            continue

        questions_generated.append(question)
        used_topics.add(topic_key)
        used_answer_sets.add(answer_set)

        if len(questions_generated) >= number_of_questions:
            return questions_generated

    # ---------------------------------------------------------
    # Style 2: "Which concept matches this definition?"
    # ---------------------------------------------------------
    # Only use this if we have enough real concepts.
    # ---------------------------------------------------------
    concept_candidates = definitions.copy()
    random.shuffle(concept_candidates)

    for pair in concept_candidates:
        topic = pair["concept"]
        topic_key = concept_key(topic)

        if topic_key in used_topics:
            continue

        distractors = select_concept_distractors(
            topic,
            definitions,
            used_option_sets
        )

        if len(distractors) != 3:
            continue

        question = make_mcq(
            (
                "According to your uploaded study material, "
                "which concept matches this description?\n\n"
                f"\"{pair['definition']}\""
            ),
            topic,
            distractors,
            topic.title()
        )

        if not question:
            continue

        option_set = tuple(sorted(
            concept_key(value)
            for value in question["options"]
        ))

        if option_set in used_option_sets:
            continue

        questions_generated.append(question)
        used_topics.add(topic_key)
        used_option_sets.add(option_set)

        if len(questions_generated) >= number_of_questions:
            return questions_generated

    # ---------------------------------------------------------
    # Style 3: Explicit "used for" questions
    # ---------------------------------------------------------
    use_candidates = uses.copy()
    random.shuffle(use_candidates)

    for item in use_candidates:
        topic = item["concept"]
        topic_key = concept_key(topic)

        if topic_key in used_topics:
            continue

        distractors = select_definition_distractors(
            {
                "definition": item["purpose"],
                "concept": topic
            },
            uses,
            used_answer_sets
        )

        if len(distractors) != 3:
            continue

        question = make_mcq(
            (
                "According to your uploaded study material, "
                f"what is **{topic}** used for?"
            ),
            item["purpose"],
            distractors,
            topic.title()
        )

        if not question:
            continue

        answer_set = tuple(sorted(
            _similarity_key(value)
            for value in question["options"]
        ))

        if answer_set in used_answer_sets:
            continue

        questions_generated.append(question)
        used_topics.add(topic_key)
        used_answer_sets.add(answer_set)

        if len(questions_generated) >= number_of_questions:
            return questions_generated

    return questions_generated


# =========================================================
# HEADER
# =========================================================

st.title("🎓 MENTAL AI")

st.subheader(
    "AI-Powered Personalized Learning Recommendation System"
)

st.write(
    "MENTAL AI analyzes your learning performance and "
    "recommends what you should study next."
)


# =========================================================
# STUDENT NAME
# =========================================================

name = st.text_input(
    "Enter your name"
)


# =========================================================
# ASSESSMENT MODE
# =========================================================

assessment_mode = st.radio(
    "🎯 Choose Assessment Mode",
    [
        "Demo Assessment",
        "Personalized Assessment"
    ],
    format_func=lambda mode: (
        "🎯 Demo Assessment — Use MENTAL AI sample questions"
        if mode == "Demo Assessment"
        else "📄 Personalized Assessment — Use your own study material"
    )
)


# =========================================================
# QUIZ
# =========================================================

if name and assessment_mode == "Demo Assessment":

    st.header(
        f"Welcome, {name} 👋"
    )

    st.write(
        "Take the quiz below to analyze your "
        "learning performance."
    )

    st.divider()

    with st.form("quiz_form"):

        answers = []

        for index, question in questions.iterrows():

            st.subheader(
                f"Question {index + 1}"
            )

            st.write(
                question["question"]
            )

            options = [
                "Select an answer",
                question["option_a"],
                question["option_b"],
                question["option_c"],
                question["option_d"]
            ]

            selected_answer = st.radio(
                "Choose your answer:",
                options,
                key=f"question_{index}"
            )

            answers.append(
                selected_answer
            )

        submitted = st.form_submit_button(
            "🚀 Submit Assessment"
        )


    # =====================================================
    # PROCESS QUIZ
    # =====================================================

    if submitted:

        unanswered = []

        for index, answer in enumerate(answers):

            if answer == "Select an answer":

                unanswered.append(
                    index + 1
                )


        if unanswered:

            st.error(
                "Please answer all questions before "
                "submitting the assessment."
            )

            st.write(
                "Unanswered questions: "
                + ", ".join(
                    map(str, unanswered)
                )
            )


        else:

            # =============================================
            # CALCULATE TOPIC SCORES
            # =============================================

            topic_scores = {}

            for index, question in questions.iterrows():

                topic = question["topic"]

                correct_answer = question["answer"]

                option_letters = {
                    question["option_a"]: "A",
                    question["option_b"]: "B",
                    question["option_c"]: "C",
                    question["option_d"]: "D"
                }

                selected_letter = option_letters[
                    answers[index]
                ]

                if topic not in topic_scores:

                    topic_scores[topic] = {
                        "correct": 0,
                        "total": 0
                    }

                topic_scores[topic]["total"] += 1

                if selected_letter == correct_answer:

                    topic_scores[topic]["correct"] += 1


            # =============================================
            # OVERALL SCORE
            # =============================================

            total_correct = sum(
                data["correct"]
                for data in topic_scores.values()
            )

            total_questions = sum(
                data["total"]
                for data in topic_scores.values()
            )

            overall_score = (
                total_correct /
                total_questions
            ) * 100


            # =============================================
            # CLASSIFY TOPICS
            # =============================================

            strong_topics = []

            average_topics = []

            weak_topics = []


            for topic, data in topic_scores.items():

                score = (
                    data["correct"] /
                    data["total"]
                ) * 100


                if score >= 75:

                    strong_topics.append(
                        (topic, score)
                    )


                elif score >= 60:

                    average_topics.append(
                        (topic, score)
                    )


                else:

                    weak_topics.append(
                        (topic, score)
                    )


            # =============================================
            # DASHBOARD
            # =============================================

            st.divider()

            st.header(
                "📊 Your Learning Dashboard"
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Overall Score",
                    f"{overall_score:.0f}%"
                )


            with col2:

                st.metric(
                    "Strong Topics",
                    len(strong_topics)
                )


            with col3:

                st.metric(
                    "Average Topics",
                    len(average_topics)
                )


            with col4:

                st.metric(
                    "Weak Topics",
                    len(weak_topics)
                )


            # =============================================
            # TOPIC PERFORMANCE
            # =============================================

            st.divider()

            st.header(
                "📈 Topic Performance"
            )


            performance_data = []


            for topic, data in topic_scores.items():

                score = (
                    data["correct"] /
                    data["total"]
                ) * 100


                if score >= 75:

                    status = "🟢 Strong"


                elif score >= 60:

                    status = "🟡 Average"


                else:

                    status = "🔴 Needs Improvement"


                st.write(
                    f"**{topic}** — "
                    f"{score:.0f}% — "
                    f"{status}"
                )


                st.progress(
                    int(score)
                )


                performance_data.append({
                    "Topic": topic,
                    "Score": score
                })


            # =============================================
            # PERFORMANCE GRAPH
            # =============================================

            performance_df = pd.DataFrame(
                performance_data
            )


            st.subheader(
                "Performance Overview"
            )


            chart_data = performance_df.set_index(
                "Topic"
            )


            st.bar_chart(
                chart_data,
                y="Score",
                y_label="Score (%)"
            )


            # =============================================
            # AI RECOMMENDATIONS
            # =============================================

            st.divider()

            st.header(
                "🤖 MENTAL AI AI Recommendations"
            )


            roadmap = []


            if weak_topics:

                st.write(
                    "MENTAL AI identified the following "
                    "areas that need improvement."
                )


                # =========================================
                # PROCESS EACH WEAK TOPIC
                # =========================================

                for weak_topic, score in weak_topics:

                    weakness_level = (
                        100 - score
                    )


                    st.subheader(
                        f"🔴 Priority Topic: "
                        f"{weak_topic}"
                    )


                    st.write(
                        f"Current score: "
                        f"**{score:.0f}%**"
                    )


                    st.write(
                        f"Weakness level: "
                        f"**{weakness_level:.0f}%**"
                    )


                    # =====================================
                    # AI RECOMMENDATION
                    # =====================================

                    recommendations = recommend_topics(
                        weak_topic,
                        weakness_level,
                        courses,
                        vectors
                    )


                    if recommendations:

                        st.write(
                            "### 📚 Recommended "
                            "Learning Topics"
                        )


                        for recommendation in recommendations:

                            recommendation_score = (
                                recommendation[
                                    "recommendation_score"
                                ]
                            )


                            similarity = (
                                recommendation[
                                    "similarity"
                                ]
                            )


                            # =================================
                            # PRIORITY LABEL
                            # =================================

                            if recommendation_score >= 60:

                                priority_label = (
                                    "🔴 High Priority"
                                )


                            elif recommendation_score >= 30:

                                priority_label = (
                                    "🟡 Medium Priority"
                                )


                            else:

                                priority_label = (
                                    "🟢 Low Priority"
                                )


                            rec_col1, rec_col2 = st.columns(
                                [3, 1]
                            )


                            with rec_col1:

                                st.write(
                                    f"📚 **"
                                    f"{recommendation['topic']}"
                                    f"**"
                                )


                                st.caption(
                                    f"Difficulty: "
                                    f"{recommendation['difficulty']} "
                                    f"| Similarity: "
                                    f"{similarity}%"
                                )


                                st.caption(
                                    f"💡 MENTAL AI recommends this "
                                    f"topic because it is "
                                    f"{similarity}% similar "
                                    f"to your weak area."
                                )


                            with rec_col2:

                                st.write(
                                    priority_label
                                )


                                st.write(
                                    f"Score: "
                                    f"**{recommendation_score}/100**"
                                )


                            # =================================
                            # ADD TO ROADMAP
                            # =================================

                            recommended_topic = (
                                recommendation["topic"]
                            )


                            if (
                                recommended_topic
                                not in roadmap
                            ):

                                roadmap.append(
                                    recommended_topic
                                )


                            st.divider()


                    else:

                        st.info(
                            "No related learning topics "
                            "were found."
                        )


            else:

                st.success(
                    "🎉 Excellent performance!"
                )


                st.write(
                    "MENTAL AI did not identify any major "
                    "weak areas."
                )


            # =============================================
            # PERSONALIZED LEARNING ROADMAP
            # =============================================

            st.header(
                "🗺️ Your Personalized Learning Roadmap"
            )


            if weak_topics:

                st.write(
                    "MENTAL AI creates this roadmap based "
                    "on your weak areas and AI-generated "
                    "topic recommendations."
                )


                step = 1


                # =========================================
                # WEAK TOPICS FIRST
                # =========================================

                for weak_topic, score in weak_topics:

                    st.write(
                        f"### Step {step} — "
                        f"🔴 {weak_topic}"
                    )


                    st.write(
                        f"Your current score is "
                        f"**{score:.0f}%**."
                    )


                    st.write(
                        "Focus on understanding this "
                        "topic before moving to the "
                        "next stage."
                    )


                    step += 1


                    # =====================================
                    # RELATED RECOMMENDATIONS
                    # =====================================

                    recommendations = recommend_topics(
                        weak_topic,
                        100 - score,
                        courses,
                        vectors
                    )


                    weak_topic_names = [
                        topic
                        for topic, _ in weak_topics
                    ]


                    for recommendation in recommendations[:2]:

                        recommended_topic = (
                            recommendation["topic"]
                        )


                        if (
                            recommended_topic
                            not in weak_topic_names
                            and
                            recommended_topic
                            not in roadmap
                        ):

                            st.write(
                                f"### Step {step} — "
                                f"📚 {recommended_topic}"
                            )


                            st.write(
                                "Recommended because "
                                "MENTAL AI found a strong "
                                "relationship with your "
                                "learning gap."
                            )


                            step += 1


                st.info(
                    "💡 Follow the roadmap from top "
                    "to bottom to strengthen your "
                    "weak areas systematically."
                )


            else:

                st.success(
                    "🏆 You are ready for more "
                    "advanced learning material!"
                )


            # =============================================
            # PERFORMANCE SUMMARY
            # =============================================

            st.divider()

            st.header(
                "📝 Performance Summary"
            )


            if overall_score >= 75:

                st.success(
                    f"Excellent work, {name}! "
                    f"Your overall score is "
                    f"{overall_score:.0f}%."
                )


            elif overall_score >= 60:

                st.warning(
                    f"Good effort, {name}! "
                    f"Your overall score is "
                    f"{overall_score:.0f}%. "
                    f"Review your weaker topics "
                    f"to improve further."
                )


            else:

                st.error(
                    f"Keep practicing, {name}! "
                    f"Your overall score is "
                    f"{overall_score:.0f}%. "
                    f"Use the personalized roadmap "
                    f"to improve your performance."
                )



def build_pdf_learning_roadmap(
    weak_topics,
    average_topics,
    strong_topics,
    detected_concepts
):
    """
    Build a simple document-specific roadmap.

    The roadmap is based only on concepts detected from the
    student's uploaded PDF and the student's assessment
    performance. It does not pretend to know content that was
    not found in the uploaded document.
    """

    weak_names = [
        topic for topic, _ in weak_topics
    ]

    average_names = [
        topic for topic, _ in average_topics
    ]

    strong_names = [
        topic for topic, _ in strong_topics
    ]

    detected_lookup = {
        concept_key(concept): concept
        for concept in detected_concepts
    }

    roadmap = []

    # Stage 1: weakest concepts first.
    for topic, score in sorted(
        weak_topics,
        key=lambda item: item[1]
    ):
        roadmap.append({
            "stage": "Priority 1",
            "concept": topic,
            "score": score,
            "action": (
                "Review this concept in your uploaded PDF, "
                "then practice it again."
            )
        })

    # Stage 2: average concepts are the next targets.
    for topic, score in sorted(
        average_topics,
        key=lambda item: item[1]
    ):
        roadmap.append({
            "stage": "Priority 2",
            "concept": topic,
            "score": score,
            "action": (
                "Strengthen this concept before moving to "
                "more advanced material."
            )
        })

    # Stage 3: concepts not assessed yet.
    assessed_keys = {
        concept_key(topic)
        for topic in (
            weak_names +
            average_names +
            strong_names
        )
    }

    for concept in detected_concepts:
        key = concept_key(concept)

        if key not in assessed_keys:
            roadmap.append({
                "stage": "Next to Learn",
                "concept": concept,
                "score": None,
                "action": (
                    "This concept was detected in your PDF "
                    "but was not directly assessed. Study it "
                    "after reviewing your weaker areas."
                )
            })

    return roadmap


# =========================================================
# PERSONALIZED STUDY MATERIAL + ASSESSMENT
# =========================================================

if name and assessment_mode == "Personalized Assessment":

    st.divider()

    st.header(
        "📄 MENTAL AI Personalized Assessment"
    )

    st.write(
        "Upload your own study material. MENTAL AI will "
        "extract important concepts, generate questions "
        "from your document, evaluate your answers, and "
        "give you feedback."
    )

    st.info(
        "🧠 Hybrid AI: PDF text extraction + NLP keyword "
        "analysis + TF-IDF/cosine similarity."
    )

    uploaded_pdf = st.file_uploader(
        "📤 Upload your study material",
        type=["pdf"],
        key="personalized_pdf"
    )

    if uploaded_pdf:

        pdf_bytes = uploaded_pdf.getvalue()

        try:

            document_text = extract_pdf_text(
                pdf_bytes
            )

            if not document_text.strip():

                st.error(
                    "MENTAL AI could not extract text from this PDF. "
                    "Please use a text-based PDF."
                )

            else:

                chunks = create_text_chunks(
                    document_text
                )

                st.success(
                    f"✅ {uploaded_pdf.name} uploaded successfully."
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Document Pages",
                        len(
                            PdfReader(
                                io.BytesIO(pdf_bytes)
                            ).pages
                        )
                    )

                with col2:
                    st.metric(
                        "Text Characters",
                        len(document_text)
                    )

                with col3:
                    st.metric(
                        "Text Sections",
                        len(chunks)
                    )

                detected_concepts = extract_candidate_terms(
                    document_text,
                    limit=12
                )

                if detected_concepts:
                    st.subheader(
                        "🧠 Concepts Detected in Your Study Material"
                    )

                    st.write(
                        "MENTAL AI will use these concepts to build "
                        "the personalized assessment:"
                    )

                    st.write(
                        " • ".join(
                            concept.title()
                            for concept in detected_concepts
                        )
                    )

                # =========================================================
                # STUDY MATERIAL ANALYSIS
                # =========================================================

                topic_analysis = analyze_document_topics(
                    document_text,
                    chunks,
                    detected_concepts,
                    limit=8
                )

                st.divider()

                st.subheader(
                    "📚 Study Material Analysis"
                )

                st.write(
                    "MENTAL AI analyzes the uploaded document itself to "
                    "identify the main topics and show how prominently "
                    "each topic appears in the study material."
                )

                profile = get_document_content_profile(
                    document_text
                )

                profile_col1, profile_col2 = st.columns(2)

                with profile_col1:
                    st.markdown("### 🧩 Main Topics")

                    if topic_analysis:

                        for item in topic_analysis:

                            st.write(
                                f"**{item['topic']}** — "
                                f"{item['coverage']:.0f}% coverage"
                            )

                            st.progress(
                                min(int(round(item['coverage'])), 100)
                            )

                            st.caption(
                                f"Mentioned {item['mentions']} time(s) "
                                f"across {item['sections']} of "
                                f"{len(chunks)} extracted text sections."
                            )

                    else:

                        st.info(
                            "MENTAL AI could not identify enough reliable "
                            "topics from this document yet."
                        )

                with profile_col2:
                    st.markdown("### 📄 Document Profile")

                    st.metric(
                        "Main topics identified",
                        len(topic_analysis)
                    )

                    st.write(
                        f"**Content profile:** "
                        f"{profile['content_type']}"
                    )

                    st.caption(
                        "The content profile describes patterns found "
                        "in the PDF. It is not an AI judgment of the "
                        "academic difficulty of the material."
                    )

                    st.write(
                        "**How coverage is calculated:** the percentage "
                        "of extracted text sections that contain the topic "
                        "at least once."
                    )

                st.divider()

                st.subheader(
                    "🤖 Generate Your Personalized Quiz"
                )

                st.write(
                    "Questions will be generated from concepts "
                    "found in your uploaded PDF. Your existing "
                    "questions.csv demo assessment is not changed."
                )

                question_count = st.slider(
                    "Number of questions",
                    5,
                    10,
                    5,
                    key="personalized_question_count"
                )

                if st.button(
                    "🧠 Generate Personalized Quiz",
                    key="generate_personalized_quiz"
                ):

                    generated_questions = (
                        generate_personalized_questions(
                            document_text,
                            question_count
                        )
                    )

                    if len(generated_questions) < 5:

                        st.warning(
                            "MENTAL AI could not generate at least "
                            "5 questions from this PDF. Try a "
                            "longer text-based study document."
                        )

                        st.session_state.pop(
                            "personalized_questions",
                            None
                        )

                    else:

                        st.session_state[
                            "personalized_questions"
                        ] = generated_questions

                        st.success(
                            f"✅ Generated "
                            f"{len(generated_questions)} "
                            f"questions from your PDF."
                        )

                if (
                    "personalized_questions"
                    in st.session_state
                ):

                    generated_questions = (
                        st.session_state[
                            "personalized_questions"
                        ]
                    )

                    st.divider()

                    st.subheader(
                        "📝 Your Personalized Quiz"
                    )

                    with st.form(
                        "personalized_quiz_form"
                    ):

                        personalized_answers = []

                        for index, question in enumerate(
                            generated_questions
                        ):

                            st.markdown(
                                f"### Question {index + 1}"
                            )

                            st.write(
                                question["question"]
                            )

                            options = [
                                "Select an answer"
                            ] + question["options"]

                            answer = st.radio(
                                "Choose your answer:",
                                options,
                                key=(
                                    "personalized_answer_"
                                    f"{index}"
                                )
                            )

                            personalized_answers.append(
                                answer
                            )

                        submitted = (
                            st.form_submit_button(
                                "🚀 Submit Personalized Assessment"
                            )
                        )

                    if submitted:

                        unanswered = [
                            index + 1
                            for index, answer
                            in enumerate(
                                personalized_answers
                            )
                            if answer == "Select an answer"
                        ]

                        if unanswered:

                            st.error(
                                "Please answer all questions "
                                "before submitting."
                            )

                            st.write(
                                "Unanswered questions: "
                                + ", ".join(
                                    map(str, unanswered)
                                )
                            )

                        else:

                            topic_scores = {}

                            for index, question in enumerate(
                                generated_questions
                            ):

                                topic = question["topic"]

                                if topic not in topic_scores:

                                    topic_scores[topic] = {
                                        "correct": 0,
                                        "total": 0
                                    }

                                topic_scores[topic]["total"] += 1

                                if (
                                    personalized_answers[index]
                                    == question["correct_answer"]
                                ):

                                    topic_scores[topic][
                                        "correct"
                                    ] += 1

                            total_correct = sum(
                                data["correct"]
                                for data in topic_scores.values()
                            )

                            total_questions = sum(
                                data["total"]
                                for data in topic_scores.values()
                            )

                            overall_score = (
                                total_correct
                                / total_questions
                            ) * 100

                            strong_topics = []
                            average_topics = []
                            weak_topics = []

                            for topic, data in (
                                topic_scores.items()
                            ):

                                score = (
                                    data["correct"]
                                    / data["total"]
                                ) * 100

                                if score >= 75:
                                    strong_topics.append(
                                        (topic, score)
                                    )

                                elif score >= 60:
                                    average_topics.append(
                                        (topic, score)
                                    )

                                else:
                                    weak_topics.append(
                                        (topic, score)
                                    )

                            st.divider()

                            st.header(
                                "📊 Your Personalized Dashboard"
                            )

                            c1, c2, c3, c4 = st.columns(4)

                            with c1:
                                st.metric(
                                    "Overall Score",
                                    f"{overall_score:.0f}%"
                                )

                            with c2:
                                st.metric(
                                    "Strong Concepts",
                                    len(strong_topics)
                                )

                            with c3:
                                st.metric(
                                    "Average Concepts",
                                    len(average_topics)
                                )

                            with c4:
                                st.metric(
                                    "Weak Concepts",
                                    len(weak_topics)
                                )

                            st.divider()

                            st.subheader(
                                "📈 Concept Performance"
                            )

                            performance_data = []

                            for topic, data in (
                                topic_scores.items()
                            ):

                                score = (
                                    data["correct"]
                                    / data["total"]
                                ) * 100

                                if score >= 75:
                                    status = "🟢 Strong"

                                elif score >= 60:
                                    status = "🟡 Average"

                                else:
                                    status = (
                                        "🔴 Needs Improvement"
                                    )

                                st.write(
                                    f"**{topic}** — "
                                    f"{score:.0f}% — "
                                    f"{status}"
                                )

                                st.progress(
                                    int(score)
                                )

                                performance_data.append({
                                    "Topic": topic,
                                    "Score": score
                                })

                            performance_df = pd.DataFrame(
                                performance_data
                            )

                            st.subheader(
                                "Performance Overview"
                            )

                            st.bar_chart(
                                performance_df.set_index(
                                    "Topic"
                                ),
                                y="Score",
                                y_label="Score (%)"
                            )

                            st.divider()

                            st.header(
                                "🤖 MENTAL AI AI Feedback"
                            )

                            if weak_topics:

                                st.warning(
                                    "MENTAL AI identified concepts "
                                    "that need more practice."
                                )

                                for weak_topic, score in sorted(
                                    weak_topics,
                                    key=lambda item: item[1]
                                ):

                                    st.write(
                                        f"### 🔴 {weak_topic}"
                                    )

                                    st.write(
                                        f"Current score: "
                                        f"**{score:.0f}%**"
                                    )

                                    st.write(
                                        "📖 Go back to the uploaded "
                                        "PDF and review the sections "
                                        "related to this concept."
                                    )

                            if average_topics:

                                st.info(
                                    "🟡 Average concepts should be "
                                    "strengthened after your weak areas."
                                )

                                for topic, score in sorted(
                                    average_topics,
                                    key=lambda item: item[1]
                                ):

                                    st.write(
                                        f"**{topic}** — "
                                        f"{score:.0f}%"
                                    )

                            if not weak_topics and not average_topics:

                                st.success(
                                    "🎉 Excellent performance! "
                                    "MENTAL AI did not identify any "
                                    "major learning gaps."
                                )

                            # =============================================
                            # DOCUMENT-SPECIFIC LEARNING ROADMAP
                            # =============================================

                            st.divider()

                            st.header(
                                "🗺️ Your Personalized Learning Roadmap"
                            )

                            st.write(
                                "This roadmap is created from the concepts "
                                "found in your uploaded PDF and your "
                                "assessment performance."
                            )

                            roadmap = build_pdf_learning_roadmap(
                                weak_topics,
                                average_topics,
                                strong_topics,
                                detected_concepts
                            )

                            if roadmap:

                                for step_number, item in enumerate(
                                    roadmap,
                                    start=1
                                ):

                                    if item["score"] is None:

                                        st.write(
                                            f"### Step {step_number} — "
                                            f"📚 {item['concept']}"
                                        )

                                        st.caption(
                                            item["stage"]
                                        )

                                    else:

                                        if item["score"] < 60:
                                            icon = "🔴"
                                        else:
                                            icon = "🟡"

                                        st.write(
                                            f"### Step {step_number} — "
                                            f"{icon} {item['concept']}"
                                        )

                                        st.write(
                                            f"Assessment score: "
                                            f"**{item['score']:.0f}%**"
                                        )

                                        st.caption(
                                            item["stage"]
                                        )

                                    st.write(
                                        item["action"]
                                    )

                                    if step_number < len(roadmap):
                                        st.divider()

                                st.info(
                                    "💡 Recommended order: first review "
                                    "your weakest concepts, then strengthen "
                                    "average concepts, and finally explore "
                                    "detected concepts that were not assessed."
                                )

                            else:

                                st.info(
                                    "MENTAL AI could not build a roadmap from "
                                    "the detected concepts. Try uploading "
                                    "a longer text-based study document."
                                )

                st.divider()

                st.subheader(
                    "🔎 Ask a Question About Your PDF"
                )

                user_question = st.text_input(
                    "Enter your question",
                    placeholder=(
                        "Example: What is a Process "
                        "Control Block?"
                    ),
                    key="personal_pdf_question"
                )

                search_button = st.button(
                    "🔍 Find Answer in Document",
                    key="personal_pdf_search"
                )

                if search_button:

                    if not user_question.strip():

                        st.warning(
                            "Please enter a question first."
                        )

                    else:

                        results = search_document(
                            user_question,
                            chunks,
                            number_of_results=3
                        )

                        if not results:

                            st.warning(
                                "MENTAL AI could not find relevant "
                                "information in your document."
                            )

                        else:

                            st.success(
                                "MENTAL AI found relevant information "
                                "in your study material."
                            )

                            for index, result in enumerate(
                                results
                            ):

                                st.markdown(
                                    f"### Result {index + 1}"
                                )

                                st.caption(
                                    f"Relevance: "
                                    f"{result['score']:.1f}%"
                                )

                                st.write(
                                    result["text"]
                                )

        except Exception as error:

            st.error(
                "MENTAL AI could not process this PDF."
            )

            st.write(
                f"Technical details: {error}"
            )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎓 MENTAL AI — AI-Powered Personalized Learning "
    "Recommendation System"
)