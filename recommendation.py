import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


def load_courses():

    courses = pd.read_csv(
        "data/courses.csv"
    )

    return courses


def create_vectors(courses):

    text_data = (
        courses["topic"]
        + " "
        + courses["description"]
        + " "
        + courses["difficulty"]
    )


    vectorizer = TfidfVectorizer(
        stop_words="english"
    )


    vectors = vectorizer.fit_transform(
        text_data
    )


    return vectorizer, vectors


def recommend_topics(
    weak_topic,
    weakness_level,
    courses,
    vectors
):

    topic_index = courses[
        courses["topic"] == weak_topic
    ].index


    if len(topic_index) == 0:

        return []


    topic_index = topic_index[0]


    similarity_scores = cosine_similarity(
        vectors[topic_index],
        vectors
    )[0]


    recommendations = []


    for index, similarity in enumerate(
        similarity_scores
    ):

        if index == topic_index:

            continue


        recommendation_score = (
            similarity * 100
        )


        recommendation_score += (
            weakness_level * 0.2
        )


        if recommendation_score > 100:

            recommendation_score = 100


        recommendations.append({

            "topic":
                courses.iloc[index]["topic"],

            "description":
                courses.iloc[index]["description"],

            "difficulty":
                courses.iloc[index]["difficulty"],

            "similarity":
                round(
                    similarity * 100,
                    1
                ),

            "recommendation_score":
                round(
                    recommendation_score
                )
        })


    recommendations.sort(
        key=lambda x:
            x["recommendation_score"],
        reverse=True
    )


    return recommendations[:3]