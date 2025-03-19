import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from itertools import product
import streamlit as st

# Preprocessing and Vectorization
def preprocess_and_vectorize(data, max_features=5000):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(data)
    return tfidf_matrix, vectorizer

# Apply LSA for dimensionality reduction
def apply_lsa(tfidf_matrix, n_components=50):
    n_components = min(n_components, tfidf_matrix.shape[1])  # Ensure n_components <= n_features
    svd = TruncatedSVD(n_components=n_components)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    return lsa_matrix, svd

# Calculate Location Similarity (simple example: exact match)
def calculate_location_similarity(candidate_location, job_location):
    return 1 if candidate_location == job_location else 0

# Build MultiDiGraph and calculate PageRank
def build_multidigraph_and_pagerank(title_lsa, resume_lsa, job_lsa, candidate_location, job_locations,
                                    similarity_threshold, expertise_weight, resume_weight, location_weight):
    graph = nx.MultiDiGraph()

    # Calculate expertise and resume similarity
    expertise_similarity = cosine_similarity(title_lsa, job_lsa).flatten()
    resume_similarity = cosine_similarity(resume_lsa, job_lsa).flatten()

    # Add nodes and edges
    graph.add_node('resume', type='resume')
    for i, (exp_score, res_score) in enumerate(zip(expertise_similarity, resume_similarity)):
        graph.add_node(f'job_{i}', type='job')

        if exp_score > similarity_threshold:
            graph.add_edge('resume', f'job_{i}', weight=expertise_weight * exp_score, type='expertise')
            graph.add_edge(f'job_{i}', 'resume', weight=expertise_weight * exp_score, type='expertise')

        if res_score > similarity_threshold:
            graph.add_edge('resume', f'job_{i}', weight=resume_weight * res_score, type='resume')
            graph.add_edge(f'job_{i}', 'resume', weight=resume_weight * res_score, type='resume')

        # Add location similarity
        loc_score = calculate_location_similarity(candidate_location, job_locations[i])
        if loc_score > 0:
            graph.add_edge('resume', f'job_{i}', weight=location_weight * loc_score, type='location')
            graph.add_edge(f'job_{i}', 'resume', weight=location_weight * loc_score, type='location')

    # Combine weights for PageRank calculation
    combined_graph = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        if combined_graph.has_edge(u, v):
            combined_graph[u][v]['weight'] += weight
        else:
            combined_graph.add_edge(u, v, weight=weight)

    # Calculate PageRank
    pagerank_scores = nx.pagerank(combined_graph, weight='weight')
    return pagerank_scores

# Calculate recommendations for all resumes for a parameter combination
def calculate_pagerank_for_params(resume_dataset, job_titles, job_descriptions, job_tags, job_locations,
                                  similarity_threshold, expertise_weight, resume_weight, location_weight, lsa_components=50):
    total_pagerank_score = 0

    for _, row in resume_dataset.iterrows():
        # Extract candidate details
        title = row['Category']
        resume = row['Resume']
        candidate_location = row['Location']

        # Combine job data
        enriched_jobs = [
            f"{job_title} {description} {' '.join(eval(tags))}"
            for job_title, description, tags in zip(job_titles, job_descriptions, job_tags)
        ]
        all_text = [title] + [resume] + enriched_jobs

        # Preprocess and vectorize
        tfidf_matrix, _ = preprocess_and_vectorize(all_text)

        # Apply LSA
        lsa_matrix, _ = apply_lsa(tfidf_matrix, lsa_components)

        # Split matrices for title, resume, and jobs
        title_lsa = lsa_matrix[0].reshape(1, -1)
        resume_lsa = lsa_matrix[1].reshape(1, -1)
        job_lsa = lsa_matrix[2:]

        # Build MultiDiGraph and calculate PageRank
        pagerank_scores = build_multidigraph_and_pagerank(
            title_lsa, resume_lsa, job_lsa, candidate_location, job_locations,
            similarity_threshold, expertise_weight, resume_weight, location_weight
        )

        # Sum PageRank scores for job nodes
        job_scores = [pagerank_scores[node] for node in pagerank_scores if node.startswith('job_')]
        total_pagerank_score += sum(job_scores)

    return total_pagerank_score

# Find the best parameters
def find_best_hyperparameters(resume_dataset, job_titles, job_descriptions, job_tags, job_locations,
                              thresholds, expertise_weights, resume_weights, location_weights, lsa_components=50):
    best_params = None
    best_score = -np.inf
    results = []

    # Iterate through all combinations of hyperparameters
    for similarity_threshold, expertise_weight, resume_weight, location_weight in product(thresholds, expertise_weights, resume_weights, location_weights):
        print(f"Evaluating similarity_threshold={similarity_threshold}, expertise_weight={expertise_weight}, resume_weight={resume_weight}, location_weight={location_weight}...")
        score = calculate_pagerank_for_params(
            resume_dataset, job_titles, job_descriptions, job_tags, job_locations,
            similarity_threshold, expertise_weight, resume_weight, location_weight, lsa_components
        )
        results.append((similarity_threshold, expertise_weight, resume_weight, location_weight, score))

        if score > best_score:
            best_score = score
            best_params = (similarity_threshold, expertise_weight, resume_weight, location_weight)

    return best_params, results

def recommend_top_jobs_for_candidate(row, job_company_name, job_titles, job_descriptions, job_tags, job_locations,
                                     best_params, lsa_components=50, top_k=10):
    # Extract candidate details
    title = row['Category']
    resume = row['Resume']
    candidate_location = row['Location']

    # Best parameters
    similarity_threshold, expertise_weight, resume_weight, location_weight = best_params

    # Combine job data
    enriched_jobs = [
        f"{job_title} {description} {' '.join(eval(tags))}"
        for job_title, description, tags in zip(job_titles, job_descriptions, job_tags)
    ]
    all_text = [title] + [resume] + enriched_jobs

    # Preprocess and vectorize
    tfidf_matrix, _ = preprocess_and_vectorize(all_text)

    # Apply LSA
    lsa_matrix, _ = apply_lsa(tfidf_matrix, lsa_components)

    # Split matrices for title, resume, and jobs
    title_lsa = lsa_matrix[0].reshape(1, -1)
    resume_lsa = lsa_matrix[1].reshape(1, -1)
    job_lsa = lsa_matrix[2:]

    # Build MultiDiGraph and calculate PageRank
    pagerank_scores = build_multidigraph_and_pagerank(
        title_lsa, resume_lsa, job_lsa, candidate_location, job_locations,
        similarity_threshold, expertise_weight, resume_weight, location_weight
    )

    # Extract job scores and recommend top K
    job_scores = [(node, pagerank_scores[node]) for node in pagerank_scores if node.startswith('job_')]
    sorted_jobs = sorted(job_scores, key=lambda x: x[1], reverse=True)[:top_k]

    # Map job IDs back to titles, descriptions, and tags
    recommendations = [
        (
            job_company_name[int(node.split('_')[1])],
            job_titles[int(node.split('_')[1])],
            job_descriptions[int(node.split('_')[1])],
            job_locations[int(node.split('_')[1])],
            job_tags[int(node.split('_')[1])],
            score
        )
        for node, score in sorted_jobs
    ]
    return recommendations

# Recommend jobs for all candidates
def recommend_jobs_for_all_candidates(resume_dataset, job_company_name, job_titles, job_descriptions, job_tags, job_locations, best_params, lsa_components=50, top_k=10):
    all_recommendations = []

    for _, row in resume_dataset.iterrows():
        recommendations = recommend_top_jobs_for_candidate(
            row, job_company_name, job_titles, job_descriptions, job_tags, job_locations,
            best_params, lsa_components, top_k
        )
        all_recommendations.append({
            "candidate_title": row['Category'],
            "candidate_location": row['Location'],
            "recommendations": recommendations
        })

    return all_recommendations

# Streamlit Web App
st.title("Job Recommendation System")

# Sidebar for candidate input
st.sidebar.header("Candidate Input")
candidate_category = st.sidebar.text_input("Candidate Category", "Data Scientist")
candidate_resume = st.sidebar.text_area("Candidate Resume", "Experienced in Python, ML, and data analysis...")
candidate_location = st.sidebar.text_input("Candidate Location", "Ho Chi Minh")

resume_dataset = pd.DataFrame([
{
    "Category": candidate_category, 
    "Resume": candidate_resume, 
    "Location": candidate_location}
])

# Load job dataset
jobs_data = pd.read_csv('data/job_info_location.csv')
job_company_name = jobs_data['company_name'].astype(str).tolist()
job_titles = jobs_data['job_title'].astype(str).tolist()
job_descriptions = jobs_data['description'].astype(str).tolist()
job_tags = jobs_data['tag_list'].astype(str).tolist()
job_locations = jobs_data['location_new'].astype(str).tolist()

# Best parameters from hyperparameter tuning
best_params = (0.5, 0.6, 0.3, 0.1)  # Replace with the actual best parameters

if st.button("Recommend Jobs"):
    # Run recommendations
    # Create a DataFrame to store recommendations
    recommendations_data = []
    recommendations = recommend_jobs_for_all_candidates(
        resume_dataset, job_company_name, job_titles, job_descriptions, job_tags, job_locations,
        best_params, lsa_components=50, top_k=10
    )

    # Display recommendations
    for rec in recommendations:
        for job in rec['recommendations']:
            # st.write(f"  - Job Title: {job[0]}, Location: {(job[2])}, PageRank Score: {job[4]:.4f}")
            recommendations_data.append({
                "Company Name": job[0],
                "Job Title": job[1],
                "Job Location": job[3],
                # "Tags": job[3],
                "PageRank Score": job[5]
            })

    # Convert to DataFrame
    recommendations_df = pd.DataFrame(recommendations_data)

    # Display the recommendations DataFrame in Streamlit
    st.write("### Top 10 Job Recommendations for All Candidates")
    st.dataframe(recommendations_df)
