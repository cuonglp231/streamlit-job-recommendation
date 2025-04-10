import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from itertools import product
import streamlit as st
import random
import ast
import matplotlib.pyplot as plt

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
                              thresholds, expertise_weights, location_weights, lsa_components=50):
    num_iterations = 10  # Số lần chạy thử nghiệm ngẫu nhiên
    best_params = None
    best_score = -np.inf
    results = []

    # Iterate through all combinations of hyperparameters
    for _ in range(num_iterations):
        similarity_threshold = random.choice(thresholds)
        expertise_weight = random.choice(expertise_weights)
        location_weight = random.choice(location_weights)
        resume_weight = 1 - expertise_weight

        print(f"Evaluating similarity_threshold={similarity_threshold}, expertise_weight={expertise_weight}, resume_weight={resume_weight}, location_weight={location_weight}...")
        
        score = calculate_pagerank_for_params(
            resume_dataset, job_titles, job_descriptions, job_tags, job_locations,
            similarity_threshold, expertise_weight, resume_weight, location_weight, lsa_components
        )

        print(f"Total PageRank Score: {score}\n")
        
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
            "actual_applied_jobs": row['Applied_Jobs'],
            "recommendations": recommendations
        })

    return all_recommendations

def evaluate_recommendation_precision_recall(actual_applied_jobs, recommended_jobs, top_k):
    """
    Tính Precision và Recall cho mỗi ứng viên X
    
    :param actual_applied_jobs: Danh sách job mà ứng viên thực tế đã apply (set)
    :param recommended_jobs: Danh sách job mà hệ thống recommend (list)
    :param top_k: Số lượng job top K được lấy từ recommended_jobs
    
    :return: (precision, recall)
    """
    recommended_top_k = set(recommended_jobs[:top_k])  # Chỉ lấy K job đầu tiên
    relevant_jobs = actual_applied_jobs & recommended_top_k  # Giao giữa R và top K
    
    precision = len(relevant_jobs) / top_k if top_k > 0 else 0
    recall = len(relevant_jobs) / len(actual_applied_jobs) if len(actual_applied_jobs) > 0 else 0

    return precision, recall

# Example usage
if __name__ == "__main__":
    # Load resume dataset (title, resume, and location)
    resume_dataset = pd.read_csv('data/cvdata/ResumeDataSet_ListAppliedJobs.csv')
    # resume_dataset = pd.DataFrame([
    #     {"Category": "data science", 
    #      "Resume": "expertise data and quantitative analysis decision analytics predictive modeling data-driven personalization kpi dashboards big data queries and interpretation data mining and visualization tools machine learning algorithms business intelligence (bi) research, reports and forecasts education details pgp in data science mumbai, maharashtra aegis school of data science & business b.e. in electronics & communication electronics & communication indore, madhya pradesh ies ips academy data scientist data scientist with pr canada skill details algorithms- exprience - months bi- exprience - months business intelligence- exprience - months machine learning- exprience - months visualization- exprience - months spark- exprience - months python- exprience - months tableau- exprience - months data analysis- exprience - monthscompany details company - aegis school of data science & business description - mostly working on industry project for providing solution along with teaching appointments: teach undergraduate and graduate-level courses in spark and machine learning as an adjunct faculty member at aegis school of data science, mumbai ( to present) company - aegis school of data & business description - data science intern, nov to jan furnish executive leadership team with insights, analytics, reports and recommendations enabling effective strategic planning across all business units, distribution channels and product lines. chat bot using aws lex and tensor flow python the goal of project creates a chat bot for an academic institution or university to handle queries related courses offered by that institute. the objective of this task is to reduce human efforts as well as reduce man made errors. even by this companies handle their client x. in this case companies are academic institutions and clients are participants or students. web scraping using selenium web driver python the task is to scrap the data from the online messaging portal in a text format and have to find the pattern form it. data visualization and data insights hadoop eco system, hive, pyspark, qliksense the goal of this project is to build a business solutions to a internet service provider company, like handling data which is generated per day basis, for that we have to visualize that data and find the usage pattern form it and have a generate a reports. image based fraud detection microsoft face api, pyspark, open cv the main goal of project is recognize similarity for a face to given database images. face recognition is the recognizing a special face from set of different faces. face is extracted and then compared with the database image if that image recognized then the person already applied for loan from somewhere else and now hiding his or her identity, this is how we are going to prevent the frauds in the initial stage itself. churn analysis for internet service provider r, python, machine learning, hadoop the objective is to identify the customer who is likely to churn in a given period of time; we have to pretend the customer giving incentive offers. sentiment analysis python, nlp, apache spark service in ibm bluemix. this project is highly emphasis on tweets from twitter data were taken for mobile networks service provider to do a sentiment analysis and analyze whether the expressed opinion was positive, negative or neutral, capture the emotions of the tweets and comparative analysis. quantifiable results: mentored - data science enthusiast each year that have all since gone on to graduate school in data science and business analytics. reviewed and evaluated - research papers on data science for one of the largest data science conference called data science congress by aegis school of business mumbai. heading a solution providing organization called data science delivered into aegis school of data science mumbai and managed - live projects using data science techniques. working for some social cause with the help of data science for social goods committee, where our team developed a product called let's find a missing child for helping society. company - ibm india pvt ltd description - mostly worked on blumix and ibm watson for data science.", 
    #      "Location": "ho chi minh"}
    # ])

    # actual_applied_jobs = {'data analytics engineer (hcl x anz bank)', 'senior data engineer', 'middle/senior data analyst (python, sql)', 'data analytics engineer (hcl x anz bank)', 'senior data engineer', 'middle/senior data analyst (python, sql)', 'senior data analyst', 'data engineer (sql, data analyst, agile)', 'head of data science,zalopay', 'senior data engineer', 'middle/senior data analyst (python, sql)', 'senior data engineer', 'data analytics engineer (hcl x anz bank)'}  # R jobs (ứng viên đã apply)

    # Load job dataset
    jobs_data = pd.read_csv('data/job_info_location.csv')
    job_company_name = jobs_data['company_name'].astype(str).tolist()
    job_titles = jobs_data['job_title'].astype(str).tolist()
    job_descriptions = jobs_data['description'].astype(str).tolist()
    job_tags = jobs_data['tag_list'].astype(str).tolist()
    job_locations = jobs_data['location_new'].astype(str).tolist()

    # Best parameters from hyperparameter tuning
    best_params = (0.5, 0.6, 0.3, 0.1)  # Replace with the actual best parameters (similarity_threshold, expertise_weight, resume_weight, location_weight)

    max_k = 20
    precision_at_k = []
    recall_at_k = []

    for top_k in range(1, max_k + 1):
        # Get top 10 job recommendations for all candidates
        recommendations = recommend_jobs_for_all_candidates(
            resume_dataset, job_company_name, job_titles, job_descriptions, job_tags, job_locations,
            best_params, lsa_components=50, top_k=top_k
        )

        list_precision = []
        list_recall = []

        # Display recommendations
        for rec in recommendations:
            recommended_jobs = []  # Kết quả hệ thống gợi ý (hệ thống đã recommend cho ứng viên)
            actual_applied_jobs = set(ast.literal_eval(rec['actual_applied_jobs']))  # R jobs (ứng viên đã apply)
            # print(f"Candidate Title: {rec['candidate_title']} (Location: {rec['candidate_location']})")
            # print(f"Top {top_k} Job Recommendations:")
            for job in rec['recommendations']:
                # print(f"  - Company Name: {job[0]}, Job Title: {job[1].lower()}, Location: {(job[3])}, PageRank Score: {job[5]:.4f}")
                recommended_jobs.append(job[1].lower())
            
            precision, recall = evaluate_recommendation_precision_recall(actual_applied_jobs, recommended_jobs, top_k)
            list_precision.append(precision)
            list_recall.append(recall)
            # print(f"Precision: {precision:.2f}")
            # print(f"Recall: {recall:.2f}")
            # print("\n")

        # Calculate average precision and recall
        avg_precision = np.mean(list_precision)
        avg_recall = np.mean(list_recall)
        print(f"Average Precision: {avg_precision:.2f}")
        print(f"Average Recall: {avg_recall:.2f}")

        precision_at_k.append(avg_precision)
        recall_at_k.append(avg_recall)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), precision_at_k, marker='o', label='Precision')
    plt.plot(range(1, max_k + 1), recall_at_k, marker='s', label='Recall')
    plt.xlabel('K (Top-K Recommendations)')
    plt.ylabel('Score')
    plt.title('Precision and Recall at Different Top-K Values')
    plt.legend()
    plt.grid(True)

    # 👉 Scale trục X theo số nguyên
    plt.xticks(range(1, max_k + 1))  
    
    plt.tight_layout()
    plt.show()
