import os
from ingestion import load_pdf
from preprocessing import clean_text
from chunking import chunk_text
from vector_store import create_collection, insert_chunks, get_embedding
from retriever import retrieve
from rag_pipeline import generate_answer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ============= Evaluation Metrics Functions =============

def is_relevant(context, ground_truth, threshold=0.5):
    """Determine if a context is relevant to ground truth using semantic similarity (embeddings)"""
    try:
        # Use OpenAI embeddings for semantic similarity
        context_vector = np.array(get_embedding(context))
        truth_vector = np.array(get_embedding(ground_truth))
        
        # Calculate cosine similarity
        similarity = cosine_similarity([context_vector], [truth_vector])[0][0]
        return similarity >= threshold
    except:
        return False

def precision_at_k(retrieved_contexts, ground_truth, k=3):
    """P@k: Percentage of top-k results that are relevant"""
    relevant_count = sum(1 for ctx in retrieved_contexts[:k] if is_relevant(ctx, ground_truth))
    return relevant_count / min(k, len(retrieved_contexts))

def recall_at_k(retrieved_contexts, ground_truth, k=3):
    """R@k: Percentage of relevant items in top-k (assuming 1 relevant item per query)"""
    for i, ctx in enumerate(retrieved_contexts[:k]):
        if is_relevant(ctx, ground_truth):
            return 1.0
    return 0.0

def mean_reciprocal_rank(retrieved_contexts, ground_truth):
    """MRR: Inverse of the rank of the first relevant item"""
    for i, ctx in enumerate(retrieved_contexts):
        if is_relevant(ctx, ground_truth):
            return 1.0 / (i + 1)
    return 0.0

def average_precision(retrieved_contexts, ground_truth):
    """AP: Average precision across all positions where relevant items appear"""
    precisions = []
    relevant_count = 0
    for i, ctx in enumerate(retrieved_contexts):
        if is_relevant(ctx, ground_truth):
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    return sum(precisions) / len(precisions) if precisions else 0.0

def ndcg(retrieved_contexts, ground_truth, k=3):
    """NDCG: Normalized Discounted Cumulative Gain"""
    # Calculate DCG
    dcg = 0.0
    for i, ctx in enumerate(retrieved_contexts[:k]):
        relevance = 1.0 if is_relevant(ctx, ground_truth) else 0.0
        dcg += relevance / np.log2(i + 2)  # i+2 because ranking starts at 1
    
    # Calculate IDCG (ideal DCG - best possible ranking)
    idcg = 1.0 / np.log2(2)  # At least one relevant item at position 1
    
    return dcg / idcg if idcg > 0 else 0.0

# ============= Main Evaluation =============

def main():
    # Get the directory of this script and construct the data path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "data", "hr_policy.pdf")
    
    # Load and prepare the RAG system
    text = load_pdf(pdf_path)
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)
    create_collection()
    insert_chunks(chunks)

    # Define test questions and ground truths (updated to match actual PDF content)
    test_questions = [
        "What is the maternity leave policy?",
        "How many days of annual leave are employees entitled to?",
        "What is the procedure for reporting harassment?"
    ]

    ground_truths = [
        "Maternity leave will be provided on full pay along with allowances. Employees covered under ESIC will have maternity leave provided by ESIC.",
        "Employees at ABC Corp are entitled to 18 Earned Leaves in a calendar year. Pro-rata basis for mid-year joins.",
        "Whistleblower policies and confidential reporting procedures. Equal right policies and electronic usage policy."
    ]

    # Collect data for evaluation
    print("===== Running RAG Evaluation =====\n")
    
    all_metrics = {
        'P@3': [],
        'R@3': [],
        'MRR': [],
        'AP': [],
        'NDCG@3': []
    }
    
    for q, gt in zip(test_questions, ground_truths):
        retrieved_contexts = retrieve(q)
        answer = generate_answer(q, retrieved_contexts)

        print(f"\nQuestion: {q}")
        print(f"Ground Truth: {gt}")
        print(f"Generated Answer: {answer}")
        print("\n--- Retrieved Contexts ---")
        for i, ctx in enumerate(retrieved_contexts, 1):
            relevance = "✓ RELEVANT" if is_relevant(ctx, gt) else "✗ NOT RELEVANT"
            print(f"{i}. {ctx[:100]}... [{relevance}]")
        
        # Calculate metrics
        p_at_3 = precision_at_k(retrieved_contexts, gt, k=3)
        r_at_3 = recall_at_k(retrieved_contexts, gt, k=3)
        mrr = mean_reciprocal_rank(retrieved_contexts, gt)
        ap = average_precision(retrieved_contexts, gt)
        ndcg_3 = ndcg(retrieved_contexts, gt, k=3)
        
        # Store metrics
        all_metrics['P@3'].append(p_at_3)
        all_metrics['R@3'].append(r_at_3)
        all_metrics['MRR'].append(mrr)
        all_metrics['AP'].append(ap)
        all_metrics['NDCG@3'].append(ndcg_3)
        
        print("\n--- Metrics ---")
        print(f"P@3 (Precision@3):    {p_at_3:.4f}")
        print(f"R@3 (Recall@3):       {r_at_3:.4f}")
        print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
        print(f"AP (Average Precision): {ap:.4f}")
        print(f"NDCG@3:               {ndcg_3:.4f}")
        print("-" * 80)
    
    # Print aggregate metrics
    print("\n===== AGGREGATE METRICS =====")
    print(f"Mean P@3:    {np.mean(all_metrics['P@3']):.4f}")
    print(f"Mean R@3:    {np.mean(all_metrics['R@3']):.4f}")
    print(f"Mean MRR:    {np.mean(all_metrics['MRR']):.4f}")
    print(f"Mean AP:     {np.mean(all_metrics['AP']):.4f}")
    print(f"Mean NDCG@3: {np.mean(all_metrics['NDCG@3']):.4f}")

if __name__ == "__main__":
    main()
