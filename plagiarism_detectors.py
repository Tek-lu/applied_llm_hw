# %%
import os
import pickle
import re
import numpy as np
from typing import Dict, Any

try:
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

_INDEXES_DIR = "indexes"       # folder containing embeddings.npy & clean_texts.pkl
_EMBEDDING_MODEL = None
_EMBEDDINGS = None
_REFERENCE_TEXTS = None



# %%
def clean_code(code):
    """
    Cleans Python code by:
    - Removing single-line comments (# ...)
    - Removing multi-line comments ('''  ''' or """  """)
    - Removing extra spaces and newlines
    """
    # Remove multi-line comments ("""...""" or '''...''')
    code = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', code)

    # Remove single-line comments
    code = re.sub(r'#.*', '', code)

    # Remove extra spaces/newlines
    code = re.sub(r'\s+', ' ', code).strip()

    return code


# %% [markdown]
# Loads: embeddings.npy (numpy) loads only once and caches it, clean_texts.pkl (list[str])   
#     

# %%
def load_file_level_indexes(indexes_dir: str = _INDEXES_DIR,
                            model_name: str = "all-MiniLM-L6-v2"):
    
    global _EMBEDDING_MODEL, _EMBEDDINGS, _REFERENCE_TEXTS

    # Load embeddings
    emb_path = os.path.join(indexes_dir, "embeddings.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"{emb_path} not found. Run 01_indexing.ipynb first.")
    _EMBEDDINGS = np.load(emb_path)

    # Load cleaned file texts
    ref_texts_path = os.path.join(indexes_dir, "clean_texts.pkl")
    if not os.path.exists(ref_texts_path):
        raise FileNotFoundError(f"{ref_texts_path} not found. Expected during file-level chunking.")
    with open(ref_texts_path, "rb") as f:
        _REFERENCE_TEXTS = pickle.load(f)

    # Basic sanity check
    if _EMBEDDINGS.shape[0] != len(_REFERENCE_TEXTS):
        raise ValueError("Mismatch between embeddings count and reference texts count. Re-indexing needed.")

    # Load model (only once)
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(model_name)

    print(f"Loaded {len(_REFERENCE_TEXTS)} file-level code entries.")
    return _EMBEDDINGS, _REFERENCE_TEXTS, _EMBEDDING_MODEL


# %% [markdown]
# Computes cosine similarity between vector (1,D) and matrix (N,D) entries.

# %%
def cosine_sim(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    if _HAS_SKLEARN:
        return cosine_similarity(a, B)[0]  # returns 1D array
    # numpy fallback
    a_norm = np.linalg.norm(a)
    B_norm = np.linalg.norm(B, axis=1)
    denom = np.where(a_norm * B_norm == 0, 1e-12, a_norm * B_norm)
    return (B @ a[0]) / denom

# %% [markdown]
# Pure Embedding Search for FILE-LEVEL chunking.
#     Returns:
#       - is_plagiarized (bool)
#       - max_similarity (float)
#       - top_matches (list with code snippets + similarity)

# %%
def detect_embedding(test_code: str,
                     top_k: int = 5,
                     threshold: float = 0.75,
                     model_name: str = "all-MiniLM-L6-v2",
                     indexes_dir: str = _INDEXES_DIR) -> Dict[str, Any]:


    global _EMBEDDINGS, _REFERENCE_TEXTS, _EMBEDDING_MODEL
    # Load if not yet loaded
    if _EMBEDDINGS is None or _REFERENCE_TEXTS is None:
        load_file_level_indexes(indexes_dir, model_name)

    # Encode test code
    test_embedding = _EMBEDDING_MODEL.encode([test_code], convert_to_numpy=True)

    # Compute cosine similarity
    sims = cosine_sim(test_embedding, _EMBEDDINGS)

    # Retrieve top-k
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_matches = [
        {
            "index": int(i),
            "similarity": float(sims[i]),
            "snippet": _REFERENCE_TEXTS[i][:1000] + ("...<truncated>" if len(_REFERENCE_TEXTS[i]) > 1000 else "")
        }
        for i in top_idx
    ]

    # Evaluate plagiarism
    max_sim = float(sims[top_idx[0]]) if len(top_idx) > 0 else 0.0
    is_plagiarized = max_sim >= threshold

    return {
        "is_plagiarized": is_plagiarized,
        "max_similarity": max_sim,
        "top_matches": top_matches,
    }


# Example usage
sample = """import re def bhg(bhg: str) -> str: if len(re.findall("[ATCG]", bhg)) != len(bhg): 
raise ValueError("Invalid Str") return bhg.translate(bhg.maketrans("ATCG", "TAGC"))
if __name__ == "__main__": import doctest doctest.testmod()
"""
result = detect_embedding(sample, top_k=3, threshold=0.70)
result



# %% [markdown]
# Now let's implemets second function. first of all I will write impoerts for openai and take api_key from env_variables.

# %%
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


with open("indexes/clean_texts.pkl", "rb") as f:
    REFERENCE_FILES = pickle.load(f)

print(f"Loaded {len(REFERENCE_FILES)} files for direct LLM analysis.")

# %% [markdown]
# Now I write implementation of function that builds a long LLM prompt that includes:
#     - Test snippet
#     - All reference code files
#     - Question to determine plagiarism

# %%
def build_llm_prompt(test_code: str, reference_files: list) -> str:
    
    prompt = "You are an expert in code similarity and plagiarism detection.\n"
    prompt += "Your task is to compare a TEST CODE with multiple REFERENCE FILES.\n"
    prompt += "Determine if the TEST CODE is plagiarized from any reference.\n"
    prompt += "Consider logic, variable renaming, refactoring, and structure changes.\n\n"

    prompt += "TEST CODE:"
    prompt += test_code + "\n\n"

    prompt += "REFERENCE CODE FILES:"
    for i, code in enumerate(reference_files):  
        prompt += f"\n--- FILE {i+1} ---\n"
        prompt += code[:1000]  # avoid exceeding token limits
        prompt += "\n"

    prompt += "\n\nIMPORTANT:\n"
    prompt += "1. Answer YES if logic is copied even with renaming/refactoring.\n"
    prompt += "2. Otherwise, answer NO.\n"
    prompt += "Then very briefly explain your reasoning.\n"

    return prompt


# %% [markdown]
#     main function, Sends raw test_code + full (or partial) reference corpus to LLM I had to limit number of files to avoid token limit excedin. 
# 

# %%
def detect_llm(test_code: str,
               max_files: int = 5,
               model: str = "gpt-3.5-turbo") -> dict:
   

    # Clean test code to reduce noise (same as in indexing)
    test_code = clean_code(test_code)

    # Select a few reference files (to fit LLM context)
    reference_subset = REFERENCE_FILES[:max_files]

    # Build prompt
    prompt = build_llm_prompt(test_code, reference_subset)

    # Call the API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert code plagiarism evaluator."},
            {"role": "user", "content": prompt},
        ]
    )

    answer = response.choices[0].message.content

    return {
        "model_used": model,
        "llm_response": answer
    }





# %% [markdown]
# Standard RAG — Embedding Retrieval + LLM Reasoning
# 
# This function:
# 1. Retrieves top-k similar files using cosine similarity.
# 2. Sends those retrieved files + test code into LLM.
# 3. LLM decides final plagiarism status.
# 
# 
# belove function builds a prompt for RAG-based analysis:
#     - Includes test code
#     - Includes only retrieved top-k reference files
#     - Asks LLM to judge plagiarism
# 
# 

# %%
def build_rag_prompt(test_code: str, retrieved_docs: list) -> str:

    prompt = "You are an expert in code plagiarism detection.\n\n"
    prompt += "Compare the TEST CODE with RETRIEVED FILES.\n"
    prompt += "Determine if plagiarism exists, even if variables were renamed or structure changed.\n\n"

    prompt += "=== TEST CODE ===\n"
    prompt += test_code + "\n\n"

    prompt += "=== RETRIEVED FILES ===\n"
    for i, doc in enumerate(retrieved_docs):
        prompt += f"\n--- FILE {i+1} (similarity: {doc['similarity']:.3f}) ---\n"
        prompt += doc["snippet"][:1000]  
        prompt += "\n"

    prompt += "\nFINAL TASK:\n"
    prompt += "1. Answer YES or NO: Is the TEST CODE plagiarized?\n"
    prompt += "2. Provide very short reason.\n"

    return prompt


# %% [markdown]
# 1. Use embeddings to retrieve top-k similar files.
# 2. Send ONLY those to an LLM for final decision.

# %%
def detect_rag(test_code: str,
               top_k: int = 5,
               threshold: float = 0.75,
               model: str = "gpt-4o-mini") -> dict:
    
    # 1) EMBEDDING RETRIEVAL (reuse detect_embedding logic)
    embed_result = detect_embedding(test_code, top_k=top_k, threshold=threshold)
    retrieved = embed_result["top_matches"]

    # 2) CLEAN test code (remove comments & spaces)
    test_code = clean_code(test_code)

    # 3) Build prompt
    prompt = build_rag_prompt(test_code, retrieved)

    # 4) LLM CALL
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert code plagiarism evaluator."},
            {"role": "user", "content": prompt}
        ]
    )

    final_answer = response.choices[0].message.content

    return {
        "retrieved_similarity": embed_result["max_similarity"],
        "retrieved_files": retrieved,
        "llm_response": final_answer
    }


# test_code = """
# def is_even(n):
#     return n % 2 == 0
# """

# result = detect_rag(test_code, top_k=3)
# print(result["llm_response"])



# %% [markdown]
# Hybrid RAG — BM25 + Embedding Retrieval + LLM
# 
# This method retrieves using:
# 1. BM25 (lexical / keyword similarity)
# 2. Embeddings (semantic similarity)
# 
# Then merges both → sends to LLM → final plagiarism decision.

# %%
from rank_bm25 import BM25Okapi

tokenized_corpus = [doc.split() for doc in REFERENCE_FILES]

# Build BM25 model (only once!)
bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index built successfully — ready for hybrid RAG")


# %%
def bm25_retrieve(test_code: str, top_k: int = 5):
    """
    Returns top-k BM25 lexical matches.
    """
    tokens = test_code.split()
    scores = bm25.get_scores(tokens)

    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({
            "index": int(i),
            "score": float(scores[i]),
            "snippet": REFERENCE_FILES[i][:1000] + ("...<truncated>" if len(REFERENCE_FILES[i]) > 1000 else "")
        })
    return results


# %% [markdown]
# Combines BM25 (keyword relevance) and embeddings (semantic relevance) with weighted fusion of scores.

# %%
def hybrid_retrieve(test_code: str,
                    top_k_embeddings: int = 5,
                    top_k_bm25: int = 5,
                    weight_emb: float = 0.5,
                    weight_bm25: float = 0.5):
    
    # 1) EMBEDDING RESULTS
    emb_result = detect_embedding(test_code, top_k=top_k_embeddings)
    emb_matches = {m["index"]: m["similarity"] for m in emb_result["top_matches"]}

    # 2) BM25 RESULTS
    bm25_matches = bm25_retrieve(test_code, top_k=top_k_bm25)
    bm25_dict = {m["index"]: m["score"] for m in bm25_matches}

    # 3) SCORE FUSION
    all_indices = set(emb_matches.keys()) | set(bm25_dict.keys())
    fused_results = []
    for idx in all_indices:
        score = (
            weight_emb  * emb_matches.get(idx, 0) +
            weight_bm25 * bm25_dict.get(idx, 0)
        )
        fused_results.append({
            "index": idx,
            "similarity": score,
            "from_embedding": emb_matches.get(idx, 0),
            "from_bm25": bm25_dict.get(idx, 0)
        })

    # 4) SORT & TAKE BEST
    fused_results = sorted(fused_results, key=lambda x: x["similarity"], reverse=True)[:max(top_k_embeddings, top_k_bm25)]

    # 5) ATTACH SNIPPETS
    for r in fused_results:
        r["snippet"] = REFERENCE_FILES[r["index"]][:1000] + (
            "...<truncated>" if len(REFERENCE_FILES[r["index"]]) > 1000 else ""
        )

    return fused_results


# %% [markdown]
# final function:
#     1. Embedding + BM25 retrieval
#     2. Fused results → LLM

# %%
def detect_hybrid_rag(test_code: str,
                      top_k_embeddings: int = 5,
                      top_k_bm25: int = 5,
                      weight_emb: float = 0.5,
                      weight_bm25: float = 0.5,
                      model: str = "gpt-4o-mini"):
    
    # Retrieve documents using fusion
    fuse_results = hybrid_retrieve(
        test_code,
        top_k_embeddings=top_k_embeddings,
        top_k_bm25=top_k_bm25,
        weight_emb=weight_emb,
        weight_bm25=weight_bm25
    )

    # Clean test code before sending to LLM
    test_code = clean_code(test_code)

    # Build final prompt
    prompt = build_rag_prompt(test_code, fuse_results)

    # Call LLM
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a code plagiarism expert."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    return {
        "retrieved_docs": fuse_results,
        "llm_response": answer
    }

# test_code = """

# def power_of_4(num: int) -> bool:
    
#     if not isinstance(num, int):
#         raise TypeError("num must be an integer")
#     if num & (num - 1) == 0:
#         c = 0
#         while num:
#             c += 1
#             num >>= 1
#         return c % 2 == 1
#     else:
#         return False


# if __name__ == "__main__":
#     import doctest

#     doctest.testmod()

# """
# result = detect_hybrid_rag(test_code)
# print(result["llm_response"])




