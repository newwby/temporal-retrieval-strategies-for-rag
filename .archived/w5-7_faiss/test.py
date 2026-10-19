"""
first faiss test, embedding only retrieval baseline
uses faiss-cpu in the container, swap to faiss on desktop
"""

from sentence_transformers import SentenceTransformer
import faiss

# sample docs for the baseline
docs = [
    {"id": 0, "text": "The economic crisis of the early 1990s was severe."},
    {"id": 1, "text": "In the mid-1990s, recovery began."},
    {"id": 2, "text": "The 2008 market crash devastated markets."},
    {"id": 3, "text": "Unemployment peaked in 1992 at 7.8%."},
    {"id": 4, "text": "In 1991, the recession deepened across manufacturing."},
    {"id": 5, "text": "By 1995, GDP growth returned to positive."},
    {"id": 6, "text": "The early 1990s saw major bank failures."},
    {"id": 7, "text": "In 1994, interest rates began climbing."},
    {"id": 8, "text": "The late 1990s brought the dot-com boom."},
    {"id": 9, "text": "Internet stocks surged in 1999."},
    {"id": 10, "text": "The 2000s opened with a tech sector correction."},
    {"id": 11, "text": "Post-9/11 economic uncertainty gripped markets."},
    {"id": 12, "text": "Housing prices soared during the mid-2000s."},
    {"id": 13, "text": "In late 2008, financial institutions collapsed."},
    {"id": 14, "text": "The 2007-2009 financial crisis was unprecedented."},
    {"id": 15, "text": "Bank bailouts defined the late 2000s."},
    {"id": 16, "text": "The 2010s began with recovery policies."},
    {"id": 17, "text": "By 2015, unemployment fell below 5%."},
    {"id": 18, "text": "Central banks kept rates low through 2018."},
    {"id": 19, "text": "In 2020, the pandemic caused economic shock."},
]

if __name__ == "__main__":
  print("Running test.py")
  # embed docs
  embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
  doc_embeddings = embedding_model.encode([d["text"] for d in docs])

  # l2 index, all-MiniLM-L6-v2 is 384 dim
  index = faiss.IndexFlatL2(384)
  index.add(doc_embeddings)

  # or a fuzzy paraphrase
  query = "early 1990s"

  # step 1: embed query (no temporal normalisation yet)
  query_embedding = embedding_model.encode(query)

  # step 2: search
  distances, doc_ids = index.search(query_embedding.reshape(1, -1), k=10)

  # step 3: get candidates
  candidates = [docs[i] for i in doc_ids[0]]

  # step 4: print ranked results
  idx = 0
  for candidate in candidates:
      idx += 1
      print(f"Rank {idx}: Candidate ID: {candidate['id']}, Text: {candidate['text']}")

#TODO add mrr, recall/precision@k and exact match metrics
#TODO add fuzzy query translation
#TODO generate structured queries
#TODO compare temporal normalisation like DeFuzzRAG

"""
sample output

Rank 1: Candidate ID: 1, Text: In the mid-1990s, recovery began.
Rank 2: Candidate ID: 0, Text: The economic crisis of the early 1990s was severe.
Rank 3: Candidate ID: 6, Text: The early 1990s saw major bank failures.
Rank 4: Candidate ID: 8, Text: The late 1990s brought the dot-com boom.
Rank 5: Candidate ID: 4, Text: In 1991, the recession deepened across manufacturing.
Rank 6: Candidate ID: 10, Text: The 2000s opened with a tech sector correction.
Rank 7: Candidate ID: 16, Text: The 2010s began with recovery policies.
Rank 8: Candidate ID: 15, Text: Bank bailouts defined the late 2000s.
Rank 9: Candidate ID: 12, Text: Housing prices soared during the mid-2000s.
Rank 10: Candidate ID: 7, Text: In 1994, interest rates began climbing.

hits for "early 1990s":
hit@10 - 4 relevant (ids 0, 6, 4, 7)
hit@5 - 3 relevant (ids 0, 6, 4)
hit@1 - none, id 1 is mid-1990s not early 1990s

embedding alone isn't enough, needs fuzzy query translation
it mixes up early/mid above, temporal normalisation might fix this

"""