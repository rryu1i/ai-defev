import os

import uuid
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from dotenv import load_dotenv

load_dotenv()

SPARSE_MODEL = "Qdrant/bm25"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLLECTION_NAME = "financial"
FILE_PATH = "./AAPL_10-K_1A_temp.md"

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

qdrant.delete_collection(collection_name=COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
)

with open(FILE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)

points = []
for chunk in chunks:
    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_embedding = list(
        sparse_model.passage_embed([chunk])
    )[
        0
    ].as_object()  ## transformar a sparse embedding em um formato que o Qdrant aceita (indice e valores -> o resto é 0)
    colbert_embedding = list(colbert_model.passage_embed([chunk]))[0].tolist()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_embedding,
            "sparse": sparse_embedding,
            "colbert": colbert_embedding,
        },
        payload={"text": chunk, "source": FILE_PATH},
    )
    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)


query_text = "What are the main financial risks?"

query_dense = list(dense_model.passage_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.passage_embed([query_text]))[0].as_object()


results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[  ## uma pre-busca
        {"query": query_dense, "using": "dense", "limit": 10},
        {"query": query_sparse, "using": "sparse", "limit": 10},
    ],
    query=models.FusionQuery(
        fusion=models.Fusion.RRF
    ),  ## combinar os resultados das buscas densa e esparsa usando a RRF )
    limit=3,
)

for r in results.points:
    print(f"Score: {r.score}")
    print(f"Payload: {r.payload['text']}")
    print("-" * 50)
