from typing import List
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np
import config
from datatypes import FieldDescriptor, FieldGroup


class FieldGrouper:
    """Groups schema fields for retrieval and extraction using semantic similarity.

    The FieldGrouper leverages sentence embeddings to cluster schema fields based on the similarity of their descriptions and paths.
    This allows for more efficient retrieval and extraction of related fields.

    Attributes:
        model_name (str): The name of the sentence transformer model to use for embedding. Defaults to config.EMBEDDING_MODEL.
        max_group_size (int): The maximum number of fields allowed in a single group. Defaults to config.MAX_FIELDS_PER_GROUP.
        model (SentenceTransformer): The sentence transformer model used for generating embeddings.
    """

    def __init__(self, model_name: str = "", max_group_size: int = 0):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.max_group_size = max_group_size or config.MAX_FIELDS_PER_GROUP
        self.model = SentenceTransformer(self.model_name)

    def group(self, fields: List[FieldDescriptor]) -> List[FieldGroup]:
        # Always use semantic clustering, include parent in the embedding text
        texts = [f"{f.parent} | {f.path}: {f.description}" for f in fields]
        embeddings = self.model.encode(texts)

        n_fields = len(fields)
        n_clusters = max(1, n_fields // self.max_group_size)
        if n_fields % self.max_group_size:
            n_clusters += 1

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        labels = kmeans.fit_predict(embeddings)

        groups = []
        for group_id in range(labels.max() + 1):
            group_fields = [fields[i] for i in range(n_fields) if labels[i] == group_id]
            groups.append(FieldGroup(group_id=f"group_{group_id}", fields=group_fields))
        return groups
