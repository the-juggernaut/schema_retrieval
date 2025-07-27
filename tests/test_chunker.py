import os
import sys
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from document_segmenter import DocumentSegmenter, SegmenterConfig, Segment


@pytest.mark.parametrize(
    "file_path",
    [
        "/home/delen007/qwerty/schema_retrieval/src/data/github/github_action_sample.md",
        "/home/delen007/qwerty/schema_retrieval/src/data/resume/Arpit_Singh_Resume_July_2025.pdf",
        "/home/delen007/qwerty/schema_retrieval/src/data/citations/NIPS-2017-attention-is-all-you-need-Bibtex.bib",
    ],
)
def test_retriever_hybrid_search():
    """
    Test Retriever with dummy segments and hybrid search.
    """
    from src.schema_processor import FieldGroup, FieldDescriptor
    from src.bm25_index import BM25Index
    from src.faiss_index import FaissIndex
    from src.retriever import Retriever

    # Create dummy segments
    segments = [
        Segment(
            id="seg1", text="John is an author of the paper.", source_file="doc1.txt"
        ),
        Segment(
            id="seg2", text="The paper was published in 2025.", source_file="doc1.txt"
        ),
        Segment(
            id="seg3", text="Aditi contributed to the research.", source_file="doc2.txt"
        ),
        Segment(
            id="seg4", text="Arpit reviewed the manuscript.", source_file="doc2.txt"
        ),
    ]

    # Build indexes
    bm25 = BM25Index()
    bm25.build(segments)
    faiss = FaissIndex(model_name="dummy")
    faiss.build(segments)

    # Create a dummy field group
    group = FieldGroup(
        group_id="group_001",
        fields=[
            FieldDescriptor(
                path="authors[].given-names",
                field_type="string",
                description="Author's given name",
                enum_values=None,
                required=True,
                parent_path="authors[]",
                depth=2,
            )
        ],
        shared_context="author given name",
        complexity_score=0.5,
        query_terms=["author", "name", "given"],
    )

    retriever = Retriever()
    results = retriever.retrieve(group, faiss, bm25, k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert any("author" in seg.text for seg in results)
