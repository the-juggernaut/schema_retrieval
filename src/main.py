"""
Schema-based Retrieval Pipeline
This script processes documents, builds indexes, and retrieves schema-based information.

"""

import os
import logging
import json
import numpy as np
import argparse

from schema_processor import SchemaProcessor, create_grouped_queries
from document_segmenter import DocumentSegmenter
from retriever import Retriever
from composer import Composer

from sentence_transformers import SentenceTransformer
import tiktoken
import config
import jsonschema


def main():
    parser = argparse.ArgumentParser(
        description="Schema-based Retrieval Pipeline",
        epilog="This script processes documents, builds indexes, and retrieves schema-based information.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Input data directory containing documents",
    )
    parser.add_argument(
        "--schema_file", type=str, required=True, help="Target schema JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output/final_output.json",
        help="Output file name",
    )
    args = parser.parse_args()

    print("Schema-based Retrieval Pipeline")
    print("=" * 60)
    print(f"Schema: {args.schema_file}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output File: {args.output_file}")

    processor = SchemaProcessor(
        max_group_size=getattr(config, "MAX_FIELDS_PER_GROUP", 5),
        max_complexity_per_group=1.2,
    )
    fields, groups = processor.process_schema(args.schema_file)

    print("\nSchema Analysis Results:")
    print(f"   Extracted {len(fields)} fields into {len(groups)} groups")

    group_queries = create_grouped_queries(groups)
    print(f"   Generated {len(group_queries)} retrieval queries")

    print("\nDetailed Analysis:")
    print(f"   Average fields per group: {len(fields) / len(groups):.1f}")
    print(f"   Max group complexity: {max(g.complexity_score for g in groups):.2f}")
    print(f"   Min group complexity: {min(g.complexity_score for g in groups):.2f}")

    print("\nSample Field Groups:")
    print("-" * 50)
    for i, group in enumerate(groups[:3]):
        print(f"\n{group.group_id.upper()}:")
        print(f"   Context: {group.shared_context}")
        print(f"   Complexity: {group.complexity_score:.2f}")
        print(f"   Fields ({len(group.fields)}):")
        for field in group.fields:
            field_info = f"     {field.path} ({field.field_type})"
            if field.required:
                field_info += " [REQUIRED]"
            if field.enum_values:
                field_info += f" [ENUM: {len(field.enum_values)} values]"
            print(field_info)
            if field.description:
                desc = field.description[:80]
                if len(field.description) > 80:
                    desc += "..."
                print(f"       {desc}")
        query = group_queries.get(group.group_id, "")
        print(f"   Retrieval Query:")
        print(f"     {query[:120]}{'...' if len(query) > 120 else ''}")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    def dense_embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        return model.encode(texts)

    segmenter = DocumentSegmenter()
    logger.info(f"Indexing all files in directory: {args.data_dir}")
    results = segmenter.process_directory(args.data_dir, embed_fn=dense_embed)
    all_segments = []
    all_embeddings = []
    for fpath, data in results.items():
        logger.info(f"Processed file: {fpath} | Segments: {len(data['segments'])}")
        all_segments.extend(data["segments"])
        if data["embeddings"] is not None:
            all_embeddings.extend(data["embeddings"])
    all_embeddings = np.array(all_embeddings)
    logger.info(f"Segmented {len(all_segments)} segments from {len(results)} files.")

    retriever = Retriever(
        embedding_dim=all_embeddings.shape[1] if all_embeddings.size else None
    )
    retriever.build_indexes(all_segments, all_embeddings)
    logger.info("Built BM25 and FAISS indexes.")

    composer = Composer(retriever, groups)
    logger.info("Running Composer to assemble final output...")
    final_output = composer.compose(k=5)

    # Aggregate compute stats from submodules
    stats = {}
    # DocumentSegmenter stats
    if hasattr(segmenter, "compute_stats"):
        stats.update({f"segmenter_{k}": v for k, v in segmenter.compute_stats.items()})
    stats["num_segments"] = len(all_segments)
    stats["num_embeddings"] = len(all_embeddings)
    # Retriever stats
    if hasattr(retriever, "compute_stats"):
        stats.update({f"retriever_{k}": v for k, v in retriever.compute_stats.items()})
    # Composer stats
    if hasattr(composer, "compute_stats"):
        stats.update({f"composer_{k}": v for k, v in composer.compute_stats.items()})

    logger.info("Final composed output (truncated):")
    logger.info(json.dumps(final_output, indent=2)[:2000])

    logger.info("Compute Stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    with open(args.schema_file) as sf:
        schema = json.load(sf)

    # Validate final_output (unwrap values for validation, handle nesting)
    def unwrap_values(obj):
        if isinstance(obj, dict):
            # If this dict has only a "value" key, unwrap it
            if set(obj.keys()) == {"value"}:
                return unwrap_values(obj["value"])
            # Otherwise, recursively unwrap each key
            return {k: unwrap_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [unwrap_values(item) for item in obj]
        else:
            return obj

    validation_output = unwrap_values(final_output)
    try:
        jsonschema.validate(instance=validation_output, schema=schema)
        logger.info("Output JSON is valid against the schema.")
    except jsonschema.ValidationError as e:
        logger.warning(f"Schema validation failed: {e.message}. Saving output anyway.")

    out_dir = os.path.dirname(args.output_file)
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"Schema was reconstructed! {args.output_file} written successfully.")


if __name__ == "__main__":
    main()
