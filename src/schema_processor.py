# src/schema_processor.py
"""Schema Processor Module
This module provides functionality to process JSON schemas, extract fields, and intelligently group them
for retrieval queries.
It supports various schema formats, extracts individual fields, and groups them based on parent path,
size, and similarity. The goal is to create meaningful groups to combine into a retrieval query.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import config


@dataclass
class FieldDescriptor:
    path: str
    field_type: str
    description: Optional[str]
    enum_values: Optional[List[str]]
    required: bool
    parent_path: str
    depth: int
    pattern: Optional[str] = None
    examples: Optional[List[str]] = None


@dataclass
class FieldGroup:
    group_id: str
    fields: List[FieldDescriptor]
    shared_context: str
    complexity_score: float
    query_terms: List[str]


class SchemaProcessor:
    """
    This class processes JSON schemas to extract fields and group them intelligently.
    It supports various schema formats, extracts individual fields, and groups them
    based on parent path, size, and similarity. The goal is to create meaningful
    groups to combine into a retrieval query.

    Key steps:
    1. Schema loading and validation
    2. Recursive field extraction
    3. Intelligent grouping using multiple strategies
       (grouping by parent path, splitting large groups, merging small groups)

    The class handles nested structures and generates contextual information for each group.
    """

    def __init__(self, max_group_size: int = 5, max_complexity_per_group: float = 1.0):
        self.max_group_size = int(
            getattr(config, "MAX_FIELDS_PER_GROUP", max_group_size)
        )
        self.max_complexity_per_group = max_complexity_per_group
        self.logger = logging.getLogger(__name__)

    def process_schema(
        self, schema_path: str
    ) -> Tuple[List[FieldDescriptor], List[FieldGroup]]:
        """Main entry point - parse any schema and create intelligent groups"""
        schema = self._load_schema(schema_path)
        fields = self._extract_all_fields(schema)
        groups = self._create_intelligent_groups(fields)

        self.logger.info(
            f"Processed schema: {len(fields)} fields → {len(groups)} groups"
        )
        return fields, groups

    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load and validate any JSON schema format"""
        with open(schema_path, "r") as f:
            schema = json.load(f)

        # Handle different schema formats
        if "$schema" in schema:
            # JSON Schema Draft-07/2019-09/2020-12
            return schema
        elif "type" in schema or "properties" in schema:
            # Direct schema object
            return schema
        else:
            # Assume it's a data example, try to infer structure
            return self._infer_schema_from_example(schema)

    def _extract_all_fields(
        self,
        schema: Dict[str, Any],
        current_path: str = "",
        parent_required: Optional[List[str]] = None,
        depth: int = 0,
    ) -> List[FieldDescriptor]:
        """Recursively extract all fields from any nested schema structure"""
        fields = []
        parent_required = parent_required or []

        # Handle definitions section first (for $ref resolution)
        definitions = schema.get("definitions", {})

        # Handle different schema structures
        if "properties" in schema:
            # Standard JSON Schema
            properties = schema["properties"]
            required_fields = schema.get("required", [])

            for field_name, field_def in properties.items():
                field_path = (
                    f"{current_path}.{field_name}" if current_path else field_name
                )
                fields.extend(
                    self._process_field(
                        field_name,
                        field_def,
                        field_path,
                        field_name in required_fields,
                        current_path,
                        depth,
                        definitions,
                    )
                )

        elif "definitions" in schema and not current_path:
            # Schema with definitions section - process definitions
            for def_name, def_schema in schema["definitions"].items():
                def_path = f"definitions.{def_name}"
                fields.extend(
                    self._extract_all_fields(
                        def_schema, def_path, parent_required, depth + 1
                    )
                )

        elif isinstance(schema, dict):
            # Generic nested object
            for key, value in schema.items():
                if key in [
                    "$schema",
                    "$id",
                    "additionalProperties",
                    "title",
                    "description",
                ]:
                    continue  # Skip metadata fields

                if isinstance(value, dict):
                    field_path = f"{current_path}.{key}" if current_path else key
                    fields.extend(
                        self._extract_all_fields(
                            value, field_path, parent_required, depth + 1
                        )
                    )
                else:
                    # Leaf field
                    field_path = f"{current_path}.{key}" if current_path else key
                    fields.append(
                        self._create_field_descriptor(
                            key, value, field_path, False, current_path, depth
                        )
                    )

        return fields

    def _process_field(
        self,
        field_name: str,
        field_def: Any,
        field_path: str,
        is_required: bool,
        parent_path: str,
        depth: int,
        definitions: Dict[str, Any],
    ) -> List[FieldDescriptor]:
        """Process individual field definition"""
        fields = []

        if isinstance(field_def, dict):
            # Handle $ref references
            if "$ref" in field_def:
                ref_path = field_def["$ref"]
                if ref_path.startswith("#/definitions/"):
                    ref_name = ref_path.split("/")[-1]
                    if ref_name in definitions:
                        # Resolve reference and process
                        resolved_def = definitions[ref_name]
                        fields.extend(
                            self._process_field(
                                field_name,
                                resolved_def,
                                field_path,
                                is_required,
                                parent_path,
                                depth,
                                definitions,
                            )
                        )
                        return fields

            # Handle allOf, anyOf, oneOf
            for combine_key in ["allOf", "anyOf", "oneOf"]:
                if combine_key in field_def:
                    for option in field_def[combine_key]:
                        fields.extend(
                            self._process_field(
                                field_name,
                                option,
                                field_path,
                                is_required,
                                parent_path,
                                depth,
                                definitions,
                            )
                        )
                    return fields

            field_type = field_def.get("type", "unknown")

            if field_type == "object" or "properties" in field_def:
                # Nested object - recurse
                fields.extend(
                    self._extract_all_fields(
                        field_def, field_path, field_def.get("required", []), depth + 1
                    )
                )
            elif field_type == "array":
                # Array field
                items_def = field_def.get("items", {})
                if isinstance(items_def, dict) and (
                    "properties" in items_def or items_def.get("type") == "object"
                ):
                    # Array of objects
                    array_path = f"{field_path}[]"
                    fields.extend(
                        self._extract_all_fields(items_def, array_path, [], depth + 1)
                    )
                else:
                    # Array of primitives
                    fields.append(
                        self._create_field_descriptor(
                            field_name,
                            field_def,
                            field_path,
                            is_required,
                            parent_path,
                            depth,
                        )
                    )
            else:
                # Primitive field
                fields.append(
                    self._create_field_descriptor(
                        field_name,
                        field_def,
                        field_path,
                        is_required,
                        parent_path,
                        depth,
                    )
                )
        else:
            # Simple value
            fields.append(
                self._create_field_descriptor(
                    field_name, field_def, field_path, is_required, parent_path, depth
                )
            )

        return fields

    def _create_field_descriptor(
        self,
        field_name: str,
        field_def: Any,
        field_path: str,
        is_required: bool,
        parent_path: str,
        depth: int,
    ) -> FieldDescriptor:
        """Create FieldDescriptor from field definition"""
        if isinstance(field_def, dict):
            field_type = field_def.get("type", "string")
            description = field_def.get("description", "")
            enum_values = field_def.get("enum", None)
            pattern = field_def.get("pattern", None)
            examples = field_def.get("examples", None)
        else:
            field_type = type(field_def).__name__
            description = f"Field {field_name}"
            enum_values = None
            pattern = None
            examples = None

        return FieldDescriptor(
            path=field_path,
            field_type=field_type,
            description=description,
            enum_values=enum_values,
            required=is_required,
            parent_path=parent_path,
            depth=depth,
            pattern=pattern,
            examples=examples,
        )

    def _create_intelligent_groups(
        self, fields: List[FieldDescriptor]
    ) -> List[FieldGroup]:
        """Create intelligent field groups based on multiple strategies"""
        if not fields:
            return []

        groups = []

        # Strategy 1: Group by parent path and depth
        path_groups = self._group_by_parent_path(fields)

        # Strategy 2: Break large groups by complexity
        refined_groups = []
        for group in path_groups:
            if len(group) <= self.max_group_size:
                refined_groups.append(group)
            else:
                refined_groups.extend(self._split_large_group(group))

        # Strategy 3: Merge small related groups
        final_groups = self._merge_small_groups(refined_groups)

        # Convert to FieldGroup objects
        for i, field_list in enumerate(final_groups):
            shared_context = self._generate_shared_context(field_list)
            complexity = self._calculate_group_complexity(field_list)
            query_terms = self._extract_query_terms(field_list)

            groups.append(
                FieldGroup(
                    group_id=f"group_{i:03d}",
                    fields=field_list,
                    shared_context=shared_context,
                    complexity_score=complexity,
                    query_terms=query_terms,
                )
            )

        return groups

    def _group_by_parent_path(
        self, fields: List[FieldDescriptor]
    ) -> List[List[FieldDescriptor]]:
        """Group fields by their parent path"""
        path_groups = {}

        for field in fields:
            # Use parent path + depth as grouping key
            if field.parent_path:
                group_key = f"{field.parent_path}_depth_{field.depth}"
            else:
                group_key = f"root_depth_{field.depth}"

            if group_key not in path_groups:
                path_groups[group_key] = []
            path_groups[group_key].append(field)

        return list(path_groups.values())

    def _split_large_group(
        self, fields: List[FieldDescriptor]
    ) -> List[List[FieldDescriptor]]:
        """Split large groups into smaller ones"""
        groups = []
        current_group = []

        for field in fields:
            current_group.append(field)

            if len(current_group) >= self.max_group_size:
                groups.append(current_group)
                current_group = []

        if current_group:
            groups.append(current_group)

        return groups

    def _merge_small_groups(
        self, groups: List[List[FieldDescriptor]]
    ) -> List[List[FieldDescriptor]]:
        """Merge very small groups with similar parent paths"""
        merged = []
        small_groups = []

        for group in groups:
            if len(group) <= 2:
                small_groups.append(group)
            else:
                merged.append(group)

        # Try to merge small groups
        current_merge = []
        for small_group in small_groups:
            current_merge.extend(small_group)

            if len(current_merge) >= 3:
                merged.append(current_merge)
                current_merge = []

        if current_merge:
            merged.append(current_merge)

        return merged

    def _generate_shared_context(self, fields: List[FieldDescriptor]) -> str:
        """Generate shared context description for a group"""
        if not fields:
            return ""

        # Find common parent path
        paths = [f.parent_path for f in fields if f.parent_path]
        common_parent = self._find_common_prefix(paths) if paths else "root"

        # Combine descriptions
        descriptions = [
            f.description for f in fields if f.description and f.description.strip()
        ]
        unique_descriptions = list(set(descriptions))

        context = f"Fields under '{common_parent}'"
        if unique_descriptions:
            context += f": {' | '.join(unique_descriptions[:3])}"

        return context

    def _calculate_group_complexity(self, fields: List[FieldDescriptor]) -> float:
        """Calculate complexity score for a group"""
        base_score = len(fields) * 0.1

        # Add complexity for different types
        type_variety = len(set(f.field_type for f in fields)) * 0.1

        # Add complexity for enums
        enum_complexity = sum(1 for f in fields if f.enum_values) * 0.2

        # Add complexity for deep nesting
        depth_complexity = max(f.depth for f in fields) * 0.1

        # Add complexity for patterns
        pattern_complexity = sum(1 for f in fields if f.pattern) * 0.15

        return (
            base_score
            + type_variety
            + enum_complexity
            + depth_complexity
            + pattern_complexity
        )

    def _extract_query_terms(self, fields: List[FieldDescriptor]) -> List[str]:
        """Extract key query terms for retrieval"""
        terms = []

        for field in fields:
            # Add field path components
            path_parts = field.path.split(".")
            terms.extend(path_parts)

            # Add description keywords
            if field.description:
                # Simple keyword extraction
                desc_words = field.description.lower().split()
                important_words = [w for w in desc_words if len(w) > 3 and w.isalpha()]
                terms.extend(important_words[:3])  # Limit to top 3 per field

            # Add enum values (limited)
            if field.enum_values:
                terms.extend(field.enum_values[:5])  # Limit enum terms

        # Remove duplicates and return unique terms
        return list(set(terms))

    def _find_common_prefix(self, paths: List[str]) -> str:
        """Find common prefix among paths"""
        if not paths:
            return ""

        common = paths[0]
        for path in paths[1:]:
            while not path.startswith(common):
                common = common[:-1]
                if not common:
                    break

        return common.rstrip(".")

    def _infer_schema_from_example(self, data: Any, path: str = "") -> Dict[str, Any]:
        """Infer schema structure from example data"""
        if isinstance(data, dict):
            properties = {}
            for key, value in data.items():
                properties[key] = self._infer_schema_from_example(
                    value, f"{path}.{key}"
                )
            return {"type": "object", "properties": properties}
        elif isinstance(data, list):
            if data:
                items_schema = self._infer_schema_from_example(data[0], f"{path}[]")
                return {"type": "array", "items": items_schema}
            else:
                return {"type": "array", "items": {"type": "string"}}
        elif isinstance(data, str):
            return {"type": "string", "description": f"String field at {path}"}
        elif isinstance(data, (int, float)):
            return {"type": "number", "description": f"Number field at {path}"}
        elif isinstance(data, bool):
            return {"type": "boolean", "description": f"Boolean field at {path}"}
        else:
            return {"type": "string", "description": f"Unknown field at {path}"}


def create_grouped_queries(groups: List[FieldGroup]) -> Dict[str, str]:
    """Create optimized retrieval queries for each group"""
    queries = {}

    for group in groups:
        # Combine field paths, descriptions, and query terms
        query_parts = []

        # Add field paths
        paths = [f.path for f in group.fields]
        query_parts.append(" ".join(paths))

        # Add descriptions
        descriptions = [f.description for f in group.fields if f.description]
        if descriptions:
            query_parts.append(" ".join(descriptions))

        # Add query terms (most important keywords)
        if group.query_terms:
            query_parts.append(" ".join(group.query_terms[:10]))  # Limit terms

        # Create comprehensive query
        full_query = " ".join(query_parts)
        queries[group.group_id] = full_query

    return queries
