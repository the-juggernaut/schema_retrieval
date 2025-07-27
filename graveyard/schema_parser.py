import json
import os
from typing import Any, List, Optional
from datatypes import FieldDescriptor


class SchemaParser:
    """Loads and parses a schema definition file, and extracts fields."""

    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        self.schema = self.load_schema(schema_path)

    def load_schema(self, schema_path: str) -> dict:
        with open(schema_path, "r") as f:
            return json.load(f)

    def extract_fields(self) -> List[FieldDescriptor]:
        schema = self.schema
        results = []

        # Index all definitions for $ref resolution
        definitions = schema.get("definitions", {})

        def get_parent(path: str) -> str:
            if "." in path:
                return path.rsplit(".", 1)[0]
            return "root"

        def resolve_ref(ref: str):
            # Only supports local refs
            if not ref.startswith("#/"):
                return {}
            parts = ref.replace("#/", "").split("/")
            target = schema
            for part in parts:
                target = target.get(part, {})
            return target

        def recurse(subschema: dict, path: str = ""):
            print(f"[DEBUG] Recursing path: {path} | keys: {list(subschema.keys())}")
            # Handle allOf/anyOf/oneOf
            for key in ["allOf", "anyOf", "oneOf"]:
                if key in subschema:
                    for option in subschema[key]:
                        print(f"[DEBUG] {key} found at {path}")
                        recurse(option, path)
            # Handle $ref
            if "$ref" in subschema:
                print(f"[DEBUG] $ref found at {path}: {subschema['$ref']}")
                ref_target = resolve_ref(subschema["$ref"])
                recurse(ref_target, path)
                return
            # Handle properties
            if "properties" in subschema:
                for key, val in subschema["properties"].items():
                    full_path = f"{path}.{key}" if path else key
                    print(f"[DEBUG] Property: {full_path}")
                    recurse(val, full_path)
            # Handle patternProperties
            if "patternProperties" in subschema:
                for patt, val in subschema["patternProperties"].items():
                    full_path = f"{path}.*" if path else "*"
                    print(f"[DEBUG] PatternProperty: {full_path}")
                    recurse(val, full_path)
            # Handle arrays
            if subschema.get("type") == "array" and "items" in subschema:
                print(f"[DEBUG] Array at {path}")
                recurse(subschema["items"], path + "[]")
            # If this is a leaf field
            if "type" in subschema and not (
                "properties" in subschema or "patternProperties" in subschema
            ):
                print(f"[DEBUG] Field extracted: {path} | type: {subschema['type']}")
                results.append(
                    FieldDescriptor(
                        path=path,
                        parent=get_parent(path),
                        type=subschema["type"],
                        description=subschema.get("description", ""),
                        enum_values=subschema.get("enum"),
                    )
                )

        # Start with top-level properties
        recurse(schema.get("properties", {}))
        return results


# test
import config

if __name__ == "__main__":
    schema_path = os.path.join(
        os.path.dirname(__file__), "data/github/github_actions_schema.json"
    )

    parser = SchemaParser(schema_path)
    fields = parser.extract_fields()
    for field in fields:
        print(field)
