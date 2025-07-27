#!/usr/bin/env python3
"""
Test script for the unified schema_processor.py module.
Tests with different schema formats to validate generic capabilities.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schema_processor import SchemaProcessor, create_grouped_queries


def test_schema_processor():
    """Test the unified schema processor with various schema formats."""

    processor = SchemaProcessor(max_group_size=5, max_complexity_per_group=1.0)

    # Test data directory
    data_dir = Path(__file__).parent.parent / "src" / "data"

    test_cases = [
        {
            "name": "Citation Schema (Complex)",
            "path": data_dir / "citations" / "paper citations_schema.json",
            "expected_min_fields": 50,
            "expected_min_groups": 5,
        },
        {
            "name": "GitHub Actions Schema",
            "path": data_dir / "github" / "github_actions_schema.json",
            "expected_min_fields": 10,
            "expected_min_groups": 2,
        },
        {
            "name": "Resume Schema",
            "path": data_dir / "resume" / "resume_schema.json",
            "expected_min_fields": 5,
            "expected_min_groups": 1,
        },
    ]

    results = []

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['name']}")
        print(f"File: {test_case['path']}")

        if not test_case["path"].exists():
            print(f"❌ SKIP: File not found")
            continue

        try:
            # Process the schema
            fields, groups = processor.process_schema(str(test_case["path"]))

            # Basic validation
            field_count = len(fields)
            group_count = len(groups)

            print(f"📊 Results:")
            print(f"   Fields extracted: {field_count}")
            print(f"   Groups created: {group_count}")

            # Check expectations
            field_ok = field_count >= test_case["expected_min_fields"]
            group_ok = group_count >= test_case["expected_min_groups"]

            print(
                f"   Fields >= {test_case['expected_min_fields']}: {'✅' if field_ok else '❌'}"
            )
            print(
                f"   Groups >= {test_case['expected_min_groups']}: {'✅' if group_ok else '❌'}"
            )

            # Show sample fields
            print(f"\n📝 Sample Fields:")
            for field in fields[:3]:
                print(f"   {field.path} ({field.field_type})")
                if field.description:
                    print(f"      → {field.description[:60]}...")

            # Show sample groups
            print(f"\n📦 Sample Groups:")
            for group in groups[:2]:
                print(f"   {group.group_id}: {len(group.fields)} fields")
                print(f"      Context: {group.shared_context[:80]}...")
                print(f"      Complexity: {group.complexity_score:.2f}")

            # Test query generation
            queries = create_grouped_queries(groups)
            print(f"\n🔍 Query Generation:")
            print(f"   Generated {len(queries)} queries")
            if queries:
                first_query = list(queries.values())[0]
                print(f"   Sample: {first_query[:100]}...")

            # Record results
            results.append(
                {
                    "name": test_case["name"],
                    "success": field_ok and group_ok,
                    "fields": field_count,
                    "groups": group_count,
                    "has_queries": len(queries) > 0,
                }
            )

        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append(
                {"name": test_case["name"], "success": False, "error": str(e)}
            )

    # Summary
    print(f"\n{'='*60}")
    print("🎯 SUMMARY")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if r.get("success", False))
    total_count = len(results)

    for result in results:
        status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
        print(f"{status} {result['name']}")
        if "fields" in result:
            print(f"      {result['fields']} fields → {result['groups']} groups")
        if "error" in result:
            print(f"      Error: {result['error']}")

    print(f"\n🏆 Overall: {success_count}/{total_count} tests passed")

    if success_count == total_count:
        print("🎉 All tests passed! The unified schema processor is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the individual results above.")

    return success_count == total_count


if __name__ == "__main__":
    test_schema_processor()
