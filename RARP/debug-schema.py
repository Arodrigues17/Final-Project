#!/usr/bin/env python3
"""
Debug script to examine the Spider schema structure
"""

import json
from pathlib import Path

# Set paths
SPIDER_DIR = Path("../datasets/spider")
DATABASE_ID = "academic"

def main():
    # Load tables data
    tables_path = SPIDER_DIR / "tables.json"
    with open(tables_path, 'r') as f:
        tables_data = json.load(f)
    
    # Find the schema for the specified database
    db_schema = None
    for schema in tables_data:
        if schema["db_id"] == DATABASE_ID:
            db_schema = schema
            break
    
    if not db_schema:
        print(f"Database {DATABASE_ID} not found!")
        return
    
    # Print schema keys and structure
    print(f"Schema keys: {list(db_schema.keys())}")
    
    # Print sample data for each key
    for key in db_schema:
        print(f"\n{key}:")
        value = db_schema[key]
        
        if isinstance(value, list) and len(value) > 0:
            print(f"  Type: {type(value)}, Length: {len(value)}")
            print(f"  First few items: {value[:3]}")
        else:
            print(f"  Value: {value}")

if __name__ == "__main__":
    main()
