#!/usr/bin/env python3
"""
Minimal RARP (RAP + SQLCoder) implementation to fix the schema issue
"""

import os
import json
import argparse
import sqlite3
from typing import Dict, List, Tuple, Any
import re
from pathlib import Path

# Set paths
SPIDER_DIR = Path("../datasets/spider")
DATABASE_DIR = SPIDER_DIR / "database"

class DatabaseSchema:
    """Class to extract and format database schema information"""
    
    def __init__(self, db_id: str, spider_tables_path: str = None):
        self.db_id = db_id
        if spider_tables_path is None:
            spider_tables_path = str(SPIDER_DIR / "tables.json")
        self.spider_tables_path = spider_tables_path
        self.db_path = DATABASE_DIR / db_id / f"{db_id}.sqlite"
        self.tables_data = self._load_spider_tables()
        self.db_schema = self._extract_schema_for_db()
    
    def _load_spider_tables(self) -> List[Dict[str, Any]]:
        """Load tables data from Spider tables.json"""
        with open(self.spider_tables_path, 'r') as f:
            tables_data = json.load(f)
        return tables_data
    
    def _extract_schema_for_db(self) -> Dict[str, Any]:
        """Extract schema information for the specific database"""
        for db_schema in self.tables_data:
            if db_schema["db_id"] == self.db_id:
                return db_schema
        raise ValueError(f"Database {self.db_id} not found in tables data")
    
    def get_formatted_schema(self) -> str:
        """Get simplified formatted schema for RAP"""
        schema_str = f"Database: {self.db_id}\n\n"
        
        # Extract table information - very simplified approach
        for i, table_name in enumerate(self.db_schema["table_names_original"]):
            schema_str += f"Table: {table_name}\n"
            
            # Get columns for this table - only column names
            columns = []
            for col_idx, col_info in enumerate(self.db_schema["column_names_original"]):
                table_idx, col_name = col_info  # Spider format is [table_idx, col_name]
                
                if table_idx == i:
                    columns.append(f"  {col_name}")
            
            schema_str += "\n".join(columns) + "\n\n"
        
        return schema_str

def main():
    """Main function to test the schema extraction"""
    parser = argparse.ArgumentParser(description="Test schema extraction")
    parser.add_argument("--db", required=True, help="Database ID from Spider dataset")
    args = parser.parse_args()
    
    # Test schema extraction
    schema = DatabaseSchema(args.db)
    formatted_schema = schema.get_formatted_schema()
    
    print(f"Formatted schema for database {args.db}:")
    print(formatted_schema)

if __name__ == "__main__":
    main()
