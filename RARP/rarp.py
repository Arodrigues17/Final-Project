#!/usr/bin/env python3
"""
RARP (RAP + SQLCoder) - A combined approach for Text-to-SQL using Groq API
This implementation combines Retrieval-Augmented Prompting (RAP) with SQLCoder prompting patterns
"""

import os
import json
import argparse
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
import re
from groq import Groq
from pathlib import Path

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Set paths
SPIDER_DIR = Path("../datasets/spider")
DATABASE_DIR = SPIDER_DIR / "database"

class DatabaseSchema:
    """Class to extract and format database schema information"""
    
    def __init__(self, db_id: str, spider_tables_path: str = str(SPIDER_DIR / "tables.json")):
        self.db_id = db_id
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
        """Get formatted schema for RAP"""
        schema_str = f"Database: {self.db_id}\n\n"
        
        # Extract table information
        for i, table_name in enumerate(self.db_schema["table_names_original"]):
            schema_str += f"Table: {table_name}\n"
            
            # Get columns for this table
            columns = []
            for col_idx, (table_idx, col_name) in enumerate(zip(
                self.db_schema["column_to_table"],
                self.db_schema["column_names_original"]
            )):
                if table_idx == i:
                    col_type = self.db_schema["column_types"][col_idx]
                    primary_key = "PRIMARY KEY" if col_idx in self.db_schema["primary_keys"] else ""
                    foreign_key = ""
                    
                    # Check if this column is a foreign key
                    for fk in self.db_schema["foreign_keys"]:
                        if fk[0] == col_idx:
                            ref_col_idx = fk[1]
                            ref_table_idx = self.db_schema["column_to_table"][ref_col_idx]
                            ref_table = self.db_schema["table_names_original"][ref_table_idx]
                            ref_col = self.db_schema["column_names_original"][ref_col_idx][1]
                            foreign_key = f"FOREIGN KEY REFERENCES {ref_table}({ref_col})"
                            break
                    
                    columns.append(f"  {col_name} {col_type} {primary_key} {foreign_key}".strip())
            
            schema_str += "\n".join(columns) + "\n\n"
        
        return schema_str
    
    def get_sample_data(self, max_rows: int = 3) -> str:
        """Get sample data from each table for better context"""
        if not self.db_path.exists():
            return "Sample data not available"
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        sample_data = []
        for table_name in self.db_schema["table_names_original"]:
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT {max_rows}")
                rows = cursor.fetchall()
                if rows:
                    sample_data.append(f"Sample data from table {table_name}:")
                    
                    # Get column names
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
                    sample_data.append("  " + " | ".join(columns))
                    
                    # Add rows
                    for row in rows:
                        sample_data.append("  " + " | ".join(str(val) for val in row))
                    sample_data.append("")
            except sqlite3.Error as e:
                sample_data.append(f"Error accessing {table_name}: {e}")
        
        conn.close()
        return "\n".join(sample_data)


class RAP:
    """Retrieval-Augmented Prompting for Text-to-SQL"""
    
    def __init__(self, db_id: str):
        self.db_schema = DatabaseSchema(db_id)
    
    def get_enhanced_context(self, query: str, include_samples: bool = True) -> str:
        """Get enhanced context for the query using RAP"""
        schema = self.db_schema.get_formatted_schema()
        
        context = f"""
Database Schema Information:
{schema}
"""
        
        if include_samples:
            samples = self.db_schema.get_sample_data()
            context += f"""
Sample Data:
{samples}
"""
        
        # Additional analysis for the query to enhance retrieval
        # This could be expanded with more sophisticated retrieval methods
        relevant_tables = self._extract_potential_tables(query)
        if relevant_tables:
            context += f"""
Based on the query, these tables might be particularly relevant: {', '.join(relevant_tables)}
"""
        
        return context
    
    def _extract_potential_tables(self, query: str) -> List[str]:
        """Extract potential tables mentioned in the query"""
        tables = self.db_schema.db_schema["table_names_original"]
        mentioned_tables = []
        
        # Simple word matching for now - could be enhanced with NLP
        query_lower = query.lower()
        for table in tables:
            # Check for plural forms and singular forms
            if table.lower() in query_lower or f"{table.lower()}s" in query_lower:
                mentioned_tables.append(table)
        
        return mentioned_tables


class SQLCoder:
    """SQLCoder-style prompting for Text-to-SQL"""
    
    def __init__(self, model: str = "llama3-70b-8192"):
        self.model = model
    
    def generate_sql(self, query: str, context: str) -> str:
        """Generate SQL using SQLCoder prompting pattern with Groq API"""
        prompt = self._create_prompt(query, context)
        
        try:
            # Call Groq API
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL engineer. Your task is to convert natural language queries to SQL based on the provided database schema."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more deterministic results
                max_tokens=1024
            )
            
            # Extract SQL from response
            sql = self._extract_sql(response.choices[0].message.content)
            return sql
        
        except Exception as e:
            return f"Error generating SQL: {str(e)}"
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt following SQLCoder patterns"""
        prompt = f"""### Task
Generate a SQL query to answer the following question: "{query}"

### Database Schema Information
{context}

### Instructions
1. Use only the tables and columns provided in the schema.
2. Make sure to handle joins appropriately when necessary.
3. Ensure the SQL is compatible with SQLite.
4. Return only the SQL query without explanation.

### SQL Query
```sql
"""
        return prompt
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from the model response"""
        # Look for SQL between code blocks
        match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no code blocks, try to extract the SQL directly
        match = re.search(r"SELECT\s+.*", response, re.DOTALL)
        if match:
            return match.group(0).strip()
        
        return response


class RARP:
    """Combined RAP + SQLCoder approach for Text-to-SQL"""
    
    def __init__(self, db_id: str, model: str = "llama3-70b-8192"):
        self.rap = RAP(db_id)
        self.sqlcoder = SQLCoder(model)
        self.db_id = db_id
    
    def generate_sql(self, query: str, include_samples: bool = True) -> Dict[str, str]:
        """Generate SQL using the combined approach"""
        # Step 1: Use RAP to get enhanced context
        context = self.rap.get_enhanced_context(query, include_samples)
        
        # Step 2: Use SQLCoder with the enhanced context
        sql = self.sqlcoder.generate_sql(query, context)
        
        return {
            "query": query,
            "db_id": self.db_id,
            "sql": sql,
        }
    
    def execute_sql(self, sql: str) -> Tuple[List[Tuple], List[str]]:
        """Execute the generated SQL and return results"""
        db_path = DATABASE_DIR / self.db_id / f"{self.db_id}.sqlite"
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get column names
            cursor.execute(sql)
            column_names = [desc[0] for desc in cursor.description]
            
            # Get results
            results = cursor.fetchall()
            conn.close()
            
            return results, column_names
        except sqlite3.Error as e:
            return [], [f"Error: {str(e)}"]


def main():
    """Main function to run the RARP approach"""
    parser = argparse.ArgumentParser(description="RARP: Combined RAP + SQLCoder approach for Text-to-SQL")
    parser.add_argument("--db", required=True, help="Database ID from Spider dataset")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--model", default="llama3-70b-8192", help="Groq model to use")
    parser.add_argument("--no-samples", action="store_true", help="Don't include sample data in context")
    args = parser.parse_args()
    
    print(f"Generating SQL for query: {args.query}")
    print(f"Using database: {args.db}")
    print(f"Using model: {args.model}")
    
    rarp = RARP(args.db, args.model)
    result = rarp.generate_sql(args.query, not args.no_samples)
    
    print("\nGenerated SQL:")
    print(result["sql"])
    
    # Execute the SQL and show results
    try:
        results, columns = rarp.execute_sql(result["sql"])
        
        print("\nSQL Execution Results:")
        print(" | ".join(columns))
        print("-" * (sum(len(col) for col in columns) + len(columns) * 3))
        
        for row in results[:10]:  # Limit to 10 rows for display
            print(" | ".join(str(val) for val in row))
        
        if len(results) > 10:
            print(f"... (showing 10/{len(results)} rows)")
    
    except Exception as e:
        print(f"\nError executing SQL: {str(e)}")


if __name__ == "__main__":
    main()
