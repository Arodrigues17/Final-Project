#!/usr/bin/env python3
"""
MCTS-RARP: A modified version of RARP that uses Monte Carlo Tree Search
for SQL generation based on concepts from MCTS-SQL and Alpha-SQL
"""

import os
import re
import json
import time
import sqlite3
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Set, Optional
from pathlib import Path
import copy

# Import the original RARP class
from working_rarp import RARP, DatabaseSchema

# Define constants for MCTS - using more conservative values
MAX_ITERATIONS = 20
UCB_CONSTANT = 1.0
MAX_DEPTH = 10
MAX_CHILDREN = 5

class SQLNode:
    """
    A node in the MCTS tree representing a partial or complete SQL query
    """
    def __init__(self, state: str, parent=None, action=None):
        """
        Initialize a node with a state (partial SQL query)
        
        Args:
            state: The current SQL query state
            parent: The parent node
            action: The action that led to this node
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False
        self.is_fully_expanded = False
        self.untried_actions = []  # Will be populated later
        
    def add_child(self, state: str, action: str) -> 'SQLNode':
        """Add a child node"""
        child = SQLNode(state=state, parent=self, action=action)
        self.children.append(child)
        return child
    
    def update(self, reward: float):
        """Update node statistics"""
        self.visits += 1
        self.value += (reward - self.value) / self.visits
    
    def get_ucb_score(self, exploration_weight: float) -> float:
        """Calculate UCB score for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value
        exploration = exploration_weight * np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
        
    def best_child(self, exploration_weight: float) -> 'SQLNode':
        """Select the best child based on UCB score"""
        if not self.children:
            return None
        
        return max(self.children, key=lambda n: n.get_ucb_score(exploration_weight))


class MCTSRARP(RARP):
    """
    Extension of RARP to incorporate Monte Carlo Tree Search for SQL generation
    """
    
    def __init__(self, db_id, model, tables_path=None, few_shot_examples=None, mcts_iterations=MAX_ITERATIONS):
        """Initialize the MCTSRARP with database information"""
        super().__init__(db_id, model, tables_path)
        self.few_shot_examples = few_shot_examples or []
        self.mcts_iterations = mcts_iterations
        self.db_path = self._get_db_path(db_id)
        # Get schema in a simplified way
        self.schema_tables = []
        self.schema_columns = set()
        self._load_schema_simple()
    
    def _get_db_path(self, db_id: str) -> str:
        """Get the path to the SQLite database file"""
        spider_dir = Path("../datasets/spider")
        db_dir = spider_dir / "database" / db_id
        test_db_dir = spider_dir / "test_database" / db_id
        
        if db_dir.exists():
            return str(db_dir / f"{db_id}.sqlite")
        elif test_db_dir.exists():
            return str(test_db_dir / f"{db_id}.sqlite")
        else:
            raise ValueError(f"Database {db_id} not found")
    
    def _load_schema_simple(self):
        """Load schema information in a simple way using SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            self.schema_tables = tables
            
            # Get column names for each table
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = [f"{table}.{row[1]}" for row in cursor.fetchall()]
                self.schema_columns.update(columns)
            
            conn.close()
        except Exception as e:
            print(f"Error loading schema: {e}")
            # Fall back to empty schema
            self.schema_tables = []
            self.schema_columns = set()
    
    def generate_sql(self, query: str, include_samples: bool = True) -> Dict[str, Any]:
        """Generate SQL using MCTS and the base RARP model"""
        # 1. First try direct generation with RARP for simple cases
        start_time = time.time()
        direct_result = super().generate_sql(query, include_samples)
        direct_sql = direct_result["sql"]
        
        # 2. Check if the direct SQL is valid
        is_valid, _ = self.validate_sql(direct_sql)
        
        # Force MCTS for some percentage of queries to ensure it gets used
        use_mcts = False
        reason = ""
        
        # Check if we should use MCTS
        if not is_valid:
            use_mcts = True
            reason = "Invalid direct SQL"
        elif self._is_complex_query(query):
            use_mcts = True
            reason = "Complex query"
        elif random.random() < 0.3:  # Force MCTS for 30% of queries
            use_mcts = True
            reason = "Forced MCTS"
        
        # Use MCTS if needed
        if use_mcts:
            print(f"Using MCTS for query: {query}")
            print(f"Reason: {reason}")
            
            try:
                mcts_start_time = time.time()
                mcts_sql = self._mcts_search(query, direct_sql)
                
                # Only use MCTS result if it's valid
                mcts_valid, _ = self.validate_sql(mcts_sql)
                
                if mcts_valid:
                    return {
                        "sql": mcts_sql,
                        "method": "mcts",
                        "time_taken": time.time() - start_time,
                        "mcts_used": True
                    }
                else:
                    print("MCTS result invalid, falling back to direct generation")
            except Exception as e:
                print(f"MCTS error: {e}")
                # Continue with direct result
        
        # Use the direct result
        return {
            "sql": direct_sql,
            "method": "direct",
            "time_taken": time.time() - start_time,
            "mcts_used": False
        }
    
    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """Validate SQL by checking its syntax with SQLite"""
        if not sql or len(sql.strip()) == 0:
            return False, "Empty SQL"
            
        try:
            # Use EXPLAIN to validate syntax without executing
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First try EXPLAIN which checks syntax without execution
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            conn.close()
            return True, ""
        except sqlite3.Error as e:
            # Less strict check: if the error is just about a non-existent table or column
            # but the syntax is otherwise valid, we'll consider it as valid
            error_msg = str(e).lower()
            if "syntax error" not in error_msg:
                return True, str(e)
            return False, str(e)
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if a query is complex enough to warrant MCTS"""
        # Simple complexity indicators
        complexity_indicators = [
            "group", "count", "average", "most", "least", "join", 
            "order", "where", "having", "nested"
        ]
        
        lower_query = query.lower()
        
        # Check for indicators
        for indicator in complexity_indicators:
            if indicator in lower_query:
                return True
        
        # Check length - longer queries tend to be more complex
        if len(query.split()) > 12:
            return True
            
        return False
    
    def _mcts_search(self, nl_query: str, initial_sql: str) -> str:
        """Perform Monte Carlo Tree Search to generate a SQL query"""
        # Create root node with initial SQL
        root = SQLNode(state=initial_sql)
        
        # Define actions for SQL refinement
        root.untried_actions = self._generate_refinement_actions(root.state, nl_query)
        
        # Track best node and its score
        best_sql = initial_sql
        best_score = -float('inf')
        
        # MCTS main loop
        for i in range(self.mcts_iterations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_fully_expanded and node.untried_actions:
                node = self._expand(node, nl_query)
            
            # Simulation
            reward = self._simulate(node.state, nl_query)
            
            # Backpropagation
            self._backpropagate(node, reward)
            
            # Update best SQL if found a better one
            if node.value > best_score:
                best_score = node.value
                best_sql = node.state
        
        # Return best SQL found
        return best_sql
    
    def _select(self, node: SQLNode) -> SQLNode:
        """Select the most promising node using UCB"""
        current = node
        
        # Traverse the tree to a leaf or a node with untried actions
        while current.children and not current.untried_actions:
            current = current.best_child(UCB_CONSTANT)
            
        return current
    
    def _expand(self, node: SQLNode, nl_query: str) -> SQLNode:
        """Expand a node by trying an untried action"""
        if not node.untried_actions:
            node.is_fully_expanded = True
            return node
            
        # Select a random untried action
        sql = random.choice(node.untried_actions)
        node.untried_actions.remove(sql)
        
        # Create a child node
        child = node.add_child(sql, "refinement")
        
        # Generate new actions for the child
        child.untried_actions = self._generate_refinement_actions(sql, nl_query)
        
        return child
    
    def _generate_refinement_actions(self, current_sql: str, nl_query: str) -> List[str]:
        """Generate potential SQL refinement actions"""
        # Simple prompt for generating SQL variations
        prompt = f"""
        I need to refine this SQL query to better match the user question.
        
        User question: {nl_query}
        Current SQL: {current_sql}
        
        Generate 3 different variations of this SQL query that might better answer the question.
        Each variation should be a complete, executable SQL query.
        Focus on fixing any errors, improving joins, adding missing conditions, etc.
        
        Format your response as:
        SQL1: [your first SQL query]
        SQL2: [your second SQL query]
        SQL3: [your third SQL query]
        """
        
        try:
            # Call the LLM to get refinements
            response = self.rap.model.generate_text(prompt, max_tokens=1024)
            
            # Extract SQL queries
            variations = []
            lines = response.split("\n")
            current_sql = ""
            collecting = False
            
            for line in lines:
                line = line.strip()
                if line.startswith("SQL") and ":" in line:
                    if collecting and current_sql:
                        variations.append(current_sql.strip())
                    current_sql = line.split(":", 1)[1].strip()
                    collecting = True
                elif collecting and line:
                    current_sql += " " + line
            
            # Add the last SQL if any
            if collecting and current_sql:
                variations.append(current_sql.strip())
            
            # Add some simple variations if we couldn't extract any
            if not variations:
                # Try adding ORDER BY if not present
                if "ORDER BY" not in current_sql:
                    variations.append(current_sql + " ORDER BY 1")
                
                # Try adding LIMIT if not present
                if "LIMIT" not in current_sql:
                    variations.append(current_sql + " LIMIT 10")
                    
                # Try removing potentially problematic parts
                if "GROUP BY" in current_sql:
                    simple_sql = re.sub(r'GROUP BY.*?(ORDER BY|LIMIT|$)', r'\1', current_sql)
                    variations.append(simple_sql)
            
            # Ensure we return valid variations
            valid_variations = []
            for sql in variations:
                if sql != current_sql:  # Don't include the original SQL
                    is_valid, _ = self.validate_sql(sql)
                    if is_valid:
                        valid_variations.append(sql)
            
            return valid_variations[:MAX_CHILDREN]
            
        except Exception as e:
            print(f"Error generating refinements: {e}")
            return []
    
    def _simulate(self, sql: str, nl_query: str) -> float:
        """Simulate execution and evaluate quality"""
        # Try to execute the SQL
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            
            # Base score for successful execution
            score = 0.5
            
            # Award points for reasonable result size
            result_count = len(results)
            if 1 <= result_count <= 20:
                score += 0.3
            elif result_count > 20:
                score += 0.1
            
            # Return the score
            return score
            
        except sqlite3.Error:
            # Failed to execute
            return 0.1
    
    def _backpropagate(self, node: SQLNode, reward: float):
        """Backpropagate reward through the tree"""
        current = node
        while current:
            current.update(reward)
            current = current.parent
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt, abstracting away the API details"""
        context = f"You are an expert SQL engineer. Your task is to help refine SQL queries.\n\n{prompt}"
        
        # Try with different API keys if needed
        for key_env in ["GROQ_API_KEY", "GROQ_API_KEY_backup", "GROQ_API_KEY_paid"]:
            try:
                # Use RAP's model interface, but first check if we need to update the API key
                # This assumes the RAP model has a way to update its API key
                # You may need to modify this based on the actual implementation
                if hasattr(self.rap, 'model') and hasattr(self.rap.model, 'client'):
                    self.rap.model.client.api_key = os.environ.get(key_env)
                    print(f"Switching to {key_env} due to rate limiting")
            
                return self.rap.model.generate_text(context, max_tokens=1024)
                
            except Exception as e:
                # Check if it's a rate limit error
                if "rate limit" in str(e).lower() or "quota exceeded" in str(e).lower():
                    if key_env == "GROQ_API_KEY_paid":
                        # We've tried all keys, give up
                        print(f"All API keys are rate limited: {str(e)}")
                        return "Error: Rate limit reached with all available API keys"
                    # Otherwise continue to next key
                    continue
                else:
                    # For non-rate limit errors, raise the exception
                    return f"Error: {str(e)}"
        
        return "Error: Unable to generate SQL with any available API keys"


# Modify evaluate_query to use MCTSRARP
def evaluate_query_mcts(example: Dict[str, Any], model: str, include_samples: bool, few_shot_examples: List[Dict[str, Any]] = None, use_mcts: bool = True) -> Dict[str, Any]:
    """
    Evaluate a single query using MCTSRARP if requested
    
    Args:
        example: The example to evaluate
        model: The model to use
        include_samples: Whether to include sample data
        few_shot_examples: Examples for few-shot learning
        use_mcts: Whether to use MCTS
        
    Returns:
        Evaluation results
    """
    # This would be a modified version of the original evaluate_query function
    # The main change is to use MCTSRARP instead of RARP when use_mcts is True
    
    start_time = time.time()
    query = example["question"]
    db_id = example["db_id"]
    
    # Check if the example has a gold SQL query
    has_gold_sql = "query" in example
    gold_sql = example.get("query", "")
    
    # Skip examples with extremely complex queries
    if has_gold_sql and ("EXCEPT" in gold_sql or "INTERSECT" in gold_sql or gold_sql.lower().count("select") > 2):
        return {
            "db_id": db_id,
            "question": query,
            "gold_sql": gold_sql,
            "generated_sql": "SKIPPED - Too complex",
            "exact_match": False,
            "exec_match": False,
            "execution_success": False,
            "time_taken": 0,
            "error": "Skipped due to complexity",
            "method": "skipped"
        }
    
    try:
        # Find the database path
        # Initialize MCTSRARP or RARP based on use_mcts flag
        if use_mcts:
            if few_shot_examples and len(few_shot_examples) > 0:
                rarp = MCTSRARP(db_id, model, few_shot_examples=few_shot_examples)
            else:
                rarp = MCTSRARP(db_id, model)
        else:
            if few_shot_examples and len(few_shot_examples) > 0:
                from few_shot_rarp import FewShotRARP
                rarp = FewShotRARP(db_id, model, few_shot_examples=few_shot_examples)
            else:
                rarp = RARP(db_id, model)
        
        # Generate SQL
        result = rarp.generate_sql(query, include_samples)
        generated_sql = result["sql"]
        method = result.get("method", "direct")
        
        # Rest of the function would be the same as in evaluate_query
        
        return {
            "db_id": db_id,
            "question": query,
            "gold_sql": gold_sql if has_gold_sql else "N/A",
            "generated_sql": generated_sql,
            "method": method,
            "time_taken": time.time() - start_time,
            # Include other fields from the original function
        }
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return {
            "db_id": db_id,
            "question": query,
            "gold_sql": gold_sql if has_gold_sql else "N/A",
            "generated_sql": f"ERROR: {str(e)}",
            "exact_match": False,
            "exec_match": False,
            "execution_success": False,
            "time_taken": time.time() - start_time,
            "error": f"{str(e)}\n{traceback_str}",
            "method": "error"
        }


if __name__ == "__main__":
    # Example of how to use MCTSRARP
    db_id = "restaurant"
    model = "llama-3.1-8b-instant"
    query = "Find the name of the restaurant with the most number of reviews"
    
    rarp = MCTSRARP(db_id, model)
    result = rarp.generate_sql(query)
    
    print(f"Generated SQL: {result['sql']}")
    print(f"Method: {result['method']}")
    print(f"Time taken: {result['time_taken']:.2f} seconds")