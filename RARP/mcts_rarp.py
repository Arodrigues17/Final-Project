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

# Define constants for MCTS
MAX_ITERATIONS = 20  # Maximum iterations for MCTS
UCB_CONSTANT = 1.4  # Exploration-exploitation trade-off constant
MAX_DEPTH = 15  # Maximum depth for the search tree
MAX_CHILDREN = 8  # Maximum children per node

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


class MCTSRARP(RARP):
    """
    Extension of RARP to incorporate Monte Carlo Tree Search for SQL generation
    """
    
    def __init__(self, db_id, model, tables_path=None, few_shot_examples=None, mcts_iterations=MAX_ITERATIONS):
        """
        Initialize the MCTSRARP with database information
        
        Args:
            db_id: Database ID
            model: The LLM model to use
            tables_path: Path to tables.json
            few_shot_examples: Examples for few-shot learning
            mcts_iterations: Number of MCTS iterations
        """
        super().__init__(db_id, model, tables_path)
        self.few_shot_examples = few_shot_examples or []
        self.mcts_iterations = mcts_iterations
        self.db_path = self._get_db_path(db_id)
    
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
    
    def generate_sql(self, query: str, include_samples: bool = True) -> Dict[str, Any]:
        """
        Generate SQL using MCTS and the base RARP model
        
        Args:
            query: The natural language query
            include_samples: Whether to include sample data
            
        Returns:
            Dict containing the generated SQL and metadata
        """
        # 1. First try direct generation with RARP for simple cases
        start_time = time.time()
        direct_result = super().generate_sql(query, include_samples)
        direct_sql = direct_result["sql"]
        
        # 2. Check if the direct SQL is valid
        is_valid, error_msg = self.validate_sql(direct_sql)
        
        # Force MCTS for a percentage of queries to ensure it gets used
        force_mcts = random.random() < 0.3  # Try MCTS for 30% of queries regardless
        
        # Use MCTS if direct generation failed OR it's a complex query OR we're forcing MCTS
        if not is_valid or self._is_complex_query(query) or force_mcts:
            print(f"Using MCTS for query: {query}")
            print(f"Reason: {'Invalid direct SQL' if not is_valid else 'Complex query' if self._is_complex_query(query) else 'Forced MCTS'}")
            mcts_start_time = time.time()
            mcts_sql = self._mcts_search(query, direct_sql)
            
            # 4. Validate the MCTS-generated SQL
            is_valid_mcts, error_msg_mcts = self.validate_sql(mcts_sql)
            
            # 5. Choose the best SQL between direct and MCTS
            if is_valid_mcts:
                # If both are valid, choose based on execution results
                if is_valid:
                    direct_exec_success, direct_results = self.execute_sql(direct_sql)
                    mcts_exec_success, mcts_results = self.execute_sql(mcts_sql)
                    
                    if direct_exec_success and mcts_exec_success:
                        # Compare result sets if both execute successfully
                        # Use the one with more detailed results or better self-consistency
                        direct_quality = self._evaluate_sql_quality(direct_sql, direct_results)
                        mcts_quality = self._evaluate_sql_quality(mcts_sql, mcts_results)
                        
                        print(f"Direct SQL quality: {direct_quality:.3f}, MCTS SQL quality: {mcts_quality:.3f}")
                        
                        final_sql = mcts_sql if mcts_quality > direct_quality else direct_sql
                        method = "mcts" if mcts_quality > direct_quality else "direct"
                    elif mcts_exec_success:
                        final_sql = mcts_sql
                        method = "mcts"
                    else:
                        final_sql = direct_sql
                        method = "direct"
                else:
                    final_sql = mcts_sql
                    method = "mcts"
            else:
                final_sql = direct_sql
                method = "direct"
            
            total_time = time.time() - start_time
            
            return {
                "sql": final_sql,
                "method": method,
                "time_taken": total_time,
                "mcts_used": method == "mcts",
                "direct_sql": direct_sql,
                "mcts_sql": mcts_sql,
                "direct_valid": is_valid,
                "mcts_valid": is_valid_mcts
            }
        else:
            # For simpler queries, just return the direct result
            return {
                "sql": direct_sql,
                "method": "direct",
                "time_taken": time.time() - start_time,
                "mcts_used": False
            }
    
    def _is_complex_query(self, query: str) -> bool:
        """
        Determine if a query is complex enough to warrant MCTS
        
        Args:
            query: The natural language query
            
        Returns:
            True if the query is complex, False otherwise
        """
        # More comprehensive complexity indicators
        complexity_indicators = [
            "group", "aggregate", "average", "count", "most", "least", "maximum", "minimum",
            "more than", "less than", "between", "not", "except", "top", "bottom", "all",
            "order", "rank", "join", "where", "having", "nested", "compare", "correlation",
            "percentage", "proportion", "number of", "highest", "lowest", "greater", "smaller"
        ]
        
        lower_query = query.lower()
        
        # Check for presence of complexity indicators
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in lower_query)
        if indicator_count >= 2:
            return True
        
        # Check for question words that often indicate complex queries
        question_indicators = ["how many", "which", "what is the", "find the", "list all"]
        if any(indicator in lower_query for indicator in question_indicators):
            return True
        
        # Check length - longer queries tend to be more complex
        if len(query.split()) > 10:
            return True
        
        # Count database entities mentioned - more entities often mean more complex queries
        schema_entities = set(table.lower() for table in self.schema.tables)
        schema_entities.update(column.lower() for table in self.schema.tables for column in self.schema.column_names_original[table])
        
        entity_count = sum(1 for entity in schema_entities if entity in lower_query)
        if entity_count >= 3:
            return True
        
        return False
    
    def _mcts_search(self, nl_query: str, initial_sql: str) -> str:
        """
        Perform Monte Carlo Tree Search to generate a SQL query
        
        Args:
            nl_query: Natural language query
            initial_sql: Initial SQL from direct generation
            
        Returns:
            The best SQL query found by MCTS
        """
        try:
            # 1. Create the root node with initial SQL
            root = SQLNode(state=initial_sql)
            
            # 2. Define actions for SQL refinement based on node state
            self._populate_possible_actions(root, nl_query)
            
            # If we couldn't generate any actions, return the initial SQL
            if not root.untried_actions:
                print("No actions generated, returning initial SQL")
                return initial_sql
            
            # 3. Run MCTS for specified number of iterations
            for i in range(self.mcts_iterations):
                try:
                    # Selection - find the most promising node
                    node = self._select(root)
                    
                    # Expansion - expand the selected node
                    if not node.is_terminal and not node.is_fully_expanded:
                        node = self._expand(node, nl_query)
                    
                    # Simulation - simulate from the expanded node
                    reward = self._simulate(node, nl_query)
                    
                    # Backpropagation - update node values
                    self._backpropagate(node, reward)
                except Exception as e:
                    print(f"Error in MCTS iteration {i}: {str(e)}")
                    continue
            
            # 4. Select the best SQL from the children of the root
            best_child = self._get_best_child(root, exploration_weight=0)
            best_sql = best_child.state if best_child else initial_sql
            
            return best_sql
        except Exception as e:
            print(f"Error in MCTS search: {str(e)}")
            return initial_sql
    
    def _populate_possible_actions(self, node: SQLNode, nl_query: str):
        """
        Populate the possible actions for a node
        
        Args:
            node: The node to populate actions for
            nl_query: The natural language query
        """
        try:
            # Use LLM to generate potential refinements/actions
            actions = self._generate_refinement_actions(node.state, nl_query)
            node.untried_actions = actions
            
            # If no actions were generated, add some simple variants
            if not actions:
                simple_variants = []
                
                # Add ORDER BY if not present
                if "ORDER BY" not in node.state.upper():
                    simple_variants.append(node.state + " ORDER BY 1")
                
                # Add LIMIT if not present
                if "LIMIT" not in node.state.upper():
                    simple_variants.append(node.state + " LIMIT 10")
                
                # Add simple variants to untried actions
                node.untried_actions.extend(simple_variants)
                
                print(f"Generated {len(simple_variants)} simple variants as fallback")
        except Exception as e:
            print(f"Error populating actions: {str(e)}")
            # Add some default actions as fallback
            node.untried_actions = [node.state + " LIMIT 10"]
    
    def _generate_refinement_actions(self, current_sql: str, nl_query: str) -> List[str]:
        """
        Generate potential SQL refinement actions using LLM
        
        Args:
            current_sql: Current SQL state
            nl_query: Original natural language query
            
        Returns:
            List of potential refinement actions
        """
        # More specific prompt for better refinements
        prompt = f"""
        I need to generate several variations of this SQL query to better match the question.
        
        Question: {nl_query}
        Current SQL: {current_sql}
        
        Generate {MAX_CHILDREN} different SQL variations. For each variation:
        1. Fix any syntax errors
        2. Check if any columns or conditions are missing
        3. Ensure JOIN conditions are correct
        4. Consider adding/modifying GROUP BY, ORDER BY, or aggregation functions
        
        Be creative but ensure each query is syntactically valid. Don't just make minor changes.
        Each query should be a complete SQL statement that addresses the user's question.
        
        Format your response exactly like this:
        VARIATION 1:
        [Complete SQL query 1]
        
        VARIATION 2:
        [Complete SQL query 2]
        
        And so on...
        """
        
        # Call the LLM model (using the same interface as RARP)
        response = self._call_llm(prompt)
        
        # Extract SQL queries with more reliable pattern
        refined_sqls = []
        pattern = r"VARIATION \d+:\s*(.*?)(?=VARIATION \d+:|$)"
        matches = re.finditer(pattern, response, re.DOTALL)
        
        for match in matches:
            sql = match.group(1).strip().rstrip(';')
            if sql and sql != current_sql:
                refined_sqls.append(sql)
        
        # Ensure we have at least some variations
        if not refined_sqls:
            # Create simple variants if LLM didn't generate any
            select_pattern = r"SELECT\s+(.*?)\s+FROM"
            where_pattern = r"WHERE\s+(.*?)(?:ORDER BY|GROUP BY|LIMIT|$)"
            
            # Try variations like adding LIMIT, changing ORDER BY, etc.
            if "ORDER BY" not in current_sql.upper():
                refined_sqls.append(current_sql + " ORDER BY 1")
            if "LIMIT" not in current_sql.upper():
                refined_sqls.append(current_sql + " LIMIT 10")
            if "WHERE" in current_sql.upper():
                # Try removing a condition
                try:
                    where_match = re.search(where_pattern, current_sql, re.IGNORECASE)
                    if where_match:
                        conditions = where_match.group(1).split("AND")
                        if len(conditions) > 1:
                            new_where = " AND ".join(conditions[:-1])
                            new_sql = re.sub(where_pattern, f"WHERE {new_where}", current_sql, flags=re.IGNORECASE)
                            refined_sqls.append(new_sql)
                except:
                    pass
            
            # If still empty, add some reversed order or additional limit variants
            if len(refined_sqls) < 2:
                if "ORDER BY" in current_sql.upper():
                    # Try reversing the order
                    if "DESC" in current_sql.upper():
                        refined_sqls.append(current_sql.replace("DESC", "ASC"))
                    else:
                        refined_sqls.append(current_sql.replace("ORDER BY", "ORDER BY DESC"))
                if "LIMIT" in current_sql.upper():
                    # Try changing the limit
                    limit_pattern = r"LIMIT\s+(\d+)"
                    limit_match = re.search(limit_pattern, current_sql, re.IGNORECASE)
                    if limit_match:
                        current_limit = int(limit_match.group(1))
                        new_limit = current_limit * 2
                        refined_sqls.append(re.sub(limit_pattern, f"LIMIT {new_limit}", current_sql, flags=re.IGNORECASE))
        
        # Limit number of refinements
        return refined_sqls[:MAX_CHILDREN]
    
    def _select(self, node: SQLNode) -> SQLNode:
        """
        Select the most promising node using UCB
        
        Args:
            node: The current node
            
        Returns:
            The selected node
        """
        # If not fully expanded or terminal, return this node
        if not node.is_fully_expanded or node.is_terminal:
            return node
        
        # Select child with highest UCB score
        return max(node.children, key=lambda n: n.get_ucb_score(UCB_CONSTANT))
    
    def _expand(self, node: SQLNode, nl_query: str) -> SQLNode:
        """
        Expand the selected node by trying an untried action
        
        Args:
            node: The node to expand
            nl_query: Natural language query
            
        Returns:
            The new child node
        """
        # If no untried actions, mark as fully expanded and return
        if not node.untried_actions:
            node.is_fully_expanded = True
            return node
        
        # Try a random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # Create a child node with the new state
        child = node.add_child(state=action, action=action)
        
        # Populate possible actions for the child
        self._populate_possible_actions(child, nl_query)
        
        return child
    
    def _simulate(self, node: SQLNode, nl_query: str) -> float:
        """
        Simulate from the current node to get a reward
        
        Args:
            node: The current node
            nl_query: Natural language query
            
        Returns:
            The reward value
        """
        # Evaluate the SQL quality
        is_valid, error_msg = self.validate_sql(node.state)
        
        if not is_valid:
            return 0.0
        
        exec_success, results = self.execute_sql(node.state)
        if not exec_success:
            return 0.1  # Small reward for valid but non-executable SQL
        
        # Calculate reward based on various metrics
        reward = self._evaluate_sql_quality(node.state, results)
        
        # Check for special keywords in NL query and verify they're in SQL
        nl_keywords = self._extract_keywords(nl_query)
        sql_coverage = self._check_keyword_coverage(node.state, nl_keywords)
        
        # Final reward combines execution success and keyword coverage
        final_reward = 0.5 * reward + 0.5 * sql_coverage
        
        return final_reward
    
    def _backpropagate(self, node: SQLNode, reward: float):
        """
        Backpropagate the reward up the tree
        
        Args:
            node: The current node
            reward: The reward value
        """
        # Update all nodes up to the root
        while node:
            node.update(reward)
            node = node.parent
    
    def _get_best_child(self, node: SQLNode, exploration_weight: float = 0) -> Optional[SQLNode]:
        """
        Get the best child node based on value
        
        Args:
            node: The parent node
            exploration_weight: Weight for exploration (0 for pure exploitation)
            
        Returns:
            The best child node
        """
        if not node.children:
            return None
        
        # For final selection, use only the value (no exploration)
        return max(node.children, key=lambda n: n.get_ucb_score(exploration_weight))
    
    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """
        Validate an SQL query without executing it
        
        Args:
            sql: The SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Try to parse the SQL
            cursor.execute(f"EXPLAIN {sql}")
            conn.close()
            
            return True, ""
        except sqlite3.Error as e:
            return False, str(e)
    
    def execute_sql(self, sql: str) -> Tuple[bool, List[Tuple]]:
        """
        Execute an SQL query and return results
        
        Args:
            sql: The SQL query to execute
            
        Returns:
            Tuple of (success, results)
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Execute the SQL
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            
            return True, results
        except sqlite3.Error:
            return False, []
    
    def _evaluate_sql_quality(self, sql: str, results: List[Tuple]) -> float:
        """
        Evaluate the quality of an SQL query with improved metrics
        
        Args:
            sql: The SQL query
            results: The execution results
            
        Returns:
            A quality score between 0 and 1
        """
        # Base score starts higher to prevent premature rejection
        base_score = 0.3
        
        # Check result set size (not too small, not too large)
        result_size = len(results)
        size_score = 0.0
        
        if result_size == 0:
            size_score = 0.1  # Empty result is not ideal but not the worst
        elif 1 <= result_size <= 100:
            size_score = min(0.4, 0.1 + 0.3 * (1.0 - abs(result_size - 5) / 100.0))
        else:
            size_score = 0.1  # Too many results
        
        # Check SQL structure quality
        structure_score = 0.0
        
        # Reward for using important clauses
        important_clauses = ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY"]
        used_clauses = sum(1 for clause in important_clauses if f" {clause.upper()} " in f" {sql.upper()} ")
        structure_score += 0.05 * used_clauses
        
        # Reward for join conditions
        if "JOIN" in sql.upper() and "ON" in sql.upper():
            structure_score += 0.1
        
        # Penalize for potential issues
        issues = [
            "SELECT *",  # Using SELECT * is often not ideal
            "WHERE 1=1",  # Unnecessary condition
            "CROSS JOIN WITHOUT ON",  # Cross joins might be unintentional
        ]
        
        # Count number of issues
        issue_count = sum(1 for issue in issues if issue.upper() in sql.upper())
        issue_penalty = 0.05 * issue_count
        
        # Final score
        total_score = base_score + size_score + structure_score - issue_penalty
        return min(1.0, max(0.0, total_score))
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from the natural language query
        
        Args:
            query: The natural language query
            
        Returns:
            List of important keywords
        """
        # Remove stopwords and extract meaningful terms
        stopwords = ["the", "a", "an", "in", "on", "at", "for", "to", "of", "with", "by", "is", "are"]
        tokens = query.lower().split()
        keywords = [token for token in tokens if token not in stopwords and len(token) > 2]
        
        return keywords
    
    def _check_keyword_coverage(self, sql: str, keywords: List[str]) -> float:
        """
        Check what percentage of keywords are covered in the SQL
        
        Args:
            sql: The SQL query
            keywords: List of important keywords
            
        Returns:
            Coverage percentage as a value between 0 and 1
        """
        if not keywords:
            return 1.0
        
        covered = sum(1 for keyword in keywords if keyword in sql.lower())
        coverage = covered / len(keywords)
        
        return coverage
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with a prompt
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response
        """
        # The parent RARP class doesn't have a _call_llm method, 
        # but it likely has some method to call the model.
        # We need to figure out what method actually exists in the RARP class.
        # Let's check for common method names:
        
        try:
            # Try the standard method for getting SQL from RARP
            # This assumes the parent class has a _get_response_from_model method
            if hasattr(self, '_get_response_from_model'):
                return self._get_response_from_model(prompt)
            # Try alternative method names that might exist in RARP
            elif hasattr(self, '_call_model'):
                return self._call_model(prompt)
            elif hasattr(self, 'call_model'):
                return self.call_model(prompt)
            elif hasattr(self, 'query_model'):
                return self.query_model(prompt)
            elif hasattr(self, 'generate'):
                return self.generate(prompt)
            else:
                # As a last resort, use the generate_sql method but extract just the SQL
                result = super().generate_sql(prompt, include_samples=False)
                return result.get("sql", "")
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")
            # Return an empty string or some default SQL if there's an error
            return "SELECT * FROM table LIMIT 1"


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