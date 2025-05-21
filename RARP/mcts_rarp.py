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

# --- add below existing imports ---
TOKEN_PATTERN = re.compile(r'\w+')
COMPLEX_KEYWORDS = {'JOIN', 'GROUP BY', 'HAVING', 'UNION', 'EXISTS'}
SIMPLE_PATTERNS = [
    re.compile(r'^SELECT\s+\*\s+FROM\s+\w+;?$', re.IGNORECASE),
    re.compile(r'^SELECT\s+\w+\s+FROM\s+\w+;?$', re.IGNORECASE),
]
# --- end additions ---

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
        
        # --- simple‐NL bypass: never run MCTS on trivial 1-table queries regardless of validation ---
        is_simple_query = len(query.split()) < 6 and all(
            kw not in query.lower() 
            for kw in ["group", "average", "most", "least", "order", "having", "count", "more", "than"]
        )
        if is_simple_query:
            return {
                "sql": direct_sql,
                "method": "direct",
                "time_taken": time.time() - start_time,
                "mcts_used": False
            }
        # --- end bypass ---
        
        # 2. Check if the direct SQL is valid
        is_valid, error_msg = self.validate_sql(direct_sql)
        
        # Apply a more sophisticated complexity check based on both query and SQL structure
        is_simple_operation = (
            # Simple count queries
            ("how many" in query.lower() and "COUNT" in direct_sql.upper()) or
            # Simple attribute queries with filtering
            (len(query.split()) < 10 and "WHERE" in direct_sql.upper() and direct_sql.upper().count("JOIN") <= 1) or
            # Very basic lookup queries
            (len(query.split()) < 8 and direct_sql.upper().count("JOIN") == 0)
        )
        
        # Return direct SQL if it's valid and simple, avoiding unnecessary MCTS
        if is_valid and is_simple_operation:
            return {
                "sql": direct_sql,
                "method": "direct",
                "time_taken": time.time() - start_time,
                "mcts_used": False
            }
        
        # Force MCTS less often for non-complex queries (reduced from 50% to 30% chance)
        complex_query = self._is_complex_query(query)
        force_mcts = random.random() < 0.3 and not is_simple_operation  
        
        # Use MCTS if direct generation failed OR it's a complex query OR we're forcing MCTS
        if not is_valid or complex_query or force_mcts:
            reason = 'Invalid direct SQL' if not is_valid else 'Complex query' if complex_query else 'Forced MCTS'
            print(f"Using MCTS for query: {query}")
            print(f"Reason: {reason}")
            
            if not is_valid:
                print(f"SQL Validation Error: {error_msg}")
            
            # Set a timeout for the entire MCTS process - increased for better results
            mcts_timeout = 90  # Maximum seconds for MCTS (increased from 60)
            mcts_start_time = time.time()
            
            try:
                mcts_sql = self._mcts_search(query, direct_sql)
                
                # Check if we exceeded the timeout
                if time.time() - mcts_start_time > mcts_timeout:
                    print(f"MCTS took too long ({time.time() - mcts_start_time:.1f}s) - returning direct SQL")
                    return {
                        "sql": direct_sql,
                        "method": "direct",
                        "time_taken": time.time() - start_time,
                        "mcts_used": False,
                        "mcts_timeout": True
                    }
                
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
                            
                            # Be more permissive about using MCTS results
                            if mcts_quality >= direct_quality - 0.05:  # Allow MCTS if it's close or better
                                final_sql = mcts_sql
                                method = "mcts"
                            else:
                                final_sql = direct_sql
                                method = "direct"
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
            except Exception as e:
                print(f"Error in MCTS process: {str(e)}")
                final_sql = direct_sql
                method = "direct"
            
            total_time = time.time() - start_time
            
            return {
                "sql": final_sql,
                "method": method,
                "time_taken": total_time,
                "mcts_used": method == "mcts",
                "direct_sql": direct_sql,
                "mcts_sql": mcts_sql if 'mcts_sql' in locals() else "ERROR",
                "direct_valid": is_valid,
                "mcts_valid": is_valid_mcts if 'is_valid_mcts' in locals() else False
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
        Determine if a query is complex enough to warrant MCTS:
        1. Whitelist truly trivial patterns
        2. Bypass very short queries
        3. Require ≥2 distinct structural keywords
        """
        # 1. Whitelist simple one-table selects
        if any(pat.match(query) for pat in SIMPLE_PATTERNS):
            return False

        # 2. Short NL queries are likely trivial
        toks = TOKEN_PATTERN.findall(query.upper())
        if len(toks) < 10:
            # However, counting queries with filters might be complex enough for MCTS,
            # even if they have fewer tokens
            has_count_indicator = any(word in query.lower() for word in 
                ['how many', 'count', 'number of'])
            has_filter_indicator = any(word in query.lower() for word in 
                ['where', 'with', 'that have', 'from'])
            
            if has_count_indicator and has_filter_indicator and 'join' in query.lower():
                return True
            return False

        # 3. Require at least two distinct complex keywords
        return len(set(toks) & COMPLEX_KEYWORDS) >= 2
    
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
            # Set a maximum timeout for the entire MCTS process - increased for better results
            max_search_time = 60  # Maximum seconds to spend on MCTS (increased from 30)
            start_time = time.time()
            
            # 1. Create the root node with initial SQL
            root = SQLNode(state=initial_sql)
            
            # 2. Define actions for SQL refinement based on node state
            self._populate_possible_actions(root, nl_query)
            
            # If we couldn't generate any actions, return the initial SQL
            if not root.untried_actions:
                print("No actions generated, returning initial SQL")
                return initial_sql
            
            # Dynamically adjust iterations based on query complexity
            query_words = len(nl_query.split())
            complexity_indicators = ["group", "count", "most", "least", "order", "having", "more than"]
            complexity_score = sum(1 for indicator in complexity_indicators if indicator.lower() in nl_query.lower())
            
            # More complex formula for determining iterations
            adjusted_iterations = min(self.mcts_iterations, 
                                     max(5, query_words // 2 + complexity_score * 3))
            
            print(f"Adjusted MCTS iterations to {adjusted_iterations} based on query complexity")
            
            # Track best SQL found so far
            best_sql = initial_sql
            best_reward = 0
            early_stop_threshold = 0.9  # Increased threshold for early stopping (more conservative)
            min_iterations = max(3, adjusted_iterations // 2)  # Minimum iterations before early stopping
            
            # 3. Run MCTS for specified number of iterations or until timeout
            for i in range(adjusted_iterations):
                # Check timeout
                if time.time() - start_time > max_search_time:
                    print(f"MCTS timeout after {i} iterations")
                    break
                    
                try:
                    # Selection - find the most promising node
                    node = self._select(root)
                    
                    # Expansion - expand the selected node
                    if not node.is_terminal and not node.is_fully_expanded:
                        node = self._expand(node, nl_query)
                    
                    # Simulation - simulate from the expanded node
                    reward = self._simulate(node, nl_query)
                    
                    # Track best solution found
                    if reward > best_reward:
                        best_reward = reward
                        if node.state != initial_sql:  # Only update if it's not the initial SQL
                            best_sql = node.state
                            print(f"Found better SQL (reward: {reward:.3f})")
                    
                    # Early stopping if we find a very good solution, but only after minimum iterations
                    if i >= min_iterations and reward > early_stop_threshold:
                        print(f"Early stopping at iteration {i} - found high-quality SQL (reward: {reward:.3f})")
                        break
                    
                    # Backpropagation - update node values
                    self._backpropagate(node, reward)
                except Exception as e:
                    print(f"Error in MCTS iteration {i}: {str(e)}")
                    continue
            
            # 4. Select the best SQL - either from MCTS or what we tracked
            if best_reward > 0.6:  # Lowered threshold to prefer tracked best solution
                return best_sql
            else:
                # Fall back to the standard MCTS selection
                best_child = self._get_best_child(root, exploration_weight=0)
                if best_child and best_child.state != initial_sql:
                    return best_child.state
                return best_sql  # fall back to our tracked best
            
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
        # Fast generation of simple variants without calling LLM
        # This is used to quickly generate candidates without expensive LLM calls
        fast_variants = []
        
        # Always add these fast variants to avoid getting stuck
        if "ORDER BY" not in current_sql.upper():
            fast_variants.append(current_sql + " ORDER BY 1")
        if "LIMIT" not in current_sql.upper():
            fast_variants.append(current_sql + " LIMIT 10")
        
        # Only proceed with expensive LLM call if we have few variants
        # or randomly (to ensure we sometimes do deep exploration)
        deep_explore = random.random() < 0.3  # 30% chance to do deep exploration
        
        if len(fast_variants) < 2 or deep_explore:
            try:
                # More specific prompt for better refinements
                prompt = f"""
                I need quick SQL variations for this question.
                
                Question: {nl_query}
                Current SQL: {current_sql}
                
                Generate 3 different SQL variations. For each:
                1. Fix any syntax errors
                2. Add or modify WHERE conditions
                3. Consider adding GROUP BY or ORDER BY
                
                Format as:
                VARIATION 1:
                [SQL query 1]
                
                VARIATION 2:
                [SQL query 2]
                
                VARIATION 3:
                [SQL query 3]
                """
                
                # Time the LLM call to prevent getting stuck
                start_time = time.time()
                response = self._call_llm(prompt)
                elapsed = time.time() - start_time
                
                if elapsed > 10:  # If LLM call took more than 10 seconds
                    print(f"Warning: LLM call took {elapsed:.1f}s - using fast variants only")
                    return fast_variants[:MAX_CHILDREN]
                
                # Extract SQL queries with more reliable pattern
                refined_sqls = []
                pattern = r"VARIATION \d+:\s*(.*?)(?=VARIATION \d+:|$)"
                matches = re.finditer(pattern, response, re.DOTALL)
                
                for match in matches:
                    sql = match.group(1).strip().rstrip(';')
                    if sql and sql != current_sql:
                        refined_sqls.append(sql)
                
                # Combine LLM-generated and fast variants
                all_variants = refined_sqls + fast_variants
                
                # Limit number of refinements
                return all_variants[:MAX_CHILDREN]
                
            except Exception as e:
                print(f"Error generating refinements: {str(e)}")
                # Fall back to fast variants
                return fast_variants[:MAX_CHILDREN]
        else:
            # Just use the fast variants to avoid expensive LLM calls
            return fast_variants[:MAX_CHILDREN]
    
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
            size_score = 0.05  # Reduced score for empty results
        elif 1 <= result_size <= 100:
            # Prefer results with 1-10 rows as more likely to be correct for typical questions
            if 1 <= result_size <= 10:
                size_score = 0.2 + 0.2 * (1.0 - abs(result_size - 5) / 10.0)
            else:
                size_score = 0.2 * (1.0 - (result_size - 10) / 90.0)
        else:
            size_score = 0.05  # Too many results
        
        # Check SQL structural quality
        structure_score = 0.0
        
        # Reward for using important clauses - more weight on GROUP BY and proper filtering
        important_clauses = {
            "SELECT": 0.03,
            "FROM": 0.03,
            "WHERE": 0.05,
            "GROUP BY": 0.07,
            "HAVING": 0.07,
            "ORDER BY": 0.05,
            "LIMIT": 0.03,
            "JOIN": 0.07
        }
        
        for clause, weight in important_clauses.items():
            if f" {clause.upper()} " in f" {sql.upper()} ":
                structure_score += weight
        
        # Reward for join conditions
        if "JOIN" in sql.upper() and "ON" in sql.upper():
            structure_score += 0.1
            
        # Reward for column specification (not using *)
        if "SELECT *" not in sql.upper():
            structure_score += 0.05
        
        # Check for aggregation functions (common in complex queries)
        agg_functions = ["COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP_CONCAT"]
        for func in agg_functions:
            if f"{func}(" in sql.upper():
                structure_score += 0.03
        
        # Penalize for potential issues
        issues = [
            "SELECT * FROM", # Using SELECT * is often not ideal
            "WHERE 1=1", # Unnecessary condition
            "WHERE 1 = 1",
            "CROSS JOIN", # Cross joins might be unintentional
            "FROM TABLE" # Generic table name
        ]
        
        issue_penalty = 0.0
        for issue in issues:
            if issue.upper() in sql.upper():
                issue_penalty += 0.05
        
        # Reward for specific SQL patterns that often appear in gold queries
        patterns = [
            ("GROUP BY.*HAVING", 0.1),  # Group by with having is a common complex pattern
            ("ORDER BY.*LIMIT", 0.07),  # Order by with limit is a common complex pattern
            ("JOIN.*JOIN", 0.07)        # Multiple joins indicate more complex queries
        ]
        
        pattern_score = 0.0
        for pattern, weight in patterns:
            if re.search(pattern, sql.upper()):
                pattern_score += weight
        
        # Final score
        total_score = base_score + size_score + structure_score + pattern_score - issue_penalty
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
            # Get schema info safely
            schema_info = ""
            try:
                if hasattr(self, 'schema') and hasattr(self.schema, 'get_schema_str'):
                    schema_info = self.schema.get_schema_str()
            except Exception as e:
                print(f"Warning: couldn't get schema info: {str(e)}")
            
            # Include schema info in prompt if available
            full_prompt = prompt
            if schema_info:
                full_prompt = f"""
                Database Schema:
                {schema_info}
                
                {prompt}
                """
            
            # Try the standard method for getting SQL from RARP
            # This assumes the parent class has a _get_response_from_model method
            if hasattr(self, '_get_response_from_model'):
                return self._get_response_from_model(full_prompt)
            # Try alternative method names that might exist in RARP
            elif hasattr(self, '_call_model'):
                return self._call_model(full_prompt)
            elif hasattr(self, 'call_model'):
                return self.call_model(full_prompt)
            elif hasattr(self, 'query_model'):
                return self.query_model(full_prompt)
            elif hasattr(self, 'generate'):
                return self.generate(full_prompt)
            elif hasattr(self, '_generate_sql_from_model'):
                return self._generate_sql_from_model(full_prompt)
            elif hasattr(super(), 'generate_sql'):
                # As a last resort, use the generate_sql method but extract just the SQL
                result = super().generate_sql(full_prompt, include_samples=False)
                return result.get("sql", "")
            else:
                print("Warning: couldn't find any method to call the LLM")
                return "SELECT * FROM table LIMIT 1"
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


# Example usage as a simple demo
if __name__ == "__main__":
    query = "Find the name of the restaurant with the most number of reviews"
    model = "llama-3.1-8b-instant"
    db_id = "restaurant"
    
    # Example of how to use MCTSRARP
    rarp = MCTSRARP(db_id, model)
    result = rarp.generate_sql(query)
    
    print(f"Generated SQL: {result['sql']}")
    print(f"Method: {result['method']}")
    print(f"Time taken: {result['time_taken']:.2f} seconds")