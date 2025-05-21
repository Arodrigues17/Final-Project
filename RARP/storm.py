#!/usr/bin/env python3
"""
Lightweight MCTS-RARP: A selective Monte Carlo Tree Search approach for Text-to-SQL
Designed to be applied only to complex queries where it can add value
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

# Import the original RARP class
from working_rarp import RARP, DatabaseSchema

# Minimal constants for MCTS
MAX_ITERATIONS = 10      # Keep iterations low for speed
UCB_CONSTANT = 1.4       # Standard UCB constant
MAX_CHILDREN = 5         # Limit branching factor
MAX_SEARCH_TIME = 15     # Max seconds per MCTS search 

class SQLNode:
    """
    Simplified node in the MCTS tree representing a SQL query state
    """
    def __init__(self, state: str, parent=None, action=None):
        self.state = state        # SQL query
        self.parent = parent      # Parent node
        self.action = action      # Action that led to this node
        self.children = []        # Child nodes
        self.visits = 0           # Visit count
        self.value = 0.0          # Value estimate
        self.untried_actions = [] # Actions not yet explored
    
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


class SelectiveMCTSRARP(RARP):
    """
    Selective MCTS extension of RARP that only applies MCTS to complex queries
    """
    
    def __init__(self, db_id, model, tables_path=None, mcts_iterations=MAX_ITERATIONS):
        """Initialize with database info"""
        super().__init__(db_id, model, tables_path)
        self.mcts_iterations = mcts_iterations
        self.db_path = self._get_db_path(db_id)
    
    def _get_db_path(self, db_id: str) -> str:
        """Get path to the SQLite database file"""
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
        Generate SQL query, applying MCTS only when appropriate
        
        Args:
            query: The natural language query
            include_samples: Whether to include sample data
            
        Returns:
            Dict containing the generated SQL and metadata
        """
        # Start with direct generation
        start_time = time.time()
        direct_result = super().generate_sql(query, include_samples)
        direct_sql = direct_result["sql"]
        
        # Check if direct SQL is valid
        is_valid, error_msg = self.validate_sql(direct_sql)
        
        # Only use MCTS for truly complex queries with specific patterns
        # or when direct generation produced invalid SQL
        if (not is_valid) or self._is_highly_complex_query(query):
            print(f"Using MCTS for query: {query}")
            print(f"Reason: {'Invalid direct SQL' if not is_valid else 'Highly complex query'}")
            
            try:
                # Apply MCTS search with strict time limit
                mcts_start_time = time.time()
                mcts_sql = self._focused_mcts_search(query, direct_sql)
                mcts_time = time.time() - mcts_start_time
                
                # Only use MCTS result if it's valid and execution succeeds
                is_valid_mcts, error_msg_mcts = self.validate_sql(mcts_sql)
                
                if is_valid_mcts:
                    # Compare execution results
                    direct_exec_success, direct_results = self.execute_sql(direct_sql) if is_valid else (False, [])
                    mcts_exec_success, mcts_results = self.execute_sql(mcts_sql)
                    
                    # Only use MCTS if it's better
                    if (not direct_exec_success and mcts_exec_success) or (mcts_exec_success and self._is_better_result(mcts_results, direct_results)):
                        final_sql = mcts_sql
                        method = "mcts"
                    else:
                        final_sql = direct_sql
                        method = "direct"
                else:
                    final_sql = direct_sql
                    method = "direct"
            except Exception as e:
                print(f"MCTS error: {str(e)}")
                final_sql = direct_sql
                method = "direct"
        else:
            # For most queries, just use direct generation
            final_sql = direct_sql
            method = "direct"
        
        # Return result with metadata
        return {
            "sql": final_sql,
            "method": method,
            "time_taken": time.time() - start_time,
            "mcts_used": method == "mcts"
        }
    
    def _is_highly_complex_query(self, query: str) -> bool:
        """
        Selectively detect only truly complex queries that would benefit from MCTS
        
        Args:
            query: The natural language query
            
        Returns:
            True if highly complex, False otherwise
        """
        lower_query = query.lower()
        
        # Strong complexity indicators that signal need for MCTS
        complex_patterns = [
            # Multiple aggregations
            (r"\b(average|sum|count|minimum|maximum).*\b(average|sum|count|minimum|maximum)\b", 3),
            
            # Nested queries
            (r"\b(in|not in|exists|not exists)\b", 2),
            
            # Complex conditions
            (r"\b(group by|having|order by).*\b(group by|having|order by)\b", 2),
            
            # Superlatives
            (r"\b(most|least|highest|lowest|largest|smallest)\b", 1),
            
            # Multiple tables implied
            (r"\b(join|across|between|relationship|from.*and)\b", 1),
            
            # Complex aggregations
            (r"\b(average|mean|median|mode|variance|std|deviation)\b", 1),
            
            # Time/date operations
            (r"\b(year|month|day|date|time|period)\b", 1)
        ]
        
        # Count weighted matches
        complexity_score = 0
        for pattern, weight in complex_patterns:
            if re.search(pattern, lower_query):
                complexity_score += weight
        
        # Only apply MCTS to highly complex queries (score >= 3)
        is_complex = complexity_score >= 3
        
        # Also consider query length
        words = query.split()
        if len(words) > 15:  # Very long queries
            is_complex = True
        
        return is_complex
    
    def _is_better_result(self, new_results: List[Tuple], old_results: List[Tuple]) -> bool:
        """
        Determine if new results are better than old results
        
        Args:
            new_results: Results from new SQL
            old_results: Results from old SQL
            
        Returns:
            True if new results are better
        """
        # If old results are empty but new ones aren't
        if not old_results and new_results:
            return True
        
        # If both have results, prefer results with fewer rows (more specific)
        if old_results and new_results:
            # Prefer 1-10 rows as ideal
            old_size = len(old_results)
            new_size = len(new_results)
            
            if 1 <= new_size <= 10 and (old_size == 0 or old_size > 10):
                return True
            
            # Prefer non-empty results with fewer columns (more focused)
            if new_results and old_results:
                old_width = len(old_results[0]) if old_results and old_results[0] else 0
                new_width = len(new_results[0]) if new_results and new_results[0] else 0
                
                # If old results have too many columns, prefer new results with fewer
                if old_width > 5 and new_width < old_width and new_width > 0:
                    return True
        
        return False
    
    def _focused_mcts_search(self, nl_query: str, initial_sql: str) -> str:
        """
        Perform a focused MCTS search with tight time constraints
        
        Args:
            nl_query: Natural language query
            initial_sql: Initial SQL from direct generation
            
        Returns:
            Best SQL query found by MCTS
        """
        try:
            start_time = time.time()
            
            # Create root node with initial SQL
            root = SQLNode(state=initial_sql)
            
            # Generate high-quality refinements
            self._generate_focused_refinements(root, nl_query, initial_sql)
            
            # If no refinements, return initial SQL
            if not root.untried_actions:
                return initial_sql
            
            # Adjust iterations based on complexity but keep it reasonable
            adjusted_iterations = min(self.mcts_iterations, 10)
            
            # Track best SQL
            best_sql = initial_sql
            best_reward = 0
            
            # Run MCTS iterations with strict time limit
            for i in range(adjusted_iterations):
                # Check timeout
                if time.time() - start_time > MAX_SEARCH_TIME:
                    print(f"MCTS timeout after {i} iterations")
                    break
                
                try:
                    # Selection - find most promising node
                    node = self._select(root)
                    
                    # Expansion - expand the selected node
                    if node.untried_actions:
                        node = self._expand(node)
                    
                    # Simulation - evaluate the SQL
                    reward = self._evaluate_sql(node.state, nl_query)
                    
                    # Track best solution
                    if reward > best_reward:
                        best_reward = reward
                        best_sql = node.state
                        print(f"Found better SQL (reward: {reward:.2f})")
                    
                    # Backpropagation
                    self._backpropagate(node, reward)
                except Exception as e:
                    print(f"Error in MCTS iteration {i}: {str(e)}")
                    continue
            
            # If we found a better solution, use it
            if best_reward > 0.6:
                return best_sql
            
            # Otherwise, find best child of root
            best_child = None
            if root.children:
                best_child = max(root.children, key=lambda c: c.value if c.visits > 0 else 0)
            
            # Return best SQL
            if best_child and best_child.visits > 0 and best_child.value > 0.5:
                return best_child.state
            
            return best_sql
        
        except Exception as e:
            print(f"MCTS search error: {str(e)}")
            return initial_sql
    
    def _generate_focused_refinements(self, node: SQLNode, nl_query: str, sql: str):
        """
        Generate focused SQL refinements based on query patterns
        
        Args:
            node: The node to populate with actions
            nl_query: Natural language query
            sql: Current SQL
        """
        refinements = []
        
        # Extract key components from the SQL
        select_pattern = r"SELECT\s+(.*?)\s+FROM"
        from_pattern = r"FROM\s+(.*?)(?:WHERE|GROUP BY|ORDER BY|LIMIT|$)"
        where_pattern = r"WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)"
        group_pattern = r"GROUP BY\s+(.*?)(?:HAVING|ORDER BY|LIMIT|$)"
        order_pattern = r"ORDER BY\s+(.*?)(?:LIMIT|$)"
        
        # Extract components
        select_match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)
        from_match = re.search(from_pattern, sql, re.IGNORECASE | re.DOTALL)
        where_match = re.search(where_pattern, sql, re.IGNORECASE | re.DOTALL)
        group_match = re.search(group_pattern, sql, re.IGNORECASE | re.DOTALL)
        order_match = re.search(order_pattern, sql, re.IGNORECASE | re.DOTALL)
        
        # Apply targeted refinements based on the query type
        lower_query = nl_query.lower()
        
        # 1. Add GROUP BY if query seems to need it
        if (any(kw in lower_query for kw in ["group", "each", "per", "by"]) and 
            not group_match and from_match):
            tables = from_match.group(1).strip()
            if "JOIN" in tables.upper():
                # For join queries, find primary table
                main_table = tables.split()[0]
                # Add grouping on primary key or first column
                group_sql = sql + f" GROUP BY {main_table}.id"
                refinements.append(group_sql)
        
        # 2. Add ORDER BY for queries that imply sorting
        if (any(kw in lower_query for kw in ["top", "highest", "most", "largest", "smallest", "least", "lowest"]) and
            not order_match):
            if "highest" in lower_query or "most" in lower_query or "largest" in lower_query:
                refinements.append(sql + " ORDER BY 1 DESC")
            elif "smallest" in lower_query or "least" in lower_query or "lowest" in lower_query:
                refinements.append(sql + " ORDER BY 1 ASC")
        
        # 3. Add LIMIT for queries that imply a limited result
        if (any(kw in lower_query for kw in ["top", "most", "least"]) and 
            "LIMIT" not in sql.upper()):
            # Add LIMIT with or without existing ORDER BY
            if order_match:
                limit_sql = sql + " LIMIT 1"
            else:
                limit_sql = sql + " ORDER BY 1 DESC LIMIT 1"
            refinements.append(limit_sql)
        
        # 4. Add missing JOIN conditions
        if from_match and "JOIN" in from_match.group(1).upper() and "ON" not in from_match.group(1).upper():
            tables = re.findall(r'\b(\w+)\b', from_match.group(1))
            if len(tables) >= 2:
                # Create potential join condition
                t1, t2 = tables[0], tables[1]
                join_sql = sql.replace(f"{t1} JOIN {t2}", f"{t1} JOIN {t2} ON {t1}.id = {t2}.{t1}_id")
                refinements.append(join_sql)
        
        # 5. Handle COUNT queries specifically
        if ("how many" in lower_query or "count" in lower_query) and "COUNT" not in sql.upper():
            # Replace SELECT with COUNT
            if select_match:
                count_sql = sql.replace(select_match.group(0), "SELECT COUNT(*) FROM")
                refinements.append(count_sql)
        
        # Add specialized refinements
        if "HAVING" not in sql.upper() and "GROUP BY" in sql.upper():
            # Add HAVING clause for grouped queries with aggregation keywords
            if any(kw in lower_query for kw in ["more than", "less than", "at least", "at most"]):
                if "more than" in lower_query or "at least" in lower_query:
                    having_sql = sql + " HAVING COUNT(*) > 1"
                else:
                    having_sql = sql + " HAVING COUNT(*) < 5"
                refinements.append(having_sql)
        
        # If we couldn't generate good refinements, add simple variants
        if not refinements:
            if "LIMIT" not in sql.upper():
                refinements.append(sql + " LIMIT 10")
            if "ORDER BY" not in sql.upper():
                refinements.append(sql + " ORDER BY 1")
        
        # Add refinements to node's untried actions
        node.untried_actions = refinements[:MAX_CHILDREN]
    
    def _select(self, node: SQLNode) -> SQLNode:
        """
        Select most promising node using UCB
        
        Args:
            node: Current node
            
        Returns:
            Selected node
        """
        current = node
        
        # Traverse tree to find leaf or unexpanded node
        while current.untried_actions == [] and current.children != []:
            # Find child with highest UCB score
            current = max(current.children, key=lambda c: c.get_ucb_score(UCB_CONSTANT))
        
        return current
    
    def _expand(self, node: SQLNode) -> SQLNode:
        """
        Expand node by trying a random untried action
        
        Args:
            node: Node to expand
            
        Returns:
            New child node
        """
        # Choose a random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # Create child node
        child = node.add_child(state=action, action=action)
        
        return child
    
    def _evaluate_sql(self, sql: str, nl_query: str) -> float:
        """
        Evaluate SQL quality without simulation
        
        Args:
            sql: SQL query to evaluate
            nl_query: Natural language query
            
        Returns:
            Quality score between 0 and 1
        """
        # Check if SQL is valid
        is_valid, _ = self.validate_sql(sql)
        if not is_valid:
            return 0.1  # Small non-zero score for invalid SQL
        
        # Execute SQL
        exec_success, results = self.execute_sql(sql)
        if not exec_success:
            return 0.2  # Better score for valid but non-executable SQL
        
        # Base score for executable SQL
        base_score = 0.5
        
        # Assess result quality
        result_size = len(results)
        size_score = 0.0
        
        # Prefer 1-10 rows for most queries
        if 1 <= result_size <= 10:
            size_score = 0.2
        elif result_size > 10:
            size_score = 0.1
        elif result_size == 0:
            size_score = 0.0
        
        # Check SQL structure quality
        structure_score = 0.0
        
        # Essential clauses that should match the query
        lower_query = nl_query.lower()
        
        # Check for proper use of aggregation functions
        if "how many" in lower_query and "COUNT" in sql.upper():
            structure_score += 0.1
        
        if "average" in lower_query and "AVG" in sql.upper():
            structure_score += 0.1
        
        if any(w in lower_query for w in ["most", "highest", "maximum"]) and "MAX" in sql.upper():
            structure_score += 0.1
        
        if any(w in lower_query for w in ["least", "lowest", "minimum"]) and "MIN" in sql.upper():
            structure_score += 0.1
        
        # Check for appropriate grouping
        if any(w in lower_query for w in ["group", "each", "per"]) and "GROUP BY" in sql.upper():
            structure_score += 0.1
        
        # Check for appropriate ordering
        order_words = ["order", "sort", "rank", "top", "bottom"]
        if any(w in lower_query for w in order_words) and "ORDER BY" in sql.upper():
            structure_score += 0.1
            
            # Check direction
            if any(w in lower_query for w in ["highest", "most", "top"]) and "DESC" in sql.upper():
                structure_score += 0.05
            elif any(w in lower_query for w in ["lowest", "least", "bottom"]) and "ASC" in sql.upper():
                structure_score += 0.05
        
        # Final score
        total_score = base_score + size_score + structure_score
        return min(1.0, total_score)
    
    def _backpropagate(self, node: SQLNode, reward: float):
        """
        Backpropagate reward through the tree
        
        Args:
            node: Current node
            reward: Reward value
        """
        # Update all nodes up to the root
        current = node
        while current:
            current.update(reward)
            current = current.parent
    
    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """
        Validate SQL without executing
        
        Args:
            sql: SQL query to validate
            
        Returns:
            (is_valid, error_message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            conn.close()
            return True, ""
        except sqlite3.Error as e:
            return False, str(e)
    
    def execute_sql(self, sql: str) -> Tuple[bool, List[Tuple]]:
        """
        Execute SQL and return results
        
        Args:
            sql: SQL query to execute
            
        Returns:
            (success, results)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return True, results
        except sqlite3.Error:
            return False, []