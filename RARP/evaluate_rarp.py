#!/usr/bin/env python3
"""
<<<<<<< Updated upstream
Comprehensive evaluation script for MCTS-RARP on the Spider dataset
Supports both standard RARP and MCTS-enhanced RARP
=======
Comprehensive evaluation script for RARP (RAP + SQLCoder) on the Spider dataset
Evaluates both execution accuracy and query matching
Supports intelligent few-shot learning with multiple selection strategies
>>>>>>> Stashed changes
"""

import os
import json
import argparse
import sqlite3
import numpy as np
from pathlib import Path
import random
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
import multiprocessing
import re
from datetime import datetime

<<<<<<< Updated upstream
# Import our RARP and MCTS-RARP implementations
=======
# Advanced few-shot selection imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_SIMILARITY_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers and/or sklearn not available. Semantic similarity will be disabled.")
    SEMANTIC_SIMILARITY_AVAILABLE = False

# Import our RARP implementation
>>>>>>> Stashed changes
from working_rarp import RARP, DatabaseSchema
from mcts_rarp import MCTSRARP, evaluate_query_mcts

# Set paths
SPIDER_DIR = Path("../datasets/spider")
DATABASE_DIR = SPIDER_DIR / "database"
RESULTS_DIR = Path("./evaluation_results")

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)

<<<<<<< Updated upstream
def load_few_shot_examples(num_examples=3, seed=42):
    """Load a few examples from the training set for few-shot learning"""
=======
# Global variable for semantic similarity model
semantic_model = None

def initialize_semantic_model():
    """Initialize the semantic similarity model (lazy loading)"""
    global semantic_model
    if SEMANTIC_SIMILARITY_AVAILABLE and semantic_model is None:
        print("Loading semantic similarity model...")
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return semantic_model

def categorize_sql_pattern(sql: str) -> List[str]:
    """Categorize SQL by patterns it uses"""
    sql_lower = sql.lower()
    patterns = []
    
    if 'join' in sql_lower or 'inner join' in sql_lower or 'left join' in sql_lower or 'right join' in sql_lower:
        patterns.append('join')
    if 'group by' in sql_lower:
        patterns.append('aggregation')
    if 'order by' in sql_lower:
        patterns.append('sorting')
    if 'having' in sql_lower:
        patterns.append('having')
    if sql_lower.count('select') > 1:
        patterns.append('subquery')
    if 'union' in sql_lower:
        patterns.append('union')
    if any(func in sql_lower for func in ['count(', 'sum(', 'avg(', 'max(', 'min(']):
        patterns.append('aggregate_function')
    if 'where' in sql_lower:
        patterns.append('filtering')
    if 'like' in sql_lower:
        patterns.append('string_matching')
    if any(op in sql_lower for op in ['>', '<', '>=', '<=', '!=']):
        patterns.append('comparison')
    
    # Complexity indicators
    if len(sql.split()) > 30:
        patterns.append('complex')
    elif len(sql.split()) > 15:
        patterns.append('medium')
    else:
        patterns.append('simple')
    
    return patterns

def calculate_query_similarity(query1: str, query2: str) -> float:
    """Calculate semantic similarity between two queries"""
    if not SEMANTIC_SIMILARITY_AVAILABLE or semantic_model is None:
        # Fallback to simple keyword overlap
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        return len(words1.intersection(words2)) / len(words1.union(words2))
    
    try:
        embeddings = semantic_model.encode([query1, query2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        # Fallback to keyword overlap
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        return len(words1.intersection(words2)) / len(words1.union(words2))

def load_semantic_few_shot_examples(test_query: str, train_dataset: List[Dict], num_examples: int = 3) -> List[Dict]:
    """Select few-shot examples based on semantic similarity to test query"""
    if not train_dataset:
        return []
    
    similarities = []
    for example in train_dataset:
        similarity = calculate_query_similarity(test_query, example['question'])
        similarities.append((similarity, example))
    
    # Sort by similarity and return top examples
    similarities.sort(reverse=True, key=lambda x: x[0])
    selected = [ex for _, ex in similarities[:num_examples]]
    
    print(f"Selected {len(selected)} examples based on semantic similarity")
    return selected

def load_database_specific_examples(db_id: str, train_dataset: List[Dict], num_examples: int = 3, 
                                   test_query: str = None) -> List[Dict]:
    """Select examples from the same database, optionally using semantic similarity"""
    same_db_examples = [ex for ex in train_dataset if ex['db_id'] == db_id]
    
    if len(same_db_examples) >= num_examples:
        if test_query and SEMANTIC_SIMILARITY_AVAILABLE:
            # Use semantic similarity within same database
            return load_semantic_few_shot_examples(test_query, same_db_examples, num_examples)
        else:
            # Random selection within same database
            return random.sample(same_db_examples, num_examples)
    else:
        print(f"Warning: Only {len(same_db_examples)} examples found for database {db_id}, falling back to semantic similarity")
        # Fall back to semantic similarity across all training data
        if test_query:
            return load_semantic_few_shot_examples(test_query, train_dataset, num_examples)
        else:
            return random.sample(train_dataset, min(num_examples, len(train_dataset)))

def load_pattern_matched_examples(target_patterns: List[str], train_dataset: List[Dict], 
                                 num_examples: int = 3, test_query: str = None) -> List[Dict]:
    """Select examples with similar SQL patterns"""
    candidates = []
    
    for example in train_dataset:
        example_patterns = categorize_sql_pattern(example['query'])
        overlap = len(set(target_patterns) & set(example_patterns))
        
        # Add semantic similarity as a tie-breaker if available
        semantic_score = 0
        if test_query:
            semantic_score = calculate_query_similarity(test_query, example['question'])
        
        if overlap > 0:
            candidates.append((overlap, semantic_score, example))
    
    if not candidates:
        print("Warning: No pattern matches found, falling back to semantic similarity")
        if test_query:
            return load_semantic_few_shot_examples(test_query, train_dataset, num_examples)
        else:
            return random.sample(train_dataset, min(num_examples, len(train_dataset)))
    
    # Sort by pattern overlap first, then by semantic similarity
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
    selected = [ex for _, _, ex in candidates[:num_examples]]
    
    print(f"Selected {len(selected)} examples based on SQL pattern matching")
    return selected

def load_diverse_examples(train_dataset: List[Dict], num_examples: int = 3) -> List[Dict]:
    """Select diverse examples across different query types"""
    categories = {
        'simple_select': [],
        'join_queries': [],
        'aggregation': [],
        'subqueries': [],
        'complex': [],
        'string_matching': [],
        'sorting': []
    }
    
    # Categorize all training examples
    for example in train_dataset:
        patterns = categorize_sql_pattern(example['query'])
        
        if 'join' in patterns:
            categories['join_queries'].append(example)
        elif 'aggregation' in patterns or 'aggregate_function' in patterns:
            categories['aggregation'].append(example)
        elif 'subquery' in patterns:
            categories['subqueries'].append(example)
        elif 'complex' in patterns:
            categories['complex'].append(example)
        elif 'string_matching' in patterns:
            categories['string_matching'].append(example)
        elif 'sorting' in patterns:
            categories['sorting'].append(example)
        else:
            categories['simple_select'].append(example)
    
    # Sample one from each category, prioritizing non-empty categories
    diverse_examples = []
    categories_with_examples = [(k, v) for k, v in categories.items() if v]
    
    # Shuffle to ensure random selection when we have more categories than examples needed
    random.shuffle(categories_with_examples)
    
    for category, examples in categories_with_examples:
        if len(diverse_examples) < num_examples:
            diverse_examples.append(random.choice(examples))
    
    # If we need more examples, fill with random samples
    while len(diverse_examples) < num_examples and len(diverse_examples) < len(train_dataset):
        remaining_examples = [ex for ex in train_dataset if ex not in diverse_examples]
        if remaining_examples:
            diverse_examples.append(random.choice(remaining_examples))
        else:
            break
    
    print(f"Selected {len(diverse_examples)} diverse examples across different query types")
    return diverse_examples

def load_hybrid_examples(test_query: str, db_id: str, train_dataset: List[Dict], 
                        num_examples: int = 3, gold_sql: str = None) -> List[Dict]:
    """Hybrid approach combining multiple selection strategies"""
    if not train_dataset:
        return []
    
    # Strategy 1: Try database-specific first (gets 60% of examples)
    db_specific_count = max(1, int(num_examples * 0.6))
    db_examples = load_database_specific_examples(db_id, train_dataset, db_specific_count, test_query)
    
    remaining_count = num_examples - len(db_examples)
    
    if remaining_count > 0:
        # Strategy 2: Pattern matching if we have gold SQL (gets remaining examples)
        if gold_sql:
            target_patterns = categorize_sql_pattern(gold_sql)
            # Exclude already selected examples
            remaining_dataset = [ex for ex in train_dataset if ex not in db_examples]
            pattern_examples = load_pattern_matched_examples(target_patterns, remaining_dataset, 
                                                           remaining_count, test_query)
            db_examples.extend(pattern_examples)
            remaining_count = num_examples - len(db_examples)
        
        # Strategy 3: Semantic similarity for any remaining slots
        if remaining_count > 0:
            remaining_dataset = [ex for ex in train_dataset if ex not in db_examples]
            semantic_examples = load_semantic_few_shot_examples(test_query, remaining_dataset, remaining_count)
            db_examples.extend(semantic_examples)
    
    print(f"Hybrid selection: {len(db_examples)} examples selected using combined strategies")
    return db_examples[:num_examples]  # Ensure we don't exceed the limit

def load_few_shot_examples(test_query: str = None, db_id: str = None, num_examples: int = 3, 
                          seed: int = 42, strategy: str = "hybrid", gold_sql: str = None) -> List[Dict]:
    """Load few-shot examples using specified strategy"""
>>>>>>> Stashed changes
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load training dataset
    try:
        with open(SPIDER_DIR / "train_spider.json", 'r') as f:
            train_dataset = json.load(f)
    except FileNotFoundError:
        print("Warning: Training dataset file not found. No few-shot examples will be used.")
        return []
    
<<<<<<< Updated upstream
    # Sample a few examples
    if num_examples <= len(train_dataset):
        few_shot_examples = random.sample(train_dataset, num_examples)
    else:
        few_shot_examples = train_dataset
    
    print(f"Selected {len(few_shot_examples)} few-shot examples from training set:")
    for i, example in enumerate(few_shot_examples):
        print(f"Example {i+1}: {example['question']} -> {example['query'][:80]}...")
    
    return few_shot_examples

=======
    if not train_dataset:
        return []
    
    # Initialize semantic model if needed
    if strategy in ["semantic", "hybrid"] and SEMANTIC_SIMILARITY_AVAILABLE:
        initialize_semantic_model()
    
    # Select examples based on strategy
    if strategy == "random":
        few_shot_examples = random.sample(train_dataset, min(num_examples, len(train_dataset)))
        print(f"Selected {len(few_shot_examples)} random examples")
    
    elif strategy == "semantic" and test_query:
        few_shot_examples = load_semantic_few_shot_examples(test_query, train_dataset, num_examples)
    
    elif strategy == "database" and db_id:
        few_shot_examples = load_database_specific_examples(db_id, train_dataset, num_examples, test_query)
    
    elif strategy == "pattern" and gold_sql:
        target_patterns = categorize_sql_pattern(gold_sql)
        few_shot_examples = load_pattern_matched_examples(target_patterns, train_dataset, num_examples, test_query)
    
    elif strategy == "diverse":
        few_shot_examples = load_diverse_examples(train_dataset, num_examples)
    
    elif strategy == "hybrid" and test_query and db_id:
        few_shot_examples = load_hybrid_examples(test_query, db_id, train_dataset, num_examples, gold_sql)
    
    else:
        print(f"Warning: Strategy '{strategy}' not available or missing required parameters. Falling back to random selection.")
        few_shot_examples = random.sample(train_dataset, min(num_examples, len(train_dataset)))
    
    print(f"Few-shot examples selected using '{strategy}' strategy:")
    for i, example in enumerate(few_shot_examples):
        print(f"Example {i+1}: {example['question'][:60]}... -> {example['query'][:60]}...")
    
    return few_shot_examples

# Enhanced RARP class with few-shot learning
class FewShotRARP(RARP):
    """Extends RARP to support few-shot learning"""
    
    def __init__(self, db_id, model, tables_path=None, few_shot_examples=None):
        """Initialize FewShotRARP with few-shot examples"""
        super().__init__(db_id, model, tables_path)
        self.few_shot_examples = few_shot_examples or []
    
    def generate_sql(self, query, include_samples=True):
        """Generate SQL with few-shot examples"""
        # Format few-shot examples for the prompt
        few_shot_prompt = ""
        if self.few_shot_examples:
            few_shot_prompt = "Here are some examples of how to convert natural language questions to SQL:\n\n"
            for i, example in enumerate(self.few_shot_examples):
                few_shot_prompt += f"Example {i+1}:\n"
                few_shot_prompt += f"Question: {example['question']}\n"
                few_shot_prompt += f"SQL: {example['query']}\n\n"
            
            few_shot_prompt += "Now, please generate SQL for the following question:\n\n"
            few_shot_prompt += f"Question: {query}\n"
            few_shot_prompt += "SQL:"
            
            # Replace the query with the enhanced query
            query = few_shot_prompt
        
        # Call the parent's generate_sql method with the enhanced query
        return super().generate_sql(query, include_samples)

>>>>>>> Stashed changes
def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison"""
    sql = sql.lower()
    sql = re.sub(r'--.*', '', sql)  # Remove comments

    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    sql = re.sub(r'\s*([=<>!]+)\s*', r'\1', sql) # Normalize spaces around operators
    sql = re.sub(r'\s*([(),])\s*', r'\1', sql) # Normalize spaces around parentheses and commas

    sql = sql.rstrip(';')

    # More robust alias normalization (still simplified, consider a proper SQL parser for complex cases)
    # This example focuses on 'AS alias' but you might need to handle implicit aliasing
    sql = re.sub(r'(\s+as\s+)[a-z_][a-z0-9_]*', r'\1alias', sql) # Normalize table/column aliases
    
    # Optional: remove quotes from identifiers if your DB is case-insensitive or handles them consistently
    # sql = re.sub(r'"([a-z_][a-z0-9_]*)"', r'\1', sql)
    # sql = re.sub(r'`([a-z_][a-z0-9_]*)`', r'\1', sql)
    
    return sql

def get_execution_result(sql: str, db_path: str) -> Tuple[bool, List[Tuple]]:
    """Execute SQL and return results"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except sqlite3.Error:
        return False, []

def results_match(results1: List[Tuple], results2: List[Tuple]) -> bool:
    """Check if two sets of results match"""
    if len(results1) != len(results2):
        return False
    
    # Convert results to sets of tuples for unordered comparison
    set1 = set(tuple(row) for row in results1)
    set2 = set(tuple(row) for row in results2)
    
    return set1 == set2

<<<<<<< Updated upstream
def evaluate_query(example: Dict[str, Any], model: str, include_samples: bool, few_shot_examples: List[Dict[str, Any]] = None, use_mcts: bool = True, mcts_iterations: int = 50, debug: bool = False) -> Dict[str, Any]:
    """Evaluate a single query using MCTS-RARP if requested"""
=======
def evaluate_query(example: Dict[str, Any], model: str, include_samples: bool, 
                  few_shot_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Evaluate a single query"""
>>>>>>> Stashed changes
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
        # Check if database directory exists
        db_dir = DATABASE_DIR / db_id
        test_db_dir = SPIDER_DIR / "test_database" / db_id
        
        # Find the correct database directory (could be in database or test_database)
        if db_dir.exists():
            db_path = db_dir / f"{db_id}.sqlite"
        elif test_db_dir.exists():
            db_path = test_db_dir / f"{db_id}.sqlite"
        else:
            return {
                "db_id": db_id,
                "question": query,
                "gold_sql": gold_sql if has_gold_sql else "N/A",
                "generated_sql": "ERROR: Database not found",
                "exact_match": False,
                "exec_match": False,
                "execution_success": False,
                "time_taken": time.time() - start_time,
                "error": f"Database {db_id} not found in database or test_database directories",
                "method": "error"
            }
        
<<<<<<< Updated upstream
        # Initialize appropriate RARP version
        tables_path = None
        if test_db_dir.exists():
            tables_path = str(SPIDER_DIR / "test_tables.json")
            
        try:
            if use_mcts:
                # Use MCTS-RARP
                rarp = MCTSRARP(db_id, model, tables_path=tables_path, few_shot_examples=few_shot_examples, mcts_iterations=mcts_iterations)
=======
        # Load few-shot examples if configured
        few_shot_examples = None
        if few_shot_config and few_shot_config.get("num_examples", 0) > 0:
            few_shot_examples = load_few_shot_examples(
                test_query=query,
                db_id=db_id,
                num_examples=few_shot_config["num_examples"],
                seed=few_shot_config.get("seed", 42),
                strategy=few_shot_config.get("strategy", "hybrid"),
                gold_sql=gold_sql if has_gold_sql else None
            )
        
        # Initialize RARP or FewShotRARP with the database
        if few_shot_examples and len(few_shot_examples) > 0:
            # Use FewShotRARP
            if test_db_dir.exists():
                tables_path = str(SPIDER_DIR / "test_tables.json")
                rarp = FewShotRARP(db_id, model, tables_path, few_shot_examples)
>>>>>>> Stashed changes
            else:
                # Use regular RARP or FewShotRARP
                if few_shot_examples and len(few_shot_examples) > 0:
                    from few_shot_rarp import FewShotRARP
                    rarp = FewShotRARP(db_id, model, tables_path, few_shot_examples)
                else:
                    rarp = RARP(db_id, model, tables_path)
                
            # Debug output if requested
            if debug:
                # For MCTS version, check if query is complex
                if use_mcts:
                    is_complex = rarp._is_complex_query(query)
                    print(f"\nQuery: {query}")
                    print(f"Is complex: {is_complex}")
            
            # Generate SQL
            result = rarp.generate_sql(query, include_samples)
            generated_sql = result["sql"]
            method = result.get("method", "direct")
            
            # More debug output
            if debug:
                if use_mcts:
                    print(f"Direct SQL: {result.get('direct_sql', 'N/A')}")
                    print(f"MCTS SQL: {result.get('mcts_sql', 'N/A')}")
                    print(f"Method used: {method}")
            
            # Default values for matching metrics
            exact_match = False
            exec_match = False
            
            # Check matches only if gold SQL is available
            if has_gold_sql:
                # Normalize SQL for comparison
                normalized_generated = normalize_sql(generated_sql)
                normalized_gold = normalize_sql(gold_sql)
                
                # Check for exact match
                exact_match = normalized_generated == normalized_gold
                
                # Try to execute the generated SQL
                gen_exec_success, gen_results = get_execution_result(generated_sql, str(db_path))
                
                # Try to execute the gold SQL
                gold_exec_success, gold_results = get_execution_result(gold_sql, str(db_path))
                
                # Check if both executed successfully and have matching results
                if gen_exec_success and gold_exec_success:
                    exec_match = results_match(gen_results, gold_results)
                    
                    if debug:
                        print(f"Execution match: {exec_match}")
                        if not exec_match:
                            print(f"Generated results: {gen_results[:3]} (showing up to 3)")
                            print(f"Gold results: {gold_results[:3]} (showing up to 3)")
            else:
                # If no gold SQL, just check if the generated SQL executes
                gen_exec_success, _ = get_execution_result(generated_sql, str(db_path))
            
            time_taken = time.time() - start_time
            
            return {
                "db_id": db_id,
                "question": query,
                "gold_sql": gold_sql if has_gold_sql else "N/A",
                "generated_sql": generated_sql,
                "method": method,
                "exact_match": exact_match,
                "exec_match": exec_match,
                "execution_success": gen_exec_success,
                "time_taken": time_taken,
                "error": None,
                "mcts_used": use_mcts and method == "mcts"
            }
        except Exception as e:
            if debug:
                print(f"Error in RARP initialization or SQL generation: {str(e)}")
            raise e
            
<<<<<<< Updated upstream
=======
            # Try to execute the gold SQL
            gold_exec_success, gold_results = get_execution_result(gold_sql, str(db_path))
            
            # Check if both executed successfully and have matching results
            if gen_exec_success and gold_exec_success:
                exec_match = results_match(gen_results, gold_results)
        else:
            # If no gold SQL, just check if the generated SQL executes
            gen_exec_success, _ = get_execution_result(generated_sql, str(db_path))
        
        time_taken = time.time() - start_time
        
        return {
            "db_id": db_id,
            "question": query,
            "gold_sql": gold_sql if has_gold_sql else "N/A",
            "generated_sql": generated_sql,
            "exact_match": exact_match,
            "exec_match": exec_match,
            "execution_success": gen_exec_success,
            "time_taken": time_taken,
            "error": None,
            "few_shot_examples": [{"question": ex["question"], "query": ex["query"]} 
                                 for ex in few_shot_examples] if few_shot_examples else None
        }
        
>>>>>>> Stashed changes
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        if debug:
            print(f"Error processing query '{query}': {str(e)}")
            print(f"Traceback: {traceback_str}")
        
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
<<<<<<< Updated upstream
            "method": "error"
=======
            "few_shot_examples": None
>>>>>>> Stashed changes
        }

def evaluate_dataset(dataset_file: str, model: str, 
                    num_samples: int = None, 
                    include_samples: bool = True,
                    seed: int = 42,
                    num_processes: int = 1,
<<<<<<< Updated upstream
                    num_few_shot: int = 0,
                    use_mcts: bool = True,
                    mcts_iterations: int = 50,
                    debug: bool = False) -> Dict[str, Any]:
    """Evaluate MCTS-RARP on a dataset"""
=======
                    few_shot_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Evaluate RARP on a dataset"""
>>>>>>> Stashed changes
    # Load dataset
    try:
        with open(SPIDER_DIR / dataset_file, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Dataset file {dataset_file} not found. Skipping.")
        return {
            "model": model,
            "dataset": dataset_file,
            "total_examples": 0,
            "exact_match_accuracy": 0,
            "execution_match_accuracy": 0,
            "execution_success_rate": 0,
            "error_rate": 0,
            "average_time_per_query": 0,
            "total_time": 0,
            "per_database_accuracy": {},
            "results": [],
<<<<<<< Updated upstream
            "few_shot_examples": None,
            "use_mcts": use_mcts,
            "mcts_iterations": mcts_iterations
=======
            "few_shot_config": few_shot_config
>>>>>>> Stashed changes
        }
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample examples if num_samples is specified
    if num_samples is not None and num_samples < len(dataset):
        examples = random.sample(dataset, num_samples)
    else:
        examples = dataset
    
    print(f"Evaluating {len(examples)} examples from {dataset_file} with model {model}")
<<<<<<< Updated upstream
    print(f"MCTS: {'Enabled' if use_mcts else 'Disabled'}, Iterations: {mcts_iterations if use_mcts else 'N/A'}")
=======
    if few_shot_config and few_shot_config.get("num_examples", 0) > 0:
        print(f"Using few-shot learning with {few_shot_config['num_examples']} examples, strategy: {few_shot_config.get('strategy', 'hybrid')}")
>>>>>>> Stashed changes
    
    # Special handling for test.json which might not have ground truth
    has_gold_sql = all("query" in example for example in examples[:5])
    
    # Evaluate in parallel if multiple processes are specified
    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.starmap(
                    evaluate_query, 
<<<<<<< Updated upstream
                    [(example, model, include_samples, few_shot_examples, use_mcts, mcts_iterations, debug) for example in examples]
=======
                    [(example, model, include_samples, few_shot_config) for example in examples]
>>>>>>> Stashed changes
                ),
                total=len(examples)
            ))
    else:
        # Evaluate sequentially
        results = []
        for example in tqdm(examples, desc="Evaluating"):
<<<<<<< Updated upstream
            result = evaluate_query(example, model, include_samples, few_shot_examples, use_mcts, mcts_iterations, debug)
=======
            result = evaluate_query(example, model, include_samples, few_shot_config)
>>>>>>> Stashed changes
            results.append(result)
    
    # Calculate metrics
    total = len(results)
    exact_matches = sum(1 for r in results if r["exact_match"]) if has_gold_sql else 0
    exec_matches = sum(1 for r in results if r["exec_match"]) if has_gold_sql else 0
    exec_success = sum(1 for r in results if r["execution_success"])
    errors = sum(1 for r in results if r["error"] is not None)
    
    # Calculate MCTS usage statistics
    mcts_used = sum(1 for r in results if r.get("mcts_used", False))
    mcts_usage_rate = mcts_used / total if total > 0 else 0
    
    # Calculate metrics by method
    direct_results = [r for r in results if r.get("method") == "direct"]
    mcts_results = [r for r in results if r.get("method") == "mcts"]
    
    direct_exec_matches = sum(1 for r in direct_results if r["exec_match"]) if has_gold_sql else 0
    mcts_exec_matches = sum(1 for r in mcts_results if r["exec_match"]) if has_gold_sql else 0
    
    direct_exec_success = sum(1 for r in direct_results if r["execution_success"])
    mcts_exec_success = sum(1 for r in mcts_results if r["execution_success"])
    
    direct_accuracy = direct_exec_matches / len(direct_results) if direct_results and has_gold_sql else 0
    mcts_accuracy = mcts_exec_matches / len(mcts_results) if mcts_results and has_gold_sql else 0
    
    direct_success_rate = direct_exec_success / len(direct_results) if direct_results else 0
    mcts_success_rate = mcts_exec_success / len(mcts_results) if mcts_results else 0
    
    # Calculate average time
    total_time = sum(r["time_taken"] for r in results)
    avg_time = total_time / total if total > 0 else 0
    
    # Group results by database
    db_metrics = defaultdict(lambda: {"total": 0, "exact": 0, "exec": 0, "success": 0, "mcts_used": 0})
    for r in results:
        db_id = r["db_id"]
        db_metrics[db_id]["total"] += 1
        if has_gold_sql:
            db_metrics[db_id]["exact"] += 1 if r["exact_match"] else 0
            db_metrics[db_id]["exec"] += 1 if r["exec_match"] else 0
        db_metrics[db_id]["success"] += 1 if r["execution_success"] else 0
        db_metrics[db_id]["mcts_used"] += 1 if r.get("mcts_used", False) else 0
    
    # Calculate per-database accuracy
    db_accuracy = {}
    for db_id, metrics in db_metrics.items():
        db_accuracy[db_id] = {
            "exact_match_accuracy": metrics["exact"] / metrics["total"] if metrics["total"] > 0 and has_gold_sql else 0,
            "execution_match_accuracy": metrics["exec"] / metrics["total"] if metrics["total"] > 0 and has_gold_sql else 0,
            "execution_success_rate": metrics["success"] / metrics["total"] if metrics["total"] > 0 else 0,
            "mcts_usage_rate": metrics["mcts_used"] / metrics["total"] if metrics["total"] > 0 else 0,
            "total_examples": metrics["total"]
        }
    
    return {
        "model": model,
        "dataset": dataset_file,
        "total_examples": total,
        "exact_match_accuracy": exact_matches / total if total > 0 and has_gold_sql else 0,
        "execution_match_accuracy": exec_matches / total if total > 0 and has_gold_sql else 0,
        "execution_success_rate": exec_success / total if total > 0 else 0,
        "error_rate": errors / total if total > 0 else 0,
        "average_time_per_query": avg_time,
        "total_time": total_time,
        "per_database_accuracy": db_accuracy,
        "results": results,
        "has_gold_sql": has_gold_sql,
<<<<<<< Updated upstream
        "few_shot_examples": [{"question": ex["question"], "query": ex["query"]} 
                             for ex in few_shot_examples] if few_shot_examples else None,
        "use_mcts": use_mcts,
        "mcts_iterations": mcts_iterations,
        "mcts_usage_rate": mcts_usage_rate,
        "direct_vs_mcts": {
            "direct_count": len(direct_results),
            "mcts_count": len(mcts_results),
            "direct_accuracy": direct_accuracy,
            "mcts_accuracy": mcts_accuracy,
            "direct_success_rate": direct_success_rate,
            "mcts_success_rate": mcts_success_rate
        }
=======
        "few_shot_config": few_shot_config
>>>>>>> Stashed changes
    }

def save_results(results: Dict[str, Any], output_file: str = None):
    """Save evaluation results to a file"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = results["model"].replace("/", "-")
        dataset_name = results["dataset"].replace(".json", "")
<<<<<<< Updated upstream
        
        mcts_suffix = f"_mcts{results['mcts_iterations']}" if results["use_mcts"] else ""
        few_shot_suffix = f"_fewshot{len(results['few_shot_examples'])}" if results.get("few_shot_examples") else ""
        
        output_file = f"{RESULTS_DIR}/eval_{dataset_name}_{model_name}{mcts_suffix}{few_shot_suffix}_{timestamp}.json"
=======
        few_shot_suffix = ""
        if results.get("few_shot_config") and results["few_shot_config"].get("num_examples", 0) > 0:
            strategy = results["few_shot_config"].get("strategy", "hybrid")
            few_shot_suffix = f"_fewshot{results['few_shot_config']['num_examples']}_{strategy}"
        output_file = f"{RESULTS_DIR}/eval_{dataset_name}_{model_name}{few_shot_suffix}_{timestamp}.json"
>>>>>>> Stashed changes
    
    with open(output_file, 'w') as f:
        # Save a summary without the full results to keep the file smaller
        summary = {k: v for k, v in results.items() if k != "results"}
        json.dump(summary, f, indent=2)
    
    # Save detailed results separately
    detailed_output = output_file.replace(".json", "_detailed.json")
    with open(detailed_output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Summary results saved to {output_file}")
    print(f"Detailed results saved to {detailed_output}")

def print_results_summary(results: Dict[str, Any]):
    """Print a summary of the evaluation results"""
    print("\n" + "="*50)
    print(f"Evaluation Summary for {results['model']} on {results['dataset']}")
    print("="*50)
    print(f"Total examples evaluated: {results['total_examples']}")
    
    if results.get("has_gold_sql", True):
        print(f"Exact match accuracy: {results['exact_match_accuracy']:.2%}")
        print(f"Execution match accuracy: {results['execution_match_accuracy']:.2%}")
    else:
        print("No gold SQL available for evaluation of exact/execution match accuracy")
    
    print(f"Execution success rate: {results['execution_success_rate']:.2%}")
    print(f"Error rate: {results['error_rate']:.2%}")
    print(f"Average time per query: {results['average_time_per_query']:.2f} seconds")
    print(f"Total evaluation time: {results['total_time']:.2f} seconds")
    
    # Print MCTS statistics if used
    if results.get("use_mcts", False):
        print("\nMCTS Statistics:")
        print(f"MCTS Iterations: {results['mcts_iterations']}")
        print(f"MCTS Usage Rate: {results['mcts_usage_rate']:.2%}")
        
        direct_vs_mcts = results.get("direct_vs_mcts", {})
        if direct_vs_mcts:
            print("\nDirect vs MCTS Comparison:")
            print(f"Direct Generation Count: {direct_vs_mcts['direct_count']}")
            print(f"MCTS Generation Count: {direct_vs_mcts['mcts_count']}")
            
            if results.get("has_gold_sql", True):
                print(f"Direct Generation Accuracy: {direct_vs_mcts['direct_accuracy']:.2%}")
                print(f"MCTS Generation Accuracy: {direct_vs_mcts['mcts_accuracy']:.2%}")
            
            print(f"Direct Generation Success Rate: {direct_vs_mcts['direct_success_rate']:.2%}")
            print(f"MCTS Generation Success Rate: {direct_vs_mcts['mcts_success_rate']:.2%}")
    
    # Print few-shot information if available
<<<<<<< Updated upstream
    if results.get("few_shot_examples"):
        print(f"\nUsed {len(results['few_shot_examples'])} few-shot examples")
=======
    if results.get("few_shot_config") and results["few_shot_config"].get("num_examples", 0) > 0:
        config = results["few_shot_config"]
        print(f"Few-shot learning: {config['num_examples']} examples using '{config.get('strategy', 'hybrid')}' strategy")
>>>>>>> Stashed changes
    
    # Print top and bottom performing databases
    if results.get("has_gold_sql", True):
        db_accuracies = [(db, metrics["execution_match_accuracy"]) 
                        for db, metrics in results["per_database_accuracy"].items()
                        if metrics["total_examples"] >= 5]  # Only consider DBs with at least 5 examples
        
        if db_accuracies:
            db_accuracies.sort(key=lambda x: x[1], reverse=True)
            
            print("\nTop 5 performing databases:")
            for db, acc in db_accuracies[:5]:
                print(f"  {db}: {acc:.2%}")
            
            print("\nBottom 5 performing databases:")
            for db, acc in db_accuracies[-5:]:
                print(f"  {db}: {acc:.2%}")
    
    # Print some example failures
    if "results" in results and results.get("has_gold_sql", True):
        failures = [r for r in results["results"] if not r["exec_match"] and not r["error"]]
        if failures:
            print("\nExample failures:")
            for i, failure in enumerate(random.sample(failures, min(3, len(failures)))):
                print(f"\nFailure {i+1}:")
                print(f"Question: {failure['question']}")
                print(f"Gold SQL: {failure['gold_sql']}")
                print(f"Generated SQL: {failure['generated_sql']}")
                print(f"Database: {failure['db_id']}")
                print(f"Method: {failure.get('method', 'N/A')}")
    elif "results" in results:
        # Just show some random examples for datasets without gold SQL
        sample_results = random.sample(results["results"], min(3, len(results["results"])))
        print("\nSample results:")
        for i, result in enumerate(sample_results):
            print(f"\nResult {i+1}:")
            print(f"Question: {result['question']}")
            print(f"Generated SQL: {result['generated_sql']}")
            print(f"Execution successful: {result['execution_success']}")
            print(f"Database: {result['db_id']}")
            print(f"Method: {result.get('method', 'N/A')}")

def main():
    """Main function to run evaluation"""
<<<<<<< Updated upstream
    parser = argparse.ArgumentParser(description="Evaluate MCTS-RARP on Spider dataset")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Groq model to use")
=======
    parser = argparse.ArgumentParser(description="Evaluate RARP on Spider dataset with intelligent few-shot learning")
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="Groq model to use")
>>>>>>> Stashed changes
    parser.add_argument("--datasets", nargs='+', default=["dev.json"], 
                      choices=["dev.json", "train_spider.json", "test.json"], 
                      help="Datasets to evaluate on (can specify multiple)")
    parser.add_argument("--samples", type=int, default=None, help="Number of examples to evaluate per dataset (None for all)")
    parser.add_argument("--no-sample-data", action="store_true", help="Don't include sample data in context")
    parser.add_argument("--output", default=None, help="Output file prefix for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--processes", type=int, default=1, help="Number of processes to use")
    parser.add_argument("--verbose-errors", action="store_true", help="Print detailed error information")
    
    # Few-shot learning arguments
    parser.add_argument("--few-shot", type=int, default=0, 
                       help="Number of few-shot examples to include (0 for none, 2-5 recommended)")
    parser.add_argument("--few-shot-strategy", default="hybrid", 
                       choices=["random", "semantic", "database", "pattern", "diverse", "hybrid"],
                       help="Strategy for selecting few-shot examples")
    parser.add_argument("--few-shot-seed", type=int, default=42, 
                       help="Random seed for few-shot example selection")
<<<<<<< Updated upstream
    parser.add_argument("--no-mcts", action="store_true", help="Disable MCTS (use regular RARP)")
    parser.add_argument("--mcts-iterations", type=int, default=30, help="Number of MCTS iterations (default increased to 30)")
    parser.add_argument("--compare", action="store_true", 
                      help="Run both MCTS and non-MCTS versions for comparison")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--fast", action="store_true", help="Use fast mode with fewer iterations and timeouts")
    parser.add_argument("--balanced", action="store_true", help="Use balanced settings with moderate timeouts")
    args = parser.parse_args()
    
    # Apply fast mode settings if requested
    if args.fast:
        args.mcts_iterations = 10
        print("Fast mode enabled - using 10 MCTS iterations with tight timeouts")
    elif args.balanced:
        args.mcts_iterations = 20
        print("Balanced mode enabled - using 20 MCTS iterations with moderate timeouts")
=======
    
    args = parser.parse_args()
    
    # Configure few-shot learning
    few_shot_config = None
    if args.few_shot > 0:
        few_shot_config = {
            "num_examples": args.few_shot,
            "strategy": args.few_shot_strategy,
            "seed": args.few_shot_seed
        }
>>>>>>> Stashed changes
    
    combined_results = {
        "model": args.model,
        "datasets": args.datasets,
<<<<<<< Updated upstream
        "few_shot_count": args.few_shot,
        "use_mcts": not args.no_mcts,
        "mcts_iterations": args.mcts_iterations,
        "per_dataset_results": {},
        "overall_metrics": {
            "total_examples": 0,
            "exact_match_count": 0,
            "exec_match_count": 0,
            "exec_success_count": 0,
            "error_count": 0,
            "total_time": 0
        }
    }
    
    # Collect dataset results
    dataset_results = {}
    
    # If comparison mode, evaluate both with and without MCTS
    if args.compare:
        # First run with MCTS
        combined_results_mcts = run_evaluation(
            args, use_mcts=True, mcts_iterations=args.mcts_iterations,
            output_prefix="mcts_"
        )
        
        # Then run without MCTS
        combined_results_no_mcts = run_evaluation(
            args, use_mcts=False, mcts_iterations=0,
            output_prefix="no_mcts_"
        )
        
        # Save comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.replace("/", "-")
        few_shot_suffix = f"_fewshot{args.few_shot}" if args.few_shot > 0 else ""
        
        comparison_output = f"{RESULTS_DIR}/comparison_{model_name}{few_shot_suffix}_{timestamp}.json"
        
        comparison_results = {
            "model": args.model,
            "datasets": args.datasets,
            "few_shot_count": args.few_shot,
            "mcts_iterations": args.mcts_iterations,
            "mcts_results": combined_results_mcts,
            "no_mcts_results": combined_results_no_mcts
        }
        
        with open(comparison_output, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\nComparison results saved to {comparison_output}")
        
        # Print comparison
        print("\n" + "="*50)
        print(f"Comparison Summary for {args.model}")
        print("="*50)
        
        for dataset in args.datasets:
            dataset_key = dataset.replace(".json", "")
            
            mcts_metrics = combined_results_mcts.get("per_dataset_results", {}).get(dataset, {})
            no_mcts_metrics = combined_results_no_mcts.get("per_dataset_results", {}).get(dataset, {})
            
            print(f"\nDataset: {dataset}")
            print(f"  MCTS Execution Match Accuracy: {mcts_metrics.get('execution_match_accuracy', 0):.2%}")
            print(f"  Non-MCTS Execution Match Accuracy: {no_mcts_metrics.get('execution_match_accuracy', 0):.2%}")
            print(f"  Improvement: {(mcts_metrics.get('execution_match_accuracy', 0) - no_mcts_metrics.get('execution_match_accuracy', 0)):.2%}")
            
            print(f"  MCTS Execution Success Rate: {mcts_metrics.get('execution_success_rate', 0):.2%}")
            print(f"  Non-MCTS Execution Success Rate: {no_mcts_metrics.get('execution_success_rate', 0):.2%}")
            print(f"  Improvement: {(mcts_metrics.get('execution_success_rate', 0) - no_mcts_metrics.get('execution_success_rate', 0)):.2%}")
            
            print(f"  MCTS Average Time: {mcts_metrics.get('average_time_per_query', 0):.2f} seconds")
            print(f"  Non-MCTS Average Time: {no_mcts_metrics.get('average_time_per_query', 0):.2f} seconds")
    else:
        # Normal evaluation mode
        combined_results = run_evaluation(
            args, use_mcts=not args.no_mcts, mcts_iterations=args.mcts_iterations
        )

def run_evaluation(args, use_mcts, mcts_iterations, output_prefix=""):
    """Run evaluation with specified parameters"""
    combined_results = {
        "model": args.model,
        "datasets": args.datasets,
        "few_shot_count": args.few_shot,
        "use_mcts": use_mcts,
        "mcts_iterations": mcts_iterations,
=======
        "few_shot_config": few_shot_config,
>>>>>>> Stashed changes
        "per_dataset_results": {},
        "overall_metrics": {
            "total_examples": 0,
            "exact_match_count": 0,
            "exec_match_count": 0,
            "exec_success_count": 0,
            "error_count": 0,
            "total_time": 0
        }
    }
    
    # Collect dataset results
    dataset_results = {}
    
    # Run evaluation for each dataset
    for dataset in args.datasets:
        print(f"\nEvaluating {'MCTS-' if use_mcts else ''}RARP with model {args.model} on {dataset}")
        print(f"Using {args.processes} processes for evaluation")
        
        if args.few_shot > 0:
            print(f"Few-shot learning: {args.few_shot} examples using '{args.few_shot_strategy}' strategy")
        
        if use_mcts:
            print(f"Using MCTS with {mcts_iterations} iterations")
        
        results = evaluate_dataset(
            dataset_file=dataset,
            model=args.model,
            num_samples=args.samples,
            include_samples=not args.no_sample_data,
            seed=args.seed,
            num_processes=args.processes,
<<<<<<< Updated upstream
            num_few_shot=args.few_shot,
            use_mcts=use_mcts,
            mcts_iterations=mcts_iterations,
            debug=args.debug
=======
            few_shot_config=few_shot_config
>>>>>>> Stashed changes
        )
        
        # Save individual dataset results
        if args.output:
            output_file = f"{args.output}_{output_prefix}{dataset.replace('.json', '')}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = args.model.replace("/", "-")
            dataset_name = dataset.replace(".json", "")
<<<<<<< Updated upstream
            
            mcts_suffix = f"_mcts{mcts_iterations}" if use_mcts else ""
            few_shot_suffix = f"_fewshot{args.few_shot}" if args.few_shot > 0 else ""
            
            output_file = f"{RESULTS_DIR}/{output_prefix}eval_{dataset_name}_{model_name}{mcts_suffix}{few_shot_suffix}_{timestamp}.json"
=======
            few_shot_suffix = ""
            if args.few_shot > 0:
                few_shot_suffix = f"_fewshot{args.few_shot}_{args.few_shot_strategy}"
            output_file = f"{RESULTS_DIR}/eval_{dataset_name}_{model_name}{few_shot_suffix}_{timestamp}.json"
>>>>>>> Stashed changes
        
        save_results(results, output_file)
        
        # Print error information if requested
        if args.verbose_errors and results["error_rate"] > 0:
            print("\nError details:")
            for i, result in enumerate([r for r in results["results"] if r["error"] is not None]):
                if i < 5:  # Limit to first 5 errors
                    print(f"\nError in query: {result['question']}")
                    print(f"Database: {result['db_id']}")
                    print(f"Error: {result['error']}")
                else:
                    break
            print(f"\nTotal errors: {int(results['error_rate'] * results['total_examples'])}")
        
        # Store results for later
        dataset_results[dataset] = results
        
        # Print summary for this dataset
        print_results_summary(results)
        
        # Add to combined results
        if results["total_examples"] > 0:
            combined_results["per_dataset_results"][dataset] = {
                "exact_match_accuracy": results["exact_match_accuracy"],
                "execution_match_accuracy": results["execution_match_accuracy"],
                "execution_success_rate": results["execution_success_rate"],
                "error_rate": results["error_rate"],
                "average_time_per_query": results["average_time_per_query"],
                "total_examples": results["total_examples"],
                "mcts_usage_rate": results.get("mcts_usage_rate", 0) if use_mcts else 0
            }
            
            # Update overall metrics
            combined_results["overall_metrics"]["total_examples"] += results["total_examples"]
            combined_results["overall_metrics"]["exact_match_count"] += int(results["exact_match_accuracy"] * results["total_examples"])
            combined_results["overall_metrics"]["exec_match_count"] += int(results["execution_match_accuracy"] * results["total_examples"])
            combined_results["overall_metrics"]["exec_success_count"] += int(results["execution_success_rate"] * results["total_examples"])
            combined_results["overall_metrics"]["error_count"] += int(results["error_rate"] * results["total_examples"])
            combined_results["overall_metrics"]["total_time"] += results["total_time"]
    
    # Calculate overall metrics
    total = combined_results["overall_metrics"]["total_examples"]
    if total > 0:
        combined_results["overall_metrics"]["exact_match_accuracy"] = combined_results["overall_metrics"]["exact_match_count"] / total
        combined_results["overall_metrics"]["execution_match_accuracy"] = combined_results["overall_metrics"]["exec_match_count"] / total
        combined_results["overall_metrics"]["execution_success_rate"] = combined_results["overall_metrics"]["exec_success_count"] / total
        combined_results["overall_metrics"]["error_rate"] = combined_results["overall_metrics"]["error_count"] / total
        combined_results["overall_metrics"]["average_time_per_query"] = combined_results["overall_metrics"]["total_time"] / total
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.replace("/", "-")
<<<<<<< Updated upstream
    
    mcts_suffix = f"_mcts{mcts_iterations}" if use_mcts else ""
    few_shot_suffix = f"_fewshot{args.few_shot}" if args.few_shot > 0 else ""
    
    combined_output = f"{RESULTS_DIR}/{output_prefix}eval_combined_{model_name}{mcts_suffix}{few_shot_suffix}_{timestamp}.json"
    
=======
    few_shot_suffix = ""
    if args.few_shot > 0:
        few_shot_suffix = f"_fewshot{args.few_shot}_{args.few_shot_strategy}"
    combined_output = f"{RESULTS_DIR}/eval_combined_{model_name}{few_shot_suffix}_{timestamp}.json"
>>>>>>> Stashed changes
    with open(combined_output, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    # Print combined results
    print("\n" + "="*50)
    print(f"Combined Evaluation Summary for {args.model} on {', '.join(args.datasets)}")
    print("="*50)
    print(f"Total examples evaluated: {combined_results['overall_metrics']['total_examples']}")
    print(f"Exact match accuracy: {combined_results['overall_metrics']['exact_match_accuracy']:.2%}")
    print(f"Execution match accuracy: {combined_results['overall_metrics']['execution_match_accuracy']:.2%}")
    print(f"Execution success rate: {combined_results['overall_metrics']['execution_success_rate']:.2%}")
    print(f"Error rate: {combined_results['overall_metrics']['error_rate']:.2%}")
    print(f"Average time per query: {combined_results['overall_metrics']['average_time_per_query']:.2f} seconds")
    print(f"Total evaluation time: {combined_results['overall_metrics']['total_time']:.2f} seconds")
    
    if use_mcts:
        print(f"MCTS enabled with {mcts_iterations} iterations")
    else:
        print("MCTS disabled")
    
    if args.few_shot > 0:
<<<<<<< Updated upstream
        print(f"Used {args.few_shot} few-shot examples from training set")
    
=======
        print(f"Few-shot learning: {args.few_shot} examples using '{args.few_shot_strategy}' strategy")
>>>>>>> Stashed changes
    print(f"Combined results saved to {combined_output}")
    
    return combined_results

if __name__ == "__main__":
    main()