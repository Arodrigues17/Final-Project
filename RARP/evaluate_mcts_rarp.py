#!/usr/bin/env python3
"""
Evaluation script for MCTS-RARP on the Spider dataset
Extends the comprehensive_evaluation.py script to support MCTS-based SQL generation
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

# Import our RARP and MCTSRARP implementations
from working_rarp import RARP, DatabaseSchema
from mcts_rarp import MCTSRARP

# Set paths
SPIDER_DIR = Path("../datasets/spider")
DATABASE_DIR = SPIDER_DIR / "database"
RESULTS_DIR = Path("./evaluation_results")

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison"""
    # Convert to lowercase
    sql = sql.lower()
    
    # Remove comments
    sql = re.sub(r'--.*', '', sql)
    
    # Remove extra whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Remove trailing semicolons
    sql = sql.rstrip(';')
    
    # Normalize aliases (this is a simplified version)
    sql = re.sub(r'(\s+as\s+)[a-zA-Z0-9_]+', r'\1alias', sql)
    
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

def evaluate_query(example: Dict[str, Any], model: str, include_samples: bool, few_shot_examples: List[Dict[str, Any]] = None, use_mcts: bool = True, mcts_iterations: int = 25, debug: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single query using MCTSRARP if requested
    
    Args:
        example: The example to evaluate
        model: The model to use
        include_samples: Whether to include sample data
        few_shot_examples: Examples for few-shot learning
        use_mcts: Whether to use MCTS
        mcts_iterations: Number of MCTS iterations
        debug: Print debug information
        
    Returns:
        Evaluation results
    """
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
        
        # Initialize appropriate RARP version
        if use_mcts:
            # For test database, we need to use a different tables.json file
            if test_db_dir.exists():
                tables_path = str(SPIDER_DIR / "test_tables.json")
                rarp = MCTSRARP(db_id, model, tables_path=tables_path, mcts_iterations=mcts_iterations)
            else:
                rarp = MCTSRARP(db_id, model, mcts_iterations=mcts_iterations)
            
            if debug:
                print(f"\nQuery: {query}")
                print(f"Database: {db_id}")
                print(f"Is complex: {rarp._is_complex_query(query)}")
        else:
            # Use regular RARP
            if test_db_dir.exists():
                tables_path = str(SPIDER_DIR / "test_tables.json")
                rarp = RARP(db_id, model, tables_path)
            else:
                rarp = RARP(db_id, model)
        
        # Generate SQL
        result = rarp.generate_sql(query, include_samples)
        generated_sql = result["sql"]
        method = result.get("method", "direct")
        
        if debug:
            print(f"Method used: {method}")
            print(f"Generated SQL: {generated_sql[:100]}{'...' if len(generated_sql) > 100 else ''}")
        
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
            "method": method
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

def evaluate_dataset(dataset_file: str, model: str, 
                    num_samples: int = None, 
                    include_samples: bool = True,
                    seed: int = 42,
                    num_processes: int = 1,
                    few_shot_examples: List[Dict[str, Any]] = None,
                    use_mcts: bool = True,
                    mcts_iterations: int = 25,
                    debug: bool = False) -> Dict[str, Any]:
    """Evaluate RARP/MCTSRARP on a dataset"""
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
            "mcts_used": use_mcts,
            "mcts_iterations": mcts_iterations if use_mcts else 0,
            "mcts_usage_rate": 0,
            "per_database_accuracy": {},
            "results": []
        }
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample examples if num_samples is specified
    if num_samples is not None and num_samples < len(dataset):
        examples = random.sample(dataset, num_samples)
    else:
        examples = dataset
    
    # Print information about the evaluation
    print(f"Evaluating {len(examples)} examples from {dataset_file} with model {model}")
    if use_mcts:
        print(f"MCTS: Enabled, Iterations: {mcts_iterations}")
    else:
        print("MCTS: Disabled (using standard RARP)")
    
    # Special handling for test.json which might not have ground truth
    has_gold_sql = all("query" in example for example in examples[:5])
    
    # Evaluate in parallel if multiple processes are specified
    if num_processes > 1 and not debug:  # Don't use multiprocessing in debug mode
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.starmap(
                    evaluate_query, 
                    [(example, model, include_samples, few_shot_examples, use_mcts, mcts_iterations, False) for example in examples]
                ),
                total=len(examples),
                desc="Evaluating"
            ))
    else:
        # Evaluate sequentially
        results = []
        for example in tqdm(examples, desc="Evaluating"):
            result = evaluate_query(example, model, include_samples, few_shot_examples, use_mcts, mcts_iterations, debug)
            results.append(result)
    
    # Calculate metrics
    total = len(results)
    exact_matches = sum(1 for r in results if r["exact_match"]) if has_gold_sql else 0
    exec_matches = sum(1 for r in results if r["exec_match"]) if has_gold_sql else 0
    exec_success = sum(1 for r in results if r["execution_success"])
    errors = sum(1 for r in results if r["error"] is not None)
    
    # Calculate MCTS usage statistics
    mcts_used = sum(1 for r in results if r.get("method") == "mcts")
    mcts_usage_rate = mcts_used / total if total > 0 else 0
    
    # Calculate method-specific performance
    direct_results = [r for r in results if r.get("method") == "direct"]
    mcts_results = [r for r in results if r.get("method") == "mcts"]
    
    direct_count = len(direct_results)
    mcts_count = len(mcts_results)
    
    direct_exec_match = sum(1 for r in direct_results if r["exec_match"]) if has_gold_sql else 0
    mcts_exec_match = sum(1 for r in mcts_results if r["exec_match"]) if has_gold_sql else 0
    
    direct_exec_success = sum(1 for r in direct_results if r["execution_success"])
    mcts_exec_success = sum(1 for r in mcts_results if r["execution_success"])
    
    direct_accuracy = direct_exec_match / direct_count if direct_count > 0 and has_gold_sql else 0
    mcts_accuracy = mcts_exec_match / mcts_count if mcts_count > 0 and has_gold_sql else 0
    
    direct_success_rate = direct_exec_success / direct_count if direct_count > 0 else 0
    mcts_success_rate = mcts_exec_success / mcts_count if mcts_count > 0 else 0
    
    # Calculate average time
    total_time = sum(r["time_taken"] for r in results)
    avg_time = total_time / total if total > 0 else 0
    
    # Group results by database
    db_metrics = defaultdict(lambda: {"total": 0, "exact": 0, "exec": 0, "success": 0})
    for r in results:
        db_id = r["db_id"]
        db_metrics[db_id]["total"] += 1
        if has_gold_sql:
            db_metrics[db_id]["exact"] += 1 if r["exact_match"] else 0
            db_metrics[db_id]["exec"] += 1 if r["exec_match"] else 0
        db_metrics[db_id]["success"] += 1 if r["execution_success"] else 0
    
    # Calculate per-database accuracy
    db_accuracy = {}
    for db_id, metrics in db_metrics.items():
        db_accuracy[db_id] = {
            "exact_match_accuracy": metrics["exact"] / metrics["total"] if metrics["total"] > 0 and has_gold_sql else 0,
            "execution_match_accuracy": metrics["exec"] / metrics["total"] if metrics["total"] > 0 and has_gold_sql else 0,
            "execution_success_rate": metrics["success"] / metrics["total"] if metrics["total"] > 0 else 0,
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
        "mcts_used": use_mcts,
        "mcts_iterations": mcts_iterations if use_mcts else 0,
        "mcts_usage_rate": mcts_usage_rate,
        "mcts_statistics": {
            "direct_generation_count": direct_count,
            "mcts_generation_count": mcts_count,
            "direct_generation_accuracy": direct_accuracy,
            "mcts_generation_accuracy": mcts_accuracy,
            "direct_generation_success_rate": direct_success_rate,
            "mcts_generation_success_rate": mcts_success_rate
        }
    }

def save_results(results: Dict[str, Any], output_file: str = None):
    """Save evaluation results to a file"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = results["model"].replace("/", "-")
        dataset_name = results["dataset"].replace(".json", "")
        mcts_suffix = f"_mcts{results['mcts_iterations']}" if results.get("mcts_used", False) else ""
        output_file = f"{RESULTS_DIR}/eval_{dataset_name}_{model_name}{mcts_suffix}_{timestamp}.json"
    
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
    
    # Print MCTS statistics if available
    if "mcts_iterations" in results and results["mcts_iterations"] > 0:
        print("\nMCTS Statistics:")
        print(f"MCTS Iterations: {results['mcts_iterations']}")
        print(f"MCTS Usage Rate: {results['mcts_usage_rate']:.2%}")
        
        # Print method comparison if available
        if "mcts_statistics" in results:
            stats = results["mcts_statistics"]
            print("\nDirect vs MCTS Comparison:")
            print(f"Direct Generation Count: {stats['direct_generation_count']}")
            print(f"MCTS Generation Count: {stats['mcts_generation_count']}")
            
            if results.get("has_gold_sql", True):
                print(f"Direct Generation Accuracy: {stats['direct_generation_accuracy']:.2%}")
                print(f"MCTS Generation Accuracy: {stats['mcts_generation_accuracy']:.2%}")
            
            print(f"Direct Generation Success Rate: {stats['direct_generation_success_rate']:.2%}")
            print(f"MCTS Generation Success Rate: {stats['mcts_generation_success_rate']:.2%}")
    
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
                print(f"Method: {failure.get('method', 'direct')}")
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
            print(f"Method: {result.get('method', 'direct')}")

def load_few_shot_examples(num_examples: int = 3, seed: int = 42) -> List[Dict[str, Any]]:
    """Load few-shot examples from the training set"""
    if num_examples <= 0:
        return []
        
    try:
        with open(SPIDER_DIR / "train_spider.json", 'r') as f:
            train_data = json.load(f)
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Filter to examples with shorter queries and successful execution
        valid_examples = []
        for example in train_data:
            if len(example["query"]) < 200 and "EXCEPT" not in example["query"] and "INTERSECT" not in example["query"]:
                db_id = example["db_id"]
                db_path = DATABASE_DIR / db_id / f"{db_id}.sqlite"
                if db_path.exists():
                    exec_success, _ = get_execution_result(example["query"], str(db_path))
                    if exec_success:
                        valid_examples.append(example)
        
        # Select random examples
        selected_examples = random.sample(valid_examples, min(num_examples, len(valid_examples)))
        
        return selected_examples
    except Exception as e:
        print(f"Error loading few-shot examples: {e}")
        return []

def run_evaluation(args, use_mcts: bool, mcts_iterations: int, output_prefix: str = ""):
    """Run evaluation with specified parameters"""
    # Load few-shot examples if requested
    few_shot_examples = None
    if args.few_shot > 0:
        few_shot_examples = load_few_shot_examples(args.few_shot, args.few_shot_seed)
        print(f"Loaded {len(few_shot_examples)} few-shot examples from training set")
    
    combined_results = {
        "model": args.model,
        "datasets": args.datasets,
        "mcts_used": use_mcts,
        "mcts_iterations": mcts_iterations if use_mcts else 0,
        "per_dataset_results": {},
        "overall_metrics": {
            "total_examples": 0,
            "exact_match_count": 0,
            "exec_match_count": 0,
            "exec_success_count": 0,
            "error_count": 0,
            "total_time": 0,
            "mcts_used_count": 0
        }
    }
    
    # Run evaluation for each dataset
    for dataset in args.datasets:
        print(f"\nEvaluating {'MCTS-' if use_mcts else ''}RARP with model {args.model} on {dataset}")
        print(f"Using {args.processes} processes for evaluation")
        
        results = evaluate_dataset(
            dataset_file=dataset,
            model=args.model,
            num_samples=args.samples,
            include_samples=not args.no_sample_data,
            seed=args.seed,
            num_processes=args.processes,
            few_shot_examples=few_shot_examples,
            use_mcts=use_mcts,
            mcts_iterations=mcts_iterations,
            debug=args.debug
        )
        
        # Save individual dataset results
        if args.output:
            output_file = f"{args.output}_{dataset.replace('.json', '')}{output_prefix}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = args.model.replace("/", "-")
            dataset_name = dataset.replace(".json", "")
            mcts_suffix = f"_mcts{mcts_iterations}" if use_mcts else ""
            output_file = f"{RESULTS_DIR}/eval_{dataset_name}_{model_name}{mcts_suffix}_{timestamp}.json"
        
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
                "mcts_usage_rate": results.get("mcts_usage_rate", 0)
            }
            
            # Update overall metrics
            combined_results["overall_metrics"]["total_examples"] += results["total_examples"]
            combined_results["overall_metrics"]["exact_match_count"] += int(results["exact_match_accuracy"] * results["total_examples"])
            combined_results["overall_metrics"]["exec_match_count"] += int(results["execution_match_accuracy"] * results["total_examples"])
            combined_results["overall_metrics"]["exec_success_count"] += int(results["execution_success_rate"] * results["total_examples"])
            combined_results["overall_metrics"]["error_count"] += int(results["error_rate"] * results["total_examples"])
            combined_results["overall_metrics"]["total_time"] += results["total_time"]
            combined_results["overall_metrics"]["mcts_used_count"] += int(results.get("mcts_usage_rate", 0) * results["total_examples"])
    
    # Calculate overall metrics
    total = combined_results["overall_metrics"]["total_examples"]
    if total > 0:
        combined_results["overall_metrics"]["exact_match_accuracy"] = combined_results["overall_metrics"]["exact_match_count"] / total
        combined_results["overall_metrics"]["execution_match_accuracy"] = combined_results["overall_metrics"]["exec_match_count"] / total
        combined_results["overall_metrics"]["execution_success_rate"] = combined_results["overall_metrics"]["exec_success_count"] / total
        combined_results["overall_metrics"]["error_rate"] = combined_results["overall_metrics"]["error_count"] / total
        combined_results["overall_metrics"]["average_time_per_query"] = combined_results["overall_metrics"]["total_time"] / total
        combined_results["overall_metrics"]["mcts_usage_rate"] = combined_results["overall_metrics"]["mcts_used_count"] / total
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.replace("/", "-")
    mcts_suffix = f"_mcts{mcts_iterations}" if use_mcts else ""
    combined_output = f"{RESULTS_DIR}/eval_combined_{model_name}{mcts_suffix}_{timestamp}.json"
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
        print(f"MCTS usage rate: {combined_results['overall_metrics']['mcts_usage_rate']:.2%}")
        
    print(f"Combined results saved to {combined_output}")
    
    return combined_results

def main():
    """Main function to run evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate MCTS-RARP on Spider dataset")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Groq model to use")
    parser.add_argument("--datasets", nargs='+', default=["dev.json"], 
                      choices=["dev.json", "train_spider.json", "test.json"], 
                      help="Datasets to evaluate on (can specify multiple)")
    parser.add_argument("--samples", type=int, default=None, help="Number of examples to evaluate per dataset (None for all)")
    parser.add_argument("--no-sample-data", action="store_true", help="Don't include sample data in context")
    parser.add_argument("--output", default=None, help="Output file prefix for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--processes", type=int, default=1, help="Number of processes to use")
    parser.add_argument("--verbose-errors", action="store_true", help="Print detailed error information")
    parser.add_argument("--few-shot", type=int, default=0, 
                       help="Number of few-shot examples to include (0 for none, 2-3 recommended)")
    parser.add_argument("--few-shot-seed", type=int, default=42, 
                       help="Random seed for few-shot example selection")
    parser.add_argument("--no-mcts", action="store_true", help="Disable MCTS (use regular RARP)")
    parser.add_argument("--mcts-iterations", type=int, default=25, help="Number of MCTS iterations")
    parser.add_argument("--compare", action="store_true", 
                      help="Run both MCTS and non-MCTS versions for comparison")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    # Run evaluation
    if args.compare:
        # Run both MCTS and non-MCTS versions
        print("=== Running standard RARP evaluation ===")
        standard_results = run_evaluation(args, False, 0, "_standard")
        
        print("\n=== Running MCTS-RARP evaluation ===")
        mcts_results = run_evaluation(args, True, args.mcts_iterations, "_mcts")
        
        # Compare results
        print("\n" + "="*50)
        print("Comparison: Standard RARP vs MCTS-RARP")
        print("="*50)
        
        if standard_results["overall_metrics"]["total_examples"] > 0 and mcts_results["overall_metrics"]["total_examples"] > 0:
            print(f"Execution match accuracy - Standard: {standard_results['overall_metrics']['execution_match_accuracy']:.2%}, MCTS: {mcts_results['overall_metrics']['execution_match_accuracy']:.2%}")
            print(f"Execution success rate - Standard: {standard_results['overall_metrics']['execution_success_rate']:.2%}, MCTS: {mcts_results['overall_metrics']['execution_success_rate']:.2%}")
            print(f"Average time per query - Standard: {standard_results['overall_metrics']['average_time_per_query']:.2f}s, MCTS: {mcts_results['overall_metrics']['average_time_per_query']:.2f}s")
    else:
        # Run single evaluation
        use_mcts = not args.no_mcts
        mcts_iterations = args.mcts_iterations if use_mcts else 0
        run_evaluation(args, use_mcts, mcts_iterations)

if __name__ == "__main__":
    main()