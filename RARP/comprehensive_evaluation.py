#!/usr/bin/env python3
"""
Comprehensive evaluation script for RARP (RAP + SQLCoder) on the Spider dataset
Evaluates both execution accuracy and query matching
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

# Import our RARP implementation
from working_rarp import RARP, DatabaseSchema

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

def evaluate_query(example: Dict[str, Any], model: str, include_samples: bool) -> Dict[str, Any]:
    """Evaluate a single query"""
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
            "error": "Skipped due to complexity"
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
                "error": f"Database {db_id} not found in database or test_database directories"
            }
        
        # Initialize RARP with the database
        # For test database, we need to use a different tables.json file
        if test_db_dir.exists():
            tables_path = str(SPIDER_DIR / "test_tables.json")
            rarp = RARP(db_id, model, tables_path)
        else:
            rarp = RARP(db_id, model)
        
        # Generate SQL
        result = rarp.generate_sql(query, include_samples)
        generated_sql = result["sql"]
        
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
            "error": None
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
            "error": f"{str(e)}\n{traceback_str}"
        }

def evaluate_dataset(dataset_file: str, model: str, 
                    num_samples: int = None, 
                    include_samples: bool = True,
                    seed: int = 42,
                    num_processes: int = 1) -> Dict[str, Any]:
    """Evaluate RARP on a dataset"""
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
            "results": []
        }
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample examples if num_samples is specified
    if num_samples is not None and num_samples < len(dataset):
        examples = random.sample(dataset, num_samples)
    else:
        examples = dataset
    
    print(f"Evaluating {len(examples)} examples from {dataset_file} with model {model}")
    
    # Special handling for test.json which might not have ground truth
    has_gold_sql = all("query" in example for example in examples[:5])
    
    # Evaluate in parallel if multiple processes are specified
    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.starmap(
                    evaluate_query, 
                    [(example, model, include_samples) for example in examples]
                ),
                total=len(examples)
            ))
    else:
        # Evaluate sequentially
        results = []
        for example in tqdm(examples, desc="Evaluating"):
            result = evaluate_query(example, model, include_samples)
            results.append(result)
    
    # Calculate metrics
    total = len(results)
    exact_matches = sum(1 for r in results if r["exact_match"]) if has_gold_sql else 0
    exec_matches = sum(1 for r in results if r["exec_match"]) if has_gold_sql else 0
    exec_success = sum(1 for r in results if r["execution_success"])
    errors = sum(1 for r in results if r["error"] is not None)
    
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
        "has_gold_sql": has_gold_sql
    }

def save_results(results: Dict[str, Any], output_file: str = None):
    """Save evaluation results to a file"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = results["model"].replace("/", "-")
        dataset_name = results["dataset"].replace(".json", "")
        output_file = f"{RESULTS_DIR}/eval_{dataset_name}_{model_name}_{timestamp}.json"
    
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

def main():
    """Main function to run evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate RARP on Spider dataset")
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
    args = parser.parse_args()
    
    combined_results = {
        "model": args.model,
        "datasets": args.datasets,
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
        print(f"\nEvaluating RARP with model {args.model} on {dataset}")
        print(f"Using {args.processes} processes for evaluation")
        
        results = evaluate_dataset(
            dataset_file=dataset,
            model=args.model,
            num_samples=args.samples,
            include_samples=not args.no_sample_data,
            seed=args.seed,
            num_processes=args.processes
        )
        
        # Save individual dataset results
        if args.output:
            output_file = f"{args.output}_{dataset.replace('.json', '')}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = args.model.replace("/", "-")
            dataset_name = dataset.replace(".json", "")
            output_file = f"{RESULTS_DIR}/eval_{dataset_name}_{model_name}_{timestamp}.json"
        
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
                "total_examples": results["total_examples"]
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
    combined_output = f"{RESULTS_DIR}/eval_combined_{model_name}_{timestamp}.json"
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
    print(f"Combined results saved to {combined_output}")

if __name__ == "__main__":
    main()