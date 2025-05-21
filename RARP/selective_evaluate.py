#!/usr/bin/env python3
"""
Evaluation script for Selective MCTS-RARP on Spider dataset
Optimized based on performance analysis
"""

import os
import json
import argparse
import sqlite3
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import re
from datetime import datetime
from pathlib import Path
import random

# Import implementations
from working_rarp import RARP
from storm import SelectiveMCTSRARP

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

def evaluate_query(example: Dict[str, Any], model: str, use_mcts: bool = False, 
                   mcts_iterations: int = 10, debug: bool = False) -> Dict[str, Any]:
    """Evaluate a single query with RARP or Selective MCTS-RARP"""
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
        db_dir = DATABASE_DIR / db_id
        test_db_dir = SPIDER_DIR / "test_database" / db_id
        
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
                "error": f"Database {db_id} not found",
                "method": "error"
            }
        
        # Load tables info if needed for test database
        tables_path = None
        if test_db_dir.exists():
            tables_path = str(SPIDER_DIR / "test_tables.json")
        
        # Initialize appropriate RARP version
        try:
            if use_mcts:
                rarp = SelectiveMCTSRARP(db_id, model, tables_path, mcts_iterations=mcts_iterations)
            else:
                rarp = RARP(db_id, model, tables_path)
            
            # Generate SQL
            result = rarp.generate_sql(query, include_samples=True)
            generated_sql = result["sql"]
            method = result.get("method", "direct")
            
            if debug:
                if use_mcts and hasattr(rarp, "_is_highly_complex_query"):
                    is_complex = rarp._is_highly_complex_query(query)
                    print(f"\nQuery: {query}")
                    print(f"Is highly complex: {is_complex}")
                    print(f"Method used: {method}")
            
            # Evaluate match
            exact_match = False
            exec_match = False
            
            if has_gold_sql:
                # Normalize SQL for comparison
                normalized_generated = normalize_sql(generated_sql)
                normalized_gold = normalize_sql(gold_sql)
                
                # Check for exact match
                exact_match = normalized_generated == normalized_gold
                
                # Try to execute both queries
                gen_exec_success, gen_results = get_execution_result(generated_sql, str(db_path))
                gold_exec_success, gold_results = get_execution_result(gold_sql, str(db_path))
                
                # Check if both executed successfully and have matching results
                if gen_exec_success and gold_exec_success:
                    exec_match = results_match(gen_results, gold_results)
                    
                    if debug and not exec_match:
                        print(f"Execution match: {exec_match}")
                        print(f"Generated results: {gen_results[:3]} (showing up to 3)")
                        print(f"Gold results: {gold_results[:3]} (showing up to 3)")
            else:
                # If no gold SQL, just check if the generated SQL executes
                gen_exec_success, _ = get_execution_result(generated_sql, str(db_path))
            
            # Return results
            return {
                "db_id": db_id,
                "question": query,
                "gold_sql": gold_sql if has_gold_sql else "N/A",
                "generated_sql": generated_sql,
                "exact_match": exact_match,
                "exec_match": exec_match,
                "execution_success": gen_exec_success,
                "time_taken": time.time() - start_time,
                "error": None,
                "method": method,
                "mcts_used": use_mcts and method == "mcts"
            }
            
        except Exception as e:
            if debug:
                print(f"Error in RARP: {str(e)}")
            raise e
            
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
            "method": "error"
        }

def evaluate_dataset(dataset_file: str, model: str, use_mcts: bool = False,
                     mcts_iterations: int = 10, num_samples: int = None,
                     seed: int = 42, debug: bool = False) -> Dict[str, Any]:
    """Evaluate on a dataset"""
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
    
    # Sample examples if requested
    random.seed(seed)
    if num_samples is not None and num_samples < len(dataset):
        examples = random.sample(dataset, num_samples)
    else:
        examples = dataset
    
    print(f"Evaluating {len(examples)} examples from {dataset_file} with model {model}")
    print(f"MCTS: {'Enabled' if use_mcts else 'Disabled'}")
    
    # Check if dataset has gold SQL
    has_gold_sql = all("query" in example for example in examples[:5])
    
    # Evaluate each example
    results = []
    for example in tqdm(examples, desc="Evaluating"):
        result = evaluate_query(example, model, use_mcts, mcts_iterations, debug)
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
    
    # Calculate accuracy by method
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
    
    # Calculate per-database accuracy
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
    
    # Return results
    return {
        "model": model,
        "dataset": dataset_file,
        "use_mcts": use_mcts,
        "total_examples": total,
        "exact_match_accuracy": exact_matches / total if total > 0 and has_gold_sql else 0,
        "execution_match_accuracy": exec_matches / total if total > 0 and has_gold_sql else 0,
        "execution_success_rate": exec_success / total if total > 0 else 0,
        "error_rate": errors / total if total > 0 else 0,
        "average_time_per_query": avg_time,
        "total_time": total_time,
        "per_database_accuracy": db_accuracy,
        "has_gold_sql": has_gold_sql,
        "mcts_usage_rate": mcts_usage_rate,
        "direct_vs_mcts": {
            "direct_count": len(direct_results),
            "mcts_count": len(mcts_results),
            "direct_accuracy": direct_accuracy,
            "mcts_accuracy": mcts_accuracy,
            "direct_success_rate": direct_success_rate,
            "mcts_success_rate": mcts_success_rate
        },
        "results": results
    }

def save_results(results: Dict[str, Any], output_file: str = None):
    """Save evaluation results to a file"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = results["model"].replace("/", "-")
        dataset_name = results["dataset"].replace(".json", "")
        mcts_suffix = "_mcts" if results.get("use_mcts", False) else ""
        output_file = f"{RESULTS_DIR}/eval_{dataset_name}_{model_name}{mcts_suffix}_{timestamp}.json"
    
    # Save summary results
    summary = {k: v for k, v in results.items() if k != "results"}
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results separately
    detailed_output = output_file.replace(".json", "_detailed.json")
    with open(detailed_output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Summary results saved to {output_file}")
    print(f"Detailed results saved to {detailed_output}")

def print_results_summary(results: Dict[str, Any]):
    """Print a summary of evaluation results"""
    print("\n" + "="*50)
    print(f"Evaluation Summary for {results['model']} on {results['dataset']}")
    print("="*50)
    print(f"Total examples evaluated: {results['total_examples']}")
    
    if results.get("has_gold_sql", True):
        print(f"Exact match accuracy: {results['exact_match_accuracy']:.2%}")
        print(f"Execution match accuracy: {results['execution_match_accuracy']:.2%}")
    else:
        print("No gold SQL available for evaluation")
    
    print(f"Execution success rate: {results['execution_success_rate']:.2%}")
    print(f"Error rate: {results['error_rate']:.2%}")
    print(f"Average time per query: {results['average_time_per_query']:.2f} seconds")
    print(f"Total evaluation time: {results['total_time']:.2f} seconds")
    
    # Print MCTS statistics if enabled
    if results.get("use_mcts", False):
        print("\nMCTS Statistics:")
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

def run_comparison(model: str, datasets: List[str], mcts_iterations: int = 10, 
                   num_samples: int = None, seed: int = 42, debug: bool = False):
    """Run comparison between regular RARP and Selective MCTS-RARP"""
    results = {
        "model": model,
        "datasets": datasets,
        "regular_results": {},
        "mcts_results": {}
    }
    
    for dataset in datasets:
        print(f"\n=== Evaluating {model} on {dataset} ===")
        
        # Run with regular RARP
        print("\nRunning with standard RARP:")
        regular_results = evaluate_dataset(
            dataset_file=dataset,
            model=model,
            use_mcts=False,
            num_samples=num_samples,
            seed=seed,
            debug=debug
        )
        
        print_results_summary(regular_results)
        regular_output = f"{RESULTS_DIR}/eval_{dataset.replace('.json', '')}_{model.replace('/', '-')}_regular_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(regular_results, regular_output)
        
        # Run with Selective MCTS-RARP
        print("\nRunning with Selective MCTS-RARP:")
        mcts_results = evaluate_dataset(
            dataset_file=dataset,
            model=model,
            use_mcts=True,
            mcts_iterations=mcts_iterations,
            num_samples=num_samples,
            seed=seed,
            debug=debug
        )
        
        print_results_summary(mcts_results)
        mcts_output = f"{RESULTS_DIR}/eval_{dataset.replace('.json', '')}_{model.replace('/', '-')}_selective_mcts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(mcts_results, mcts_output)
        
        # Store summary results
        regular_summary = {k: v for k, v in regular_results.items() if k != "results"}
        mcts_summary = {k: v for k, v in mcts_results.items() if k != "results"}
        
        results["regular_results"][dataset] = regular_summary
        results["mcts_results"][dataset] = mcts_summary
        
        # Print comparison
        reg_exec_acc = regular_results["execution_match_accuracy"]
        mcts_exec_acc = mcts_results["execution_match_accuracy"]
        
        reg_exec_rate = regular_results["execution_success_rate"]
        mcts_exec_rate = mcts_results["execution_success_rate"]
        
        reg_time = regular_results["average_time_per_query"]
        mcts_time = mcts_results["average_time_per_query"]
        
        print("\n=== Comparison ===")
        print(f"Execution Match Accuracy:  Regular RARP: {reg_exec_acc:.2%},  Selective MCTS: {mcts_exec_acc:.2%},  Diff: {(mcts_exec_acc - reg_exec_acc):.2%}")
        print(f"Execution Success Rate:    Regular RARP: {reg_exec_rate:.2%},  Selective MCTS: {mcts_exec_rate:.2%},  Diff: {(mcts_exec_rate - reg_exec_rate):.2%}")
        print(f"Average Time Per Query:    Regular RARP: {reg_time:.3f}s,  Selective MCTS: {mcts_time:.3f}s,  Factor: {mcts_time / reg_time:.1f}x")
    
    # Save comparison results
    comparison_output = f"{RESULTS_DIR}/comparison_{model.replace('/', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComparison results saved to {comparison_output}")

def main():
    """Main function to run evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate Selective MCTS-RARP on Spider dataset")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Model to use")
    parser.add_argument("--datasets", nargs='+', default=["dev.json"], 
                      choices=["dev.json", "train_spider.json", "test.json"], 
                      help="Datasets to evaluate on (can specify multiple)")
    parser.add_argument("--samples", type=int, default=None, help="Number of examples to evaluate per dataset (None for all)")
    parser.add_argument("--output", default=None, help="Output file prefix for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--mcts", action="store_true", help="Use Selective MCTS-RARP")
    parser.add_argument("--mcts-iterations", type=int, default=10, help="Number of MCTS iterations")
    parser.add_argument("--compare", action="store_true", help="Run comparison between regular RARP and Selective MCTS-RARP")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    # Run comparison if requested
    if args.compare:
        run_comparison(
            model=args.model,
            datasets=args.datasets,
            mcts_iterations=args.mcts_iterations,
            num_samples=args.samples,
            seed=args.seed,
            debug=args.debug
        )
        return
    
    # Run single evaluation
    for dataset in args.datasets:
        print(f"\nEvaluating {'Selective MCTS-RARP' if args.mcts else 'RARP'} with model {args.model} on {dataset}")
        
        results = evaluate_dataset(
            dataset_file=dataset,
            model=args.model,
            use_mcts=args.mcts,
            mcts_iterations=args.mcts_iterations,
            num_samples=args.samples,
            seed=args.seed,
            debug=args.debug
        )
        
        # Save results
        if args.output:
            output_file = f"{args.output}_{dataset.replace('.json', '')}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = args.model.replace("/", "-")
            dataset_name = dataset.replace(".json", "")
            mcts_suffix = "_selective_mcts" if args.mcts else ""
            output_file = f"{RESULTS_DIR}/eval_{dataset_name}_{model_name}{mcts_suffix}_{timestamp}.json"
        
        save_results(results, output_file)
        print_results_summary(results)

if __name__ == "__main__":
    main()