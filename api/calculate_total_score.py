import os
import json
import glob
import re
import argparse
from typing import Dict, List, Tuple

def normalize_dataset_name(name: str) -> str:
    """
    Standardize dataset name, remove suffixes and special characters
    
    Args:
        name: Original dataset name
        
    Returns:
        str: Standardized dataset name
    """
    # Remove timestamp and file extension
    name = re.sub(r'_\d{8}_\d{6}\.json$', '', name)
    # Remove common suffixes
    name = re.sub(r'(_benchmark|_modified)$', '', name, flags=re.IGNORECASE)
    
    # Special dataset name mapping
    name_lower = name.lower()
    if name_lower.startswith('dao2'):
        return 'dao'
    if name_lower.startswith('dao'):
        return 'dao'
    if name_lower.startswith('meme'):
        return 'meme'
    if name_lower.startswith('nft'):
        return 'nft'
    if name_lower.startswith('security') or name_lower == 'security':
        return 'security'
    if name_lower.startswith('smartcontract') or name_lower == 'smart_contract':
        return 'smart_contract'
    if name_lower.startswith('token') or name_lower == 'tokenomist':
        return 'token'
    
    # Convert to lowercase and remove spaces
    name = name_lower.replace(' ', '_')
    return name

def load_test_results(model_name: str) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, str]]:
    """
    Load subjective and objective test results for a specified model
    
    Args:
        model_name: Model name
        
    Returns:
        Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, str]]: Subjective and objective test results, and dataset name mapping
    """
    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.dirname(current_dir)  # Parent directory is test directory
    
    # Subjective results directory
    subjective_dir = os.path.join(test_dir, "test_results", model_name, "subjective")
    # Objective results directory
    objective_dir = os.path.join(test_dir, "test_results", model_name, "objective")
    
    # Load subjective results
    subjective_results = {}
    subjective_files = {}
    if os.path.exists(subjective_dir):
        for result_file in glob.glob(os.path.join(subjective_dir, "*.json")):
            file_name = os.path.basename(result_file)
            dataset_name = normalize_dataset_name(file_name)
            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
                subjective_results[dataset_name] = result
                subjective_files[dataset_name] = file_name
    
    # Load objective results
    objective_results = {}
    objective_files = {}
    if os.path.exists(objective_dir):
        for result_file in glob.glob(os.path.join(objective_dir, "*.json")):
            file_name = os.path.basename(result_file)
            # Skip all_results.json
            if file_name == "all_results.json":
                continue
            dataset_name = normalize_dataset_name(file_name)
            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
                objective_results[dataset_name] = result
                objective_files[dataset_name] = file_name
    
    # Create dataset name mapping
    dataset_mapping = {}
    for dataset_name in subjective_results.keys():
        dataset_mapping[dataset_name] = {
            "subjective_file": subjective_files.get(dataset_name, ""),
            "objective_file": objective_files.get(dataset_name, "")
        }
    
    for dataset_name in objective_results.keys():
        if dataset_name not in dataset_mapping:
            dataset_mapping[dataset_name] = {
                "subjective_file": "",
                "objective_file": objective_files.get(dataset_name, "")
            }
    
    return subjective_results, objective_results, dataset_mapping

def calculate_total_score(model_name: str) -> Dict:
    """
    Calculate the total score for a model
    
    Args:
        model_name: Model name
        
    Returns:
        Dict: Total score results
    """
    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.dirname(current_dir)  # Parent directory is test directory
    
    # Load test results
    subjective_results, objective_results, dataset_mapping = load_test_results(model_name)
    
    # Calculate scores for each dataset
    dataset_scores = {}
    total_score = 0
    
    # Get all dataset names
    all_datasets = set(list(subjective_results.keys()) + list(objective_results.keys()))
    
    if not all_datasets:
        print(f"Warning: No test result datasets found")
        return {
            "model_name": model_name,
            "total_score": 0,
            "dataset_scores": {},
            "dataset_mapping": dataset_mapping,
            "error": "No test result datasets found"
        }
    
    for dataset_name in all_datasets:
        # Get subjective score
        subjective_score = 0
        subjective_total = 0
        if dataset_name in subjective_results:
            result = subjective_results[dataset_name]
            subjective_score = result.get("total_score", 0)
            subjective_total = result.get("total_possible", 0)
            # If total_possible is 0, try to calculate from results
            if subjective_total == 0 and "results" in result:
                subjective_total = sum(item.get("max_score", 0) for item in result["results"])
        
        # Get objective score
        objective_score = 0
        objective_total = 0
        if dataset_name in objective_results:
            result = objective_results[dataset_name]
            objective_score = result.get("total_score", 0)
            objective_total = result.get("max_score", 0)  # Use max_score as the total for objective questions
        
        # Calculate total score rate = (objective score + subjective score) / (objective total + subjective total)
        total_score_value = subjective_score + objective_score
        total_possible = subjective_total + objective_total
        
        # Calculate combined score for this dataset
        dataset_score = total_score_value / total_possible if total_possible > 0 else 0
        
        dataset_scores[dataset_name] = {
            "subjective_score": subjective_score,
            "subjective_total": subjective_total,
            "objective_score": objective_score,
            "objective_total": objective_total,
            "total_score": total_score_value,
            "total_possible": total_possible,
            "dataset_score": dataset_score,
            "subjective_file": dataset_mapping[dataset_name]["subjective_file"],
            "objective_file": dataset_mapping[dataset_name]["objective_file"]
        }
        total_score += dataset_score
    
    # Calculate final score (each dataset accounts for 1/9)
    if len(dataset_scores) == 0:
        print(f"Warning: No valid test results found")
        final_score = 0
    else:
        final_score = (total_score / len(dataset_scores)) * 100
    
    # Calculate each dataset's score out of 100 points
    dataset_weights = {}
    for dataset_name in dataset_scores:
        # Each dataset's score out of 100 = dataset score * 100 / number of datasets
        dataset_weights[dataset_name] = dataset_scores[dataset_name]["dataset_score"] * 100 / len(dataset_scores)
    
    # Build results
    result = {
        "model_name": model_name,
        "total_score": final_score,
        "dataset_scores": dataset_scores,
        "dataset_mapping": dataset_mapping,
        "dataset_weights": dataset_weights,  # Add each dataset's score out of 100 points
    }

    # Save results
    result_file = os.path.join(test_dir, "test_results", f"total_score_{model_name}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Also save results to model's directory
    model_dir = os.path.join(test_dir, "test_results", model_name)
    model_result_file = os.path.join(model_dir, "total_score.json")
    with open(model_result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Total score calculated for model {model_name}: {final_score:.2f}")
    
    return result

def get_all_models() -> List[str]:
    """
    Get all model names that have test results
    
    Returns:
        List[str]: List of model names
    """
    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.dirname(current_dir)  # Parent directory is test directory
    results_dir = os.path.join(test_dir, "test_results")
    
    # Get all subdirectories in results_dir
    models = []
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path) and not item.startswith("."):
                models.append(item)
    
    return models

def main():
    parser = argparse.ArgumentParser(description="Calculate total score for models")
    parser.add_argument("--model", help="Model name to calculate score for")
    parser.add_argument("--all", action="store_true", help="Calculate scores for all models")
    args = parser.parse_args()
    
    if args.all:
        print("Calculating scores for all models...")
        models = get_all_models()
        if not models:
            print("No models found with test results.")
            return
        
        # Calculate score for each model
        all_results = {}
        for model_name in models:
            print(f"Calculating score for model {model_name}...")
            model_result = calculate_total_score(model_name)
            all_results[model_name] = model_result
        
        # Save combined results
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.dirname(current_dir)
        combined_file = os.path.join(test_dir, "test_results", "all_models_scores.json")
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"All model scores calculated and saved to {combined_file}")
    elif args.model:
        print(f"Calculating score for model {args.model}...")
        calculate_total_score(args.model)
    else:
        print("Please specify a model name with --model or use --all to calculate scores for all models.")

if __name__ == "__main__":
    main() 