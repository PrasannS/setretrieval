import random
import argparse
import json
from statistics import mean
from setretrieval.inference.oai_request_client import ParallelResponsesClient
from setretrieval.utils.utils import pickdump, pickload
import os

random.seed(42)

def calculate_metrics(queries, passages, preds):
    """
    Calculate false positives, false negatives, and true positives for each query.
    
    Args:
        queries: List of query strings
        passages: List of lists containing gold passage strings
        preds: List of lists containing predicted passage strings
    
    Returns:
        List of dicts with metrics for each query
    """
    metrics = []
    
    for i, (query, gold, pred) in enumerate(zip(queries, passages, preds)):
        gold_set = set(gold)
        pred_set = set(pred)
        
        true_positives = list(gold_set & pred_set)
        false_positives = list(pred_set - gold_set)
        false_negatives = list(gold_set - pred_set)
        
        # Store indices of false positives in original preds list
        fp_indices = [j for j, p in enumerate(pred) if p in false_positives]
        
        metrics.append({
            'index': i,
            'query': query,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'fp_indices': fp_indices,
            'gold': gold,
            'pred': pred, 
            'recall': len(true_positives) / min(len(gold), len(pred))
        })
    
    return metrics

def generate_comparison_prompts(metrics):
    """
    Generate comparison prompts for queries with >1 false negative.
    Each false positive is compared with a random false negative.
    
    Args:
        metrics: List of metric dicts from calculate_metrics
    
    Returns:
        List of comparison prompt dicts
    """
    prompts = []
    
    for metric in metrics:
        # Only process queries with more than one false negative
        if len(metric['false_negatives']) <= 1:
            continue
        
        index = metric['index']
        query = metric['query']
        false_positives = metric['false_positives']
        false_negatives = metric['false_negatives']
        pred = metric['pred']
        
        # For each false positive, compare with a random false negative
        for fp in false_positives:
            # Get the index of this false positive in the original preds list
            predind = pred.index(fp)
            
            # Select a random false negative
            fn = random.choice(false_negatives)
            
            # Randomize the order
            if random.random() < 0.5:
                passage1 = fp
                passage2 = fn
                first = "falsepos"
            else:
                passage1 = fn
                passage2 = fp
                first = "falseneg"
            
            # Create the prompt
            prompt = (
                f"Answer 1 or 2, no other text. Based on carefully reading and analyzing both passages, which passage is more relevant to the query, 1 or 2?\n\n"
                f"Query: {query}\n\n"
                f"Passage 1: {passage1}\n\n"
                f"Passage 2: {passage2}"
            )
            
            prompts.append({
                'prompt': prompt,
                'first': first,
                'index': index,
                'predind': predind
            })

    return prompts


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_path", type=str, default="propercache/cache/setresults/wiki_newpipetest_qwen06bsingle_preds2.jsonl")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite")
    args = parser.parse_args()


    model2keys = {
        "gemini-2.5-flash-lite": "_gemflashcomps.pkl",
        "gemini-2.5-pro": "_geminiprocomps.pkl",
    }

    assert args.model in model2keys, f"Model {args.model} not supported"

    queries, passages, preds = [], [], []
    with open(args.preds_path, "r") as f:
        for line in f:
            data = json.loads(line)
            queries.append(data["question"])
            passages.append(data["golds"])
            preds.append(data["preds"])

    
    # Calculate metrics
    metrics = calculate_metrics(queries, passages, preds)

    print(f"Mean recall: {mean([m['recall'] for m in metrics])}")

    if os.path.exists(args.preds_path.replace(".jsonl", model2keys[args.model])):
        fps_better_fns, fns_better_fps = pickload(args.preds_path.replace(".jsonl", model2keys[args.model]))
    else:
        # Generate comparison prompts
        comparison_prompts = generate_comparison_prompts(metrics)

        oai_client = ParallelResponsesClient(max_concurrent=100)

        breakpoint()
        useprompts = [p['prompt'] for p in comparison_prompts]
        results = []
        for i in range(0, len(useprompts), 1000):
            results.extend(oai_client.run(model=args.model, prompts=useprompts[i:i+1000]))
            print("Cost: ", oai_client.total_cost)

        fps_better_fns = [[] for _ in range(len(queries))]
        fns_better_fps = [[] for _ in range(len(queries))]
        # breakpoint()
        for i, r in enumerate(results):
            if r['response'] is None:
                print(f"Weirdness: {r['error']}")
                continue
            if ("1" in r['response']) == (comparison_prompts[i]['first']=="falsepos"):
                fps_better_fns[comparison_prompts[i]['index']].append(comparison_prompts[i]['predind'])
            else:
                fns_better_fps[comparison_prompts[i]['index']].append(comparison_prompts[i]['predind'])
        
        pickdump((fps_better_fns, fns_better_fps), args.preds_path.replace(".jsonl", model2keys[args.model]))
        
        breakpoint()

    # print out recall, recall if we add fps_better_fns to numerator / denominator
    nmets = []
    for i in range(len(metrics)):
        recall = metrics[i]['recall']
        fps_better_fn = fps_better_fns[i]
        fns_better_fp = fns_better_fps[i]
        nmets.append({
            'Recall @100': recall,
            'Recall @100 adjusted with false positives': (len(metrics[i]['true_positives']) + len(fps_better_fn)) / min(len(metrics[i]['gold']) + len(fps_better_fn), len(metrics[i]['pred']))
        })
    for m in nmets[0].keys():
        print(f"Mean {m}: {mean([nm[m] for nm in nmets])}")