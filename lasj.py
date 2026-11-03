#!/usr/bin/env python
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import argparse
import json_repair
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

_client = None

def _get_client():
    """Lazy client creation — reads env vars at call time, not import time."""
    global _client
    if _client is not None:
        return _client
    if os.environ.get("JUDGE_USE_VLLM") == "1":
        _client = OpenAI(
            base_url=os.getenv("JUDGE_VLLM_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "token-abc123"),
        )
    elif os.environ.get("AZURE_OPENAI") == "1":
        _azure_key = os.getenv("AZURE_API_KEY", os.getenv("OPENAI_API_KEY"))
        _client = AzureOpenAI(
            api_key=_azure_key,
            azure_endpoint=os.getenv("AZURE_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
            api_version=os.getenv("AZURE_API_VERSION", "2025-01-01-preview"),
        )
    else:
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
    return _client

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json_repair.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error：line {line_num} - {e}")
    return data

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def judge_equivalence(question, gt_answer, pred_answer, model="gpt-4o"):
    try:
        pred_answer = pred_answer["answer"]
    except Exception as e:
        pass
    prompt = f"""Are these two answers equivalent?

Question: {question}
Labeled Answer: {gt_answer}
Predicted Answer: {pred_answer}

Rules: Focus on factual content, ignore formatting. Allow small numerical margin.

Reply with ONLY one word: correct or incorrect"""
    if pred_answer == '':
        return {
            "question": question,
            "judgement": "incorrect",
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
        }
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a fair judge evaluating if two answers to a question are equivalent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        raw_content = response.choices[0].message.content.strip().lower()

        # Parse plain text — look for "correct" or "incorrect"
        if 'incorrect' in raw_content:
            judgement = 'incorrect'
        elif 'correct' in raw_content:
            judgement = 'correct'
        else:
            # Fallback: try JSON
            try:
                result = json.loads(raw_content)
                judgement = result.get('judgement', 'incorrect').strip().lower()
            except (json.JSONDecodeError, AttributeError):
                judgement = 'incorrect'

        return {
            "question": question,
            "judgement": judgement,
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
        }
        
    except Exception as e:
        print(f"Error judging equivalence: {str(e)}")
        return {
            "question": question,
            "judgement": "error",
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
        }

def process_batch(items, model, max_workers=5):

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, item in enumerate(items):

            question = item.get('question', '') if item is not None else ''
            gt_answer = item.get('golden_answer', '') if item is not None else ''
            pred_answer = item.get('agent_result', {}) if item is not None else {}
            
            futures[executor.submit(
                judge_equivalence,
                question,
                gt_answer,
                pred_answer,
                model
            )] = idx
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging answers"):
            results.append(future.result())
    return results

def calculate_accuracy(results):
    total = len(results)
    if total == 0:
        return 0.0
        
    correct = sum(1 for r in results if r['judgement'] == 'correct')
    incorrect = sum(1 for r in results if r['judgement'] == 'incorrect')
    errors = sum(1 for r in results if r['judgement'] == 'error')
    
    accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0
    
    print(f"Total items: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Errors: {errors}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{correct + incorrect})")
    
    return accuracy

def main(args):
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} items")
    
    if args.sample_size and args.sample_size < len(data):
        data = data[:args.sample_size]
        print(f"Processing first {args.sample_size} items")
    
    results = process_batch(data, args.model, args.max_workers)
    
    save_results(results, args.output_file)
    print(f"Results saved to {args.output_file}")
    calculate_accuracy(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge equivalence between predicted and labeled answers using OpenAI API")
    
    parser.add_argument("--input_file", default="./data/<example.json>", help="Path to input JSONL file containing questions, answers and agent results")
    parser.add_argument("--output_file", default="./output/<example.jsonl>", help="Path to save judgement results (JSONL)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use for judging")
    parser.add_argument("--sample_size", type=int, help="Number of items to process (optional)")
    parser.add_argument("--max_workers", type=int, default=20, help="Number of parallel workers for API calls")
    
    args = parser.parse_args()
    main(args)
