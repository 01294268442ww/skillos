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
import sys

# Allow "python runners/run_*.py" from the repository root: Python puts the
# script's own directory on sys.path, not the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import argparse
import json
import logging
import csv
import base64
import re
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import uuid
from FlashOAgents import OpenAIServerModel
from FlashOAgents import VisualInspectorTool, TextInspectorTool, AudioInspectorTool
from base_agent import MMSearchAgent
from utils import read_jsonl, write_jsonl
from hyperskill.memory_types import MemoryType, TrajectoryData, PROVIDER_MAPPING
from hyperskill.config import get_memory_config
from eval_utils import (
    TaskTimer, TokenCounter, save_task_result,
    generate_unified_report, enrich_result_with_metrics, create_run_directory
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)
def xor_decrypt(data, key):
    """XOR decrypt data with a key"""
    key_bytes = key.encode('utf-8')
    key_length = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])
def parse_task_indices(indices_str):
    """Parse index string like "5", "1-10" or "1,3,5-8,10" into a 1-based index set."""
    if not indices_str:
        return None
    
    indices = set()
    parts = indices_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = part.split('-')
                start, end = int(start.strip()), int(end.strip())
                if start > end:
                    raise ValueError(f"Invalid range: {part} (start > end)")
                indices.update(range(start, end + 1))
            except ValueError as e:
                logger.error(f"Invalid range format: {part}. Error: {e}")
                raise
        else:
            try:
                indices.add(int(part))
            except ValueError:
                logger.error(f"Invalid number format: {part}")
                raise
    
    return indices
def grade_question(question_text, correct_answer, llm_response, judge_model):
    if not llm_response:
        return 0, "", "Empty response"

    # Always use LLM judge with full response (no regex extraction)
    judge_prompt = f"""Are these two answers equivalent?

Question: {question_text}
Labeled Answer: {correct_answer}
Predicted Answer: {llm_response}

Rules: Focus on factual content, ignore formatting. Allow small numerical margin.

Reply with ONLY one word: correct or incorrect"""

    try:
        from openai import AzureOpenAI
        _key = os.environ.get("AZURE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        _judge_client = AzureOpenAI(
            api_key=_key,
            azure_endpoint=os.environ.get("AZURE_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
            api_version=os.environ.get("AZURE_API_VERSION", "2025-01-01-preview"),
        )
        resp = _judge_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        judge_response = resp.choices[0].message.content or ""
        # Strip think tags manually
        judge_response = re.sub(r'<think>.*?</think>', '', judge_response, flags=re.DOTALL | re.IGNORECASE).strip()
        logger.info(f"  Judge raw (first 200): {judge_response[:200]}")
    except Exception as e:
        logger.warning(f"Judge model call failed: {e}")
        return 0, "", f"Judge error: {e}"

    resp_lower = judge_response.lower()
    if 'incorrect' in resp_lower:
        score = 0
    elif 'correct' in resp_lower:
        score = 1
    else:
        # Fallback: try JSON
        try:
            import json_repair
            result = json_repair.loads(judge_response)
            score = 1 if result.get("judgement", "").strip().lower() == "correct" else 0
        except Exception:
            score = 0
    logger.info(f"  Judge result: score={score}")
    return score, "", judge_response[:200]
def load_memory_provider(memory_type_str, model=None):
    """Load and initialize memory provider from type string"""
    if not memory_type_str:
        return None
    
    try:
        memory_type = MemoryType(memory_type_str)
    except ValueError:
        logger.error(f"Invalid memory type: {memory_type_str}")
        return None
    
    if memory_type not in PROVIDER_MAPPING:
        logger.error(f"Memory type {memory_type_str} not found in PROVIDER_MAPPING")
        return None
    
    try:
        class_name, module_name = PROVIDER_MAPPING[memory_type]
        module = __import__(f"hyperskill.{module_name}", fromlist=[class_name])
        provider_class = getattr(module, class_name)
        config = get_memory_config(memory_type)
        if model is not None:
            try:
                # many providers expect a 'model' in config
                config["model"] = model
            except Exception:
                pass
        provider = provider_class(config=config)
        
        if not provider.initialize():
            logger.error(f"Failed to initialize memory provider: {memory_type_str}")
            return None
        
        logger.info(f"Memory provider loaded: {memory_type_str}")
        return provider
    except Exception as e:
        logger.error(f"Failed to load memory provider {memory_type_str}: {e}")
        import traceback
        traceback.print_exc()
        return None
def process_item(item, model_config, summary_interval, prompts_type, max_steps, memory_type_str=None, enable_memory_evolution=True, judge_model=None):
    """Process a single XBench task with timing and metrics tracking"""
    task_model = OpenAIServerModel(**model_config)
    task_model.reset_total_counts()

    memory_provider = None
    if memory_type_str:
        memory_provider = load_memory_provider(memory_type_str, task_model)

    timer = TaskTimer()
    timer.start()

    search_agent = MMSearchAgent(
        task_model,
        summary_interval=summary_interval,
        prompts_type=prompts_type,
        max_steps=max_steps,
        memory_provider=memory_provider,
        disable_planning=True,  # planning disabled for all runs
    )

    question = item["prompt"]
    golden_answer = item["answer"]
    task_id = item.get("id")
    
    try:
        result = search_agent(question)
        
        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []
        
        trajectory = result.get("agent_trajectory", [])
        
        is_correct = False
        score = 0
        extracted_answer = ""
        grader_explanation = ""
        if judge_model:
            try:
                agent_result = result.get("agent_result", "")
                if agent_result is None:
                    agent_response = ""
                elif isinstance(agent_result, dict):
                    agent_response = agent_result.get("answer", "")
                    if not agent_response:
                        agent_response = json.dumps(agent_result, ensure_ascii=False)
                else:
                    agent_response = str(agent_result) if agent_result else ""
                
                if not isinstance(agent_response, str):
                    agent_response = str(agent_response) if agent_response else ""
                
                score, extracted_answer, grader_explanation = grade_question(
                    question,
                    golden_answer,
                    agent_response,
                    judge_model
                )
                is_correct = (score == 1)
            except Exception as e:
                logger.warning(f"Judgement failed: {e}")
                score = 0
                extracted_answer = ""
                grader_explanation = f"Error: {str(e)}"
        
        if memory_provider and enable_memory_evolution:
            try:
                step_count = getattr(search_agent.agent_fn, 'step_number', len(agent_messages))
                trajectory_data = TrajectoryData(
                    query=question,
                    trajectory=agent_messages,
                    result=result.get("agent_result"),
                    metadata={
                        "task_id": task_id,
                        "status": "success",
                        "is_correct": is_correct,
                        "step_count": step_count,
                        "full_query": question,
                    }
                )
                success, msg = memory_provider.take_in_memory(trajectory_data)
                if success:
                    logger.info(f"Memory ingested: {msg}")
                else:
                    logger.warning(f"Memory ingestion failed: {msg}")
            except Exception as e:
                logger.warning(f"take_in_memory failed: {e}")
        
        token_counter = TokenCounter.from_model(task_model)

        pipeline_meta = None
        if memory_provider and hasattr(memory_provider, 'get_last_pipeline_meta'):
            pipeline_meta = memory_provider.get_last_pipeline_meta()

        task_result = {
            "task_id": task_id,
            "question": question,
            "full_query": question,
            "golden_answer": golden_answer,
            "status": "success",
            "agent_messages": agent_messages,
            "score": score,
            # generate_unified_report tallies on "judgement"; without it every
            # xBench run reports 0.00% regardless of the per-task scores.
            "judgement": "correct" if is_correct else "incorrect",
            "extracted_answer": extracted_answer,
            "grader_explanation": grader_explanation,
            "memory_pipeline": pipeline_meta,
            **result,
        }
        
        timer.stop()
        return enrich_result_with_metrics(task_result, timer, token_counter)
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Exception occurred while processing question: {question[:100]}...\nError: {error_msg}")
        
        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []
        
        if memory_provider and enable_memory_evolution:
            try:
                step_count = getattr(search_agent.agent_fn, 'step_number', len(agent_messages))
                trajectory_data = TrajectoryData(
                    query=question,
                    trajectory=agent_messages,
                    result=None,
                    metadata={
                        "task_id": task_id,
                        "status": "error",
                        "is_correct": False,
                        "step_count": step_count,
                        "full_query": question,
                    }
                )
                success, msg = memory_provider.take_in_memory(trajectory_data)
                if success:
                    logger.info(f"Memory ingested (error case): {msg}")
                else:
                    logger.warning(f"Memory ingestion failed (error case): {msg}")
            except Exception as e:
                logger.warning(f"take_in_memory failed (error case): {e}")
        
        pipeline_meta = None
        if memory_provider and hasattr(memory_provider, 'get_last_pipeline_meta'):
            pipeline_meta = memory_provider.get_last_pipeline_meta()

        task_result = {
            "task_id": task_id,
            "question": question,
            "full_query": question,
            "golden_answer": golden_answer,
            "status": "error",
            "error": str(e),
            "error_traceback": error_msg,
            "agent_result": None,
            "agent_trajectory": [],
            "agent_messages": agent_messages,
            "score": 0,
            "judgement": None,
            "extracted_answer": "",
            "grader_explanation": f"Error: {str(e)}",
            "memory_pipeline": pipeline_meta,
        }
        
        timer.stop()
        token_counter = TokenCounter.from_model(task_model)
        return enrich_result_with_metrics(task_result, timer, token_counter)
def main(args):
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    
    model_config = {
        "model_id": os.environ.get("DEFAULT_MODEL"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 16384 if os.environ.get("AZURE_OPENAI") == "1" else 32768,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
        "temperature": 0.0,
    }

    if args.model_base_url:
        os.environ["AZURE_OPENAI"] = "0"
        model_config["api_base"] = args.model_base_url
        model_config["api_key"] = os.environ.get("OPENAI_API_KEY", "token-abc123")
        # Cap completion tokens to avoid exceeding model context window
        model_config["max_completion_tokens"] = min(model_config["max_completion_tokens"], 8192)
    if args.model_name:
        model_config["model_id"] = args.model_name

    model = OpenAIServerModel(**model_config)

    judge_api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    judge_api_base = os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE")
    if args.model_base_url:
        judge_api_base = args.model_base_url
        judge_api_key = os.environ.get("OPENAI_API_KEY", "token-abc123")
    judge_model = OpenAIServerModel(
        args.judge_model if not args.model_name else args.model_name,
        custom_role_conversions=custom_role_conversions,
        api_key=judge_api_key,
        api_base=judge_api_base,
        max_completion_tokens=4096,
    )

    data = []
    with open(args.infile, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for question in reader:
            key = question["canary"]
            question["prompt"] = xor_decrypt(base64.b64decode(question["prompt"]), key).decode('utf-8')
            question["answer"] = xor_decrypt(base64.b64decode(question["answer"]), key).decode('utf-8')
            data.append(question)

    if args.task_indices:
        try:
            selected_indices = parse_task_indices(args.task_indices)
            data = [data[i-1] for i in sorted(selected_indices) if 0 < i <= len(data)]
            logger.info(f"Selected {len(data)} tasks from indices: {args.task_indices}")
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing task indices: {e}")
            return
    elif args.sample_num is not None:
        data = data[:args.sample_num]
    
    data_to_run = data
    logger.info(f"Total data to process: {len(data_to_run)}")

    memory_name = ""
    if args.memory_provider:
        try:
            temp_provider = load_memory_provider(args.memory_provider, model)
            if temp_provider:
                memory_name = temp_provider.get_memory_type().value + "_"
        except Exception as e:
            logger.warning(f"Failed to create temp provider for memory name: {e}")
            try:
                memory_type = MemoryType(args.memory_provider)
                memory_name = memory_type.value + "_"
            except Exception:
                pass
    
    if args.direct_output_dir:
        run_dir = args.direct_output_dir
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Using direct output directory: {run_dir}")
    else:
        out_dir = os.path.dirname(args.outfile) or "."
        base_name = os.path.splitext(os.path.basename(args.outfile))[0]
        run_dir = create_run_directory(out_dir, base_name, memory_name)
        logger.info(f"Run directory created: {run_dir}")

    results = []
    file_lock = threading.Lock()

    def safe_write(result):
        """Thread-safe result saving"""
        with file_lock:
            result_to_save = {k: v for k, v in result.items() 
                             if k not in ["agent_messages", "grader_explanation"]}
            
            ordered_result = {}
            if "score" in result_to_save:
                ordered_result["score"] = result_to_save["score"]
            if "extracted_answer" in result_to_save:
                ordered_result["extracted_answer"] = result_to_save["extracted_answer"]
            for k, v in result_to_save.items():
                if k not in ["score", "extracted_answer"]:
                    ordered_result[k] = v
            
            if not args.skip_summary:
                write_jsonl(args.outfile, [ordered_result], "a")
            
            tid = ordered_result.get("task_id") or uuid.uuid4().hex
            filename = f"{tid}.json"
            save_task_result(ordered_result, run_dir, filename)

    effective_concurrency = args.concurrency
    if args.memory_provider:
        logger.info(f"Memory provider enabled: {args.memory_provider} (each thread creates independent instance, using {effective_concurrency} workers)")
    with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
        summary_interval = random.randint(args.summary_interval - 1, args.summary_interval + 1)

        futures = [
            executor.submit(
                process_item, 
                item, 
                model_config,
                summary_interval, 
                args.prompts_type, 
                args.max_steps, 
                args.memory_provider,
                args.enable_memory_evolution,
                judge_model,
                
            ) for item in data_to_run
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    safe_write(result)
                    
                    metrics = result.get("metrics", {})
                    if result.get("status") == "success":
                        logger.info(f"Task done [{len(results)}/{len(futures)}]: {result['question'][:80]}... "
                                  f"| Time: {metrics.get('elapsed_time', 0):.1f}s | Tokens: {metrics.get('total_tokens', 0)} | Score: {result.get('score', 'N/A')}")
                    elif result.get("status") == "error":
                        logger.warning(f"Task error [{len(results)}/{len(futures)}]: {result['question'][:80]}... | Error: {result.get('error', 'Unknown')}")
            except Exception as e:
                import traceback
                logger.error(f"Failed to get result from future: {traceback.format_exc()}")

    logger.info(f"Processing completed. Completed this run: {len(results)}")
    
    report_path = os.path.join(run_dir, "report.txt")
    generate_unified_report(
        results,
        report_path,
        dataset_name="XBench",
        has_levels=False
    )
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal data generation')

    parser.add_argument('--infile', type=str, default="./xbench-evals-main/data/DeepSearch.csv", help='input path')
    parser.add_argument('--outfile', type=str, default="./output/<example.jsonl>", help='output path')
    parser.add_argument('--sample_num', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--task_indices', type=str, default=None, 
                        help='Task indices to run, supports: single number (e.g., "5"), range (e.g., "23-165"), or mixed (e.g., "1,3,5-10,20")')
    parser.add_argument('--summary_interval', type=int, default=8, help='Summary interval')
    parser.add_argument('--prompts_type', type=str, default="default", help='Type of prompts to use')
    parser.add_argument('--concurrency', type=int, default=1, help='Number of concurrency (default=1 to avoid memory provider concurrency issues)')
    parser.add_argument('--max_steps', type=int, default=40, help='Maximum number of steps')
    parser.add_argument('--memory_provider', type=str, default=None, help='Memory provider name ("hyperskill") or omit for no memory')
    parser.add_argument('--enable_memory_evolution', action='store_true', default=True,
                        help='Enable memory system evolution (take_in_memory). Default: True')
    parser.add_argument('--disable_memory_evolution', dest='enable_memory_evolution', action='store_false',
                        help='Disable memory system evolution (skip take_in_memory)')
    parser.add_argument('--skip_summary', action='store_true', help='Only save per-task json files, skip appending to summary outfile')
    parser.add_argument('--judge_model', type=str, default=os.getenv('DEFAULT_JUDGE_MODEL', 'gpt-4o'), help='Judge model id')
    parser.add_argument('--direct_output_dir', type=str, default=None, help='Direct output directory (skips timestamped nesting)')
    parser.add_argument('--model_base_url', type=str, default=None,
                        help='OpenAI-compatible API base URL (e.g., http://localhost:8000/v1 for vLLM)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name override (e.g., Qwen/Qwen3-8B)')

    parser.add_argument('--storage_prefix', type=str, default=None,
                        help='Storage prefix to isolate memory per run')
    args = parser.parse_args()

    # Selects the benchmark-specific decomposition / extraction prompts.
    os.environ.setdefault("HYPERSKILL_DOMAIN", "xbench")

    if args.storage_prefix:
        os.environ["STORAGE_PREFIX"] = args.storage_prefix
    elif not os.environ.get("STORAGE_PREFIX"):
        import uuid
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        model_name = args.model_name or os.environ.get("DEFAULT_MODEL", "unknown")
        model_short = model_name.split("/")[-1].lower()
        mem = args.memory_provider or "no_memory"
        os.environ["STORAGE_PREFIX"] = f"{model_short}/xbench/{mem}_{run_id}"

    main(args)
    