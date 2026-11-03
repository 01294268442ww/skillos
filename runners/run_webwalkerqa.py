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
import json_repair
import logging
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import uuid
from FlashOAgents import OpenAIServerModel
from base_agent import SearchAgent
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
def judge_webwalkerqa_answer(question, golden_answer, pred_answer, model="gpt-4o"):
    """Judge if the predicted answer is correct for WebWalkerQA tasks."""
    from openai import OpenAI
    
    try:
        if isinstance(pred_answer, dict):
            pred_answer = pred_answer.get("answer", pred_answer)
    except Exception:
        pass
    
    if not pred_answer or (isinstance(pred_answer, str) and pred_answer.strip() == ''):
        return {
            "question": question,
            "judgement": "incorrect",
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }
    
    prompt = f"""Compare the [Response] with the [Correct Answer]. Are they consistent?

[Original Question]: {question}

[Correct Answer]: {golden_answer}

[Response]: {pred_answer}

Rules:
- Focus on whether the core factual content matches
- Ignore formatting differences
- Allow small margin of error for numbers

Reply with ONLY one word: correct or incorrect"""

    try:
        if os.environ.get("JUDGE_USE_VLLM") == "1":
            client = OpenAI(
                base_url=os.getenv("JUDGE_VLLM_URL", "http://localhost:8000/v1"),
                api_key=os.getenv("OPENAI_API_KEY", "token-abc123"),
            )
        elif os.environ.get("AZURE_OPENAI") == "1":
            from openai import AzureOpenAI
            _key = os.getenv("AZURE_API_KEY", os.getenv("OPENAI_API_KEY"))
            client = AzureOpenAI(
                api_key=_key,
                azure_endpoint=os.getenv("AZURE_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
                api_version=os.getenv("AZURE_API_VERSION", "2025-01-01-preview"),
            )
        else:
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE")
            )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a fair judge for web navigation tasks. Focus on core answer correctness, not formatting."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        result_text = response.choices[0].message.content.strip().lower()

        # Parse plain text response — look for "correct" or "incorrect"
        if 'incorrect' in result_text:
            judgement = 'incorrect'
        elif 'correct' in result_text:
            judgement = 'correct'
        else:
            # Fallback: try JSON
            try:
                result = json.loads(result_text)
                judgement = result.get('judgement', 'incorrect').strip().lower()
            except (json.JSONDecodeError, AttributeError):
                logger.warning(f"Could not parse judgement: {result_text[:100]}")
                judgement = 'incorrect'
        
        return {
            "question": question,
            "judgement": judgement,
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }
        
    except Exception as e:
        logger.error(f"Error judging answer: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "question": question,
            "judgement": "error",
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }
WEBWALKERQA_PROMPT_TEMPLATE = """You are tasked with answering a question that requires navigating through a website to find the information.

Question: {question}

Starting URL: {root_url}

Please:
1. Start from the provided root URL
2. Navigate through the website to find the information needed
3. Use web search and page crawling tools to explore the site
4. Provide a clear and accurate answer based on what you find

Important: You MUST begin by accessing {root_url}
"""
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
def process_item(item, model_config, summary_interval, prompts_type, max_steps, memory_type_str=None, item_index=None, enable_memory_evolution=True, judge_model=None):
    """Process a single WebWalkerQA task with timing and metrics tracking"""
    task_model = OpenAIServerModel(**model_config)
    task_model.reset_total_counts()

    memory_provider = None
    if memory_type_str:
        memory_provider = load_memory_provider(memory_type_str, task_model)

    timer = TaskTimer()
    timer.start()

    search_agent = SearchAgent(
        task_model,
        summary_interval=summary_interval,
        prompts_type=prompts_type,
        max_steps=max_steps,
        memory_provider=memory_provider,
        disable_planning=True,  # planning disabled for all runs
    )

    question = item.get("question", "")
    golden_answer = item.get("answer", "")
    root_url = item.get("root_url", "")
    info = item.get("info", {})
    
    domain = info.get("domain", "")
    difficulty = info.get("difficulty_level", "")
    lang = info.get("lang", "en")
    question_type = info.get("type", "")
    source_websites = info.get("source_website", [])
    golden_path = info.get("golden_path", [])
    
    enhanced_question = WEBWALKERQA_PROMPT_TEMPLATE.format(
        question=question,
        root_url=root_url
    )
    
    try:
        result = search_agent(enhanced_question)
        
        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []
        
        trajectory = result.get("agent_trajectory", [])
        
        is_correct = False
        judgement = None
        if judge_model:
            try:
                eval_res = judge_webwalkerqa_answer(
                    question,
                    golden_answer,
                    result.get("agent_result", {}),
                    model=judge_model,
                )
                judgement = eval_res.get("judgement")
                judgement_str = eval_res.get("judgement", "").strip().lower()
                is_correct = (judgement_str == "correct")
            except Exception as e:
                logger.warning(f"Judgement failed: {e}")
        
        if memory_provider and enable_memory_evolution:
            try:
                step_count = getattr(search_agent.agent_fn, 'step_number', len(agent_messages))
                trajectory_data = TrajectoryData(
                    query=question,
                    trajectory=agent_messages,
                    result=result.get("agent_result"),
                    metadata={
                        "item_index": item_index,
                        "status": "success",
                        "is_correct": is_correct,
                        "step_count": step_count,
                        "domain": domain,
                        "difficulty": difficulty,
                        "full_query": enhanced_question,
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
            "agent_result": result.get("agent_result"),
            "judgement": judgement,
            "item_index": item_index,
            "question": question,
            "enhanced_question": enhanced_question,
            "golden_answer": golden_answer,
            "root_url": root_url,
            "domain": domain,
            "difficulty": difficulty,
            "language": lang,
            "type": question_type,
            "source_websites": source_websites,
            "golden_path": golden_path,
            "status": "success",
            "agent_trajectory": trajectory,
            "agent_messages": agent_messages,
            "memory_pipeline": pipeline_meta,
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
                        "item_index": item_index,
                        "status": "error",
                        "is_correct": False,
                        "step_count": step_count,
                        "domain": domain,
                        "difficulty": difficulty,
                        "full_query": enhanced_question,
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
            "agent_result": None,
            "judgement": None,
            "status": "error",
            "error": str(e),
            "error_traceback": error_msg,
            "item_index": item_index,
            "question": question,
            "enhanced_question": enhanced_question,
            "golden_answer": golden_answer,
            "root_url": root_url,
            "domain": domain,
            "difficulty": difficulty,
            "language": lang,
            "type": question_type,
            "agent_trajectory": [],
            "agent_messages": agent_messages,
            "memory_pipeline": pipeline_meta,
        }
        
        timer.stop()
        token_counter = TokenCounter.from_model(task_model)
        return enrich_result_with_metrics(task_result, timer, token_counter)
def main(args):
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    
    # GPT-4o Azure supports max 16384 completion tokens
    default_max_tokens = 16384 if os.environ.get("AZURE_OPENAI") == "1" else 32768
    model_config = {
        "model_id": os.environ.get("DEFAULT_MODEL"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": default_max_tokens,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
        "temperature": 0.0,
    }

    if args.model_base_url:
        os.environ["AZURE_OPENAI"] = "0"
        os.environ["JUDGE_USE_VLLM"] = "1"
        os.environ["JUDGE_VLLM_URL"] = args.model_base_url
        model_config["api_base"] = args.model_base_url
        model_config["api_key"] = os.environ.get("OPENAI_API_KEY", "token-abc123")
        # Cap completion tokens to avoid exceeding model context window
        model_config["max_completion_tokens"] = min(model_config["max_completion_tokens"], 8192)
    if args.model_name:
        model_config["model_id"] = args.model_name
        args.judge_model = args.model_name

    model = OpenAIServerModel(**model_config)

    if args.infile.lower().endswith('.json'):
        with open(args.infile, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            data = []
            for idx, it in enumerate(raw):
                if isinstance(it, dict):
                    it = dict(it)
                    it["_global_index"] = idx + 1
                data.append(it)
    else:
        raw = read_jsonl(args.infile)
        data = []
        for idx, it in enumerate(raw):
            if isinstance(it, dict):
                it = dict(it)
                it["_global_index"] = idx + 1
            data.append(it)

    logger.info(f"Loaded {len(data)} items from {args.infile}")

    if args.difficulty:
        difficulty_filter = args.difficulty.lower()
        before = len(data)
        data = [
            it for it in data 
            if isinstance(it, dict) and it.get("info", {}).get("difficulty_level", "").lower() == difficulty_filter
        ]
        logger.info(f"Difficulty filter applied: difficulty={difficulty_filter}, kept {len(data)}/{before}")

    if args.lang:
        lang_filter = args.lang.lower()
        before = len(data)
        data = [
            it for it in data 
            if isinstance(it, dict) and it.get("info", {}).get("lang", "").lower() == lang_filter
        ]
        logger.info(f"Language filter applied: lang={lang_filter}, kept {len(data)}/{before}")

    if args.domain:
        domain_filter = args.domain.lower()
        before = len(data)
        data = [
            it for it in data 
            if isinstance(it, dict) and it.get("info", {}).get("domain", "").lower() == domain_filter
        ]
        logger.info(f"Domain filter applied: domain={domain_filter}, kept {len(data)}/{before}")

    if args.question_type:
        type_filter = args.question_type.lower()
        before = len(data)
        data = [
            it for it in data 
            if isinstance(it, dict) and it.get("info", {}).get("type", "").lower() == type_filter
        ]
        logger.info(f"Type filter applied: type={type_filter}, kept {len(data)}/{before}")

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
        logger.info(f"Limited to first {args.sample_num} tasks")
    
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
            idx = result.get("item_index")
            filename = f"{idx}.json" if idx is not None else None
            save_task_result(result, run_dir, filename)

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
                (item.get("_global_index") if isinstance(item, dict) else None),
                args.enable_memory_evolution,
                args.judge_model,
                
            ) for item in data_to_run
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing WebWalkerQA"):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    safe_write(result)
                    
                    metrics = result.get("metrics", {})
                    if result.get("status") == "success":
                        logger.info(f"Task done [{len(results)}/{len(futures)}]: {result['question'][:80]}... "
                                  f"| Time: {metrics.get('elapsed_time', 0):.1f}s | Tokens: {metrics.get('total_tokens', 0)}")
                    elif result.get("status") == "error":
                        logger.warning(f"Task error [{len(results)}/{len(futures)}]: {result['question'][:80]}... | Error: {result.get('error', 'Unknown')}")
            except Exception as e:
                import traceback
                logger.error(f"Failed to get result from future: {traceback.format_exc()}")

    logger.info(f"Processing completed. Total results: {len(results)}")
    
    write_jsonl(args.outfile, results)
    logger.info(f"Results saved to {args.outfile}")
    
    report_path = os.path.join(run_dir, "report.txt")
    generate_unified_report(
        results,
        report_path,
        dataset_name="WebWalkerQA",
        has_levels=True,
        level_key="difficulty"
    )
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WebWalkerQA evaluation with Flash Searcher')

    parser.add_argument('--infile', type=str, 
                        default="./data/webwalkerqa/webwalkerqa_subset_170.jsonl",
                        help='Input path (JSONL or JSON file)')
    parser.add_argument('--outfile', type=str, default="./webwalkerqa_output/webwalkerqa_results.jsonl", 
                        help='Output path for results')
    parser.add_argument('--sample_num', type=int, default=None, 
                        help='Number of samples to process')
    parser.add_argument('--task_indices', type=str, default=None, 
                        help='Task indices to run, supports: single number (e.g., "5"), range (e.g., "1-10"), or mixed (e.g., "1,3,5-10,20")')
    parser.add_argument('--summary_interval', type=int, default=8, 
                        help='Summary interval for agent')
    parser.add_argument('--prompts_type', type=str, default="default", 
                        help='Type of prompts to use')
    parser.add_argument('--concurrency', type=int, default=1, 
                        help='Number of concurrent tasks (default=1 to avoid memory provider concurrency issues)')
    parser.add_argument('--max_steps', type=int, default=40, 
                        help='Maximum number of steps for agent')
    parser.add_argument('--judge_model', type=str, default=os.getenv('DEFAULT_JUDGE_MODEL', 'gpt-4o'),
                        help='Model used for answer judgement')
    parser.add_argument('--memory_provider', type=str, default=None, 
                        help='Memory provider type (e.g., "cerebra_fusion", "riva_memory") or None')
    parser.add_argument('--enable_memory_evolution', action='store_true', default=True,
                        help='Enable memory system evolution (take_in_memory). Default: True')
    parser.add_argument('--disable_memory_evolution', dest='enable_memory_evolution', action='store_false',
                        help='Disable memory system evolution (skip take_in_memory)')
    
    # WebWalkerQA specific filters
    parser.add_argument('--difficulty', type=str, choices=['easy', 'medium', 'hard'], default=None,
                        help='Filter tasks by difficulty level')
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], default=None,
                        help='Filter tasks by language')
    parser.add_argument('--domain', type=str, default=None,
                        help='Filter tasks by domain (e.g., conference, game, organization)')
    parser.add_argument('--question_type', type=str, choices=['single_source', 'multi_source'], default=None,
                        help='Filter tasks by question type')
    parser.add_argument('--direct_output_dir', type=str, default=None, help='Direct output directory (skips timestamped nesting)')
    parser.add_argument('--model_base_url', type=str, default=None,
                        help='OpenAI-compatible API base URL (e.g., http://localhost:8000/v1 for vLLM)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name override (e.g., Qwen/Qwen3-8B)')

    parser.add_argument('--storage_prefix', type=str, default=None,
                        help='Storage prefix to isolate memory per run')
    args = parser.parse_args()

    # Selects the benchmark-specific decomposition / extraction prompts.
    os.environ.setdefault("HYPERSKILL_DOMAIN", "webwalkerqa")

    if args.storage_prefix:
        os.environ["STORAGE_PREFIX"] = args.storage_prefix
    elif not os.environ.get("STORAGE_PREFIX"):
        import uuid
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        model_name = args.model_name or os.environ.get("DEFAULT_MODEL", "unknown")
        model_short = model_name.split("/")[-1].lower()
        mem = args.memory_provider or "no_memory"
        os.environ["STORAGE_PREFIX"] = f"{model_short}/webwalkerqa/{mem}_{run_id}"

    main(args)

