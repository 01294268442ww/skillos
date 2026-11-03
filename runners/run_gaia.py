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
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import uuid
from FlashOAgents import OpenAIServerModel
from FlashOAgents import VisualInspectorTool, TextInspectorTool, AudioInspectorTool, get_zip_description, get_single_file_description
from base_agent import MMSearchAgent
from utils import read_jsonl, write_jsonl
from lasj import judge_equivalence
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
def parse_task_indices(indices_str):
    """Parse task indices with optional level and ignore syntax."""
    if not indices_str:
        return None
    
    indices_str = indices_str.strip()
    
    # Check if using new level or ignore syntax
    if '[level' not in indices_str.lower() and '[ignore]' not in indices_str.lower():
        # Legacy mode: simple indices without level filter
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
    
    # New level/ignore syntax mode
    import re
    select_specs = []
    ignore_specs = []
    
    # Strategy: Find tags and their content in order
    # Pattern matches: [levelX][ignore], [levelX], or [ignore] followed by optional content
    
    # First pass: identify all tags with positions
    tag_pattern = r'\[level(\d+)\]|\[ignore\]'
    tags = []
    
    for match in re.finditer(tag_pattern, indices_str, re.IGNORECASE):
        if match.group(1):  # [levelX]
            tags.append({
                'type': 'level',
                'level': match.group(1),
                'start': match.start(),
                'end': match.end()
            })
        else:  # [ignore]
            tags.append({
                'type': 'ignore',
                'level': None,
                'start': match.start(),
                'end': match.end()
            })
    
    if not tags:
        raise ValueError(f"Invalid syntax: {indices_str}")
    
    # Second pass: combine adjacent level+ignore and extract indices
    i = 0
    while i < len(tags):
        tag = tags[i]
        
        # Check if this is a [levelX] followed immediately by [ignore]
        if (tag['type'] == 'level' and 
            i + 1 < len(tags) and 
            tags[i + 1]['type'] == 'ignore'):
            
            # Check if they are truly adjacent (NO space or very minimal)
            between_text = indices_str[tag['end']:tags[i + 1]['start']]
            # Only treat as combined if there's NO space at all
            if len(between_text) == 0:  # Truly adjacent, no space
                # This is [levelX][ignore]
                level_num = tag['level']
                is_ignore = True
                end_pos = tags[i + 1]['end']
                i += 2  # Skip both tags
            else:
                # They are separate (has space)
                level_num = tag['level']
                is_ignore = False
                end_pos = tag['end']
                i += 1
        elif tag['type'] == 'level':
            level_num = tag['level']
            is_ignore = False
            end_pos = tag['end']
            i += 1
        else:  # tag['type'] == 'ignore'
            level_num = None
            is_ignore = True
            end_pos = tag['end']
            i += 1
        
        # Extract indices part (from end_pos to next tag or end of string)
        if i < len(tags):
            next_start = tags[i]['start']
            indices_part = indices_str[end_pos:next_start].strip()
        else:
            indices_part = indices_str[end_pos:].strip()
        
        level_num = level_num if level_num else None
        indices_part = indices_part.strip()
        
        if not indices_part:
            # No indices specified, means all tasks of this level
            spec = {"level": level_num, "indices": None}
            if is_ignore:
                ignore_specs.append(spec)
            else:
                select_specs.append(spec)
        else:
            # Parse the indices for this level
            indices = set()
            parts = indices_part.split(',')
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                if '-' in part:
                    try:
                        start, end = part.split('-')
                        start, end = int(start.strip()), int(end.strip())
                        if start > end:
                            raise ValueError(f"Invalid range: {part} (start > end)")
                        indices.update(range(start, end + 1))
                    except ValueError as e:
                        level_str = f"level{level_num}" if level_num else "global"
                        logger.error(f"Invalid range format in {level_str}: {part}. Error: {e}")
                        raise
                else:
                    try:
                        indices.add(int(part))
                    except ValueError:
                        level_str = f"level{level_num}" if level_num else "global"
                        logger.error(f"Invalid number format in {level_str}: {part}")
                        raise
            
            if indices:  # Only add if we have valid indices
                spec = {"level": level_num, "indices": indices}
                if is_ignore:
                    ignore_specs.append(spec)
                else:
                    select_specs.append(spec)
    
    return {"select": select_specs, "ignore": ignore_specs}
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
    """Process a single GAIA task with timing and metrics tracking"""
    task_model = OpenAIServerModel(**model_config)
    task_model.reset_total_counts()
    
    memory_provider = None
    if memory_type_str:
        memory_provider = load_memory_provider(memory_type_str, task_model)
    
    visual_tool = VisualInspectorTool(task_model, 100000)
    text_tool = TextInspectorTool(task_model, 100000)
    audio_tool = AudioInspectorTool(task_model, 100000)
    
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

    question = item["Question"]
    golden_answer = item["Final answer"]
    task_id = item.get("task_id")
    level = item.get("Level", "unknown")
    original_question = question

    if item["file_name"]:
        item["file_name"] = f"data/gaia/validation/" + item["file_name"]
        try:
            if ".zip" in item["file_name"]:
                question += "\n\nTo solve the task above, you will have to use these attached files:\n"
                question += get_zip_description(
                    item["file_name"], question, visual_tool, text_tool, audio_tool,
                )
            else:
                question += "\n\nTo solve the task above, you will have to use this attached file:"
                question += get_single_file_description(
                    item["file_name"], question, visual_tool, text_tool, audio_tool,
                )
        except Exception as e:
            logger.warning(f"Failed to process attached file {item['file_name']}: {e}")
            question += f"\n\n[Attached file: {item['file_name']} - could not be processed]"
    
    try:
        result = search_agent(question)
        
        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []
        
        trajectory = result.get("agent_trajectory", [])
        
        is_correct = False
        judgement = None
        if judge_model:
            try:
                eval_res = judge_equivalence(
                    original_question,
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
                    query=original_question,
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
        
        # Capture pipeline metadata from HyperSkill provider
        pipeline_meta = None
        if memory_provider and hasattr(memory_provider, 'get_last_pipeline_meta'):
            pipeline_meta = memory_provider.get_last_pipeline_meta()

        task_result = {
            "agent_result": result.get("agent_result"),
            "judgement": judgement,
            "task_id": task_id,
            "item_index": item_index,
            "level": level,
            "question": original_question,
            "full_query": question,
            "golden_answer": golden_answer,
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
                    query=original_question,
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
            "agent_result": None,
            "judgement": None,
            "status": "error",
            "error": str(e),
            "error_traceback": error_msg,
            "task_id": task_id,
            "item_index": item_index,
            "level": level,
            "question": original_question,
            "full_query": question,
            "golden_answer": golden_answer,
            "agent_trajectory": [],
            "agent_messages": agent_messages,
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
        "max_completion_tokens": 16384,
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
        if os.environ.get("AZURE_OPENAI") == "1":
            args.judge_model = "gpt-4o"
        else:
            args.judge_model = args.model_name

    model = OpenAIServerModel(**model_config)

    if args.infile.lower().endswith('.json'):
        with open(args.infile, 'r') as f:
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

    if getattr(args, 'level', None) and not args.task_indices:
        level_str = str(args.level).strip()
        before = len(data)
        filtered = []
        for it in data:
            lv = it.get("Level") if isinstance(it, dict) else it.get("task")
            if lv is None:
                continue
            if str(lv).strip() == level_str:
                filtered.append(it)
        data = filtered
        logger.info(f"Level filter applied: level={level_str}, kept {len(data)}/{before}")

    if args.task_indices:
        try:
            parsed = parse_task_indices(args.task_indices)
            
            if isinstance(parsed, set):
                data = [data[i-1] for i in sorted(parsed) if 0 < i <= len(data)]
                logger.info(f"Selected {len(data)} tasks from indices: {args.task_indices}")
            elif isinstance(parsed, dict):
                select_specs = parsed.get("select", [])
                ignore_specs = parsed.get("ignore", [])
                
                selected_tasks = []
                if select_specs:
                    for spec in select_specs:
                        level_num = spec["level"]
                        indices = spec["indices"]
                        
                        level_filtered = []
                        for it in data:
                            lv = it.get("Level") if isinstance(it, dict) else None
                            if lv is not None and str(lv).strip() == level_num:
                                level_filtered.append(it)
                        
                        if indices is None:
                            selected_tasks.extend(level_filtered)
                            logger.info(f"Selected all {len(level_filtered)} tasks from level{level_num}")
                        else:
                            for idx in sorted(indices):
                                array_idx = idx - 1
                                if 0 <= array_idx < len(level_filtered):
                                    task = level_filtered[array_idx]
                                    selected_tasks.append(task)
                            
                            actual_count = len([i for i in indices if 0 < i <= len(level_filtered)])
                            global_indices = [level_filtered[i-1].get("_global_index") for i in sorted(indices) if 0 < i <= len(level_filtered)]
                            logger.info(f"Selected {actual_count} tasks from level{level_num} (level-relative indices: {sorted(indices)}, "
                                      f"global indices: {global_indices})")
                    
                    logger.info(f"Total selected tasks: {len(selected_tasks)}")
                else:
                    selected_tasks = data[:]
                    logger.info(f"No select specs, starting with all {len(selected_tasks)} tasks")
                
                if ignore_specs:
                    tasks_to_ignore = []
                    for spec in ignore_specs:
                        level_num = spec["level"]
                        indices = spec["indices"]
                        
                        if level_num is None:
                            if indices is None:
                                tasks_to_ignore.extend(selected_tasks)
                                logger.info(f"Ignoring all selected tasks")
                            else:
                                for it in selected_tasks:
                                    global_idx = it.get("_global_index")
                                    if global_idx and global_idx in indices:
                                        tasks_to_ignore.append(it)
                                
                                actual_count = len(tasks_to_ignore) - len([t for t in tasks_to_ignore if t not in selected_tasks or t.get("_global_index") not in indices])
                                ignored_global_indices = sorted([t.get("_global_index") for t in tasks_to_ignore if t.get("_global_index") in indices])
                                logger.info(f"Ignoring {len([t for t in selected_tasks if t.get('_global_index') in indices])} tasks with global indices: {sorted(indices)} "
                                          f"(found: {ignored_global_indices})")
                        else:
                            level_filtered = []
                            for it in selected_tasks:
                                lv = it.get("Level") if isinstance(it, dict) else None
                                if lv is not None and str(lv).strip() == level_num:
                                    level_filtered.append(it)
                            
                            if indices is None:
                                tasks_to_ignore.extend(level_filtered)
                                logger.info(f"Ignoring all {len(level_filtered)} tasks from level{level_num}")
                            else:
                                for idx in indices:
                                    array_idx = idx - 1
                                    if 0 <= array_idx < len(level_filtered):
                                        task = level_filtered[array_idx]
                                        tasks_to_ignore.append(task)
                                
                                actual_count = len([i for i in indices if 0 < i <= len(level_filtered)])
                                global_indices = [level_filtered[i-1].get("_global_index") for i in sorted(indices) if 0 < i <= len(level_filtered)]
                                logger.info(f"Ignoring {actual_count} tasks from level{level_num} (level-relative indices: {sorted(indices)}, "
                                          f"global indices: {global_indices})")
                    
                    tasks_to_ignore_set = set(id(t) for t in tasks_to_ignore)
                    data = [t for t in selected_tasks if id(t) not in tasks_to_ignore_set]
                    logger.info(f"After applying ignore specs: {len(data)} tasks remaining")
                else:
                    data = selected_tasks
            else:
                raise ValueError(f"Unexpected return type from parse_task_indices: {type(parsed)}")
                
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing task indices: {e}")
            import traceback
            traceback.print_exc()
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
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
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

    logger.info(f"Processing completed. Completed this run: {len(results)}")
    
    write_jsonl(args.outfile, results)
    logger.info(f"Results saved to {args.outfile}")
    
    report_path = os.path.join(run_dir, "report.txt")
    generate_unified_report(
        results, 
        report_path, 
        dataset_name="GAIA",
        has_levels=True,
        level_key="level"
    )
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal data generation')

    parser.add_argument('--infile', type=str, default="./data/gaia/validation/metadata.jsonl", help='input path')
    parser.add_argument('--outfile', type=str, default="./gaia_output/<example.jsonl>", help='output path')
    parser.add_argument('--sample_num', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--task_indices', type=str, default=None, 
                        help='Task indices to run. Supports: '
                             '1) Simple: "5", "1-10", "1,3,5-8,10" (no level filter), '
                             '2) Level syntax: "[level1]35-53" (level1 indices 35-53), "[level2]" (all level2), '
                             '"[level1]1,3,5 [level2] [level3]10-20" (combined), '
                             '3) Ignore syntax: "[ignore] 1,2" (ignore by global _global_index), '
                             '"[level1][ignore] 1,2" (ignore level1-relative indices 1,2), '
                             '"[level1] [ignore] 3,5,9" (select all level1, ignore global indices 3,5,9)')
    parser.add_argument('--summary_interval', type=int, default=8, help='Summary interval')
    parser.add_argument('--prompts_type', type=str, default="default", help='Type of prompts to use')
    parser.add_argument('--concurrency', type=int, default=1, help='Number of concurrency')
    parser.add_argument('--max_steps', type=int, default=40, help='Maximum number of steps')
    parser.add_argument('--judge_model', type=str, default=os.getenv('DEFAULT_JUDGE_MODEL', 'gpt-4o'), help='Model used for immediate judgement')
    parser.add_argument('--memory_provider', type=str, default=None, help='Memory provider name ("hyperskill") or omit for no memory')
    parser.add_argument('--enable_memory_evolution', action='store_true', default=True,
                        help='Enable memory system evolution (take_in_memory). Default: True')
    parser.add_argument('--disable_memory_evolution', dest='enable_memory_evolution', action='store_false',
                        help='Disable memory system evolution (skip take_in_memory)')
    parser.add_argument('--level', type=str, choices=['1','2','3'], default=None, help='Filter GAIA tasks by level before applying indices')
    parser.add_argument('--direct_output_dir', type=str, default=None, help='Direct output directory (skips timestamped nesting)')
    parser.add_argument('--model_base_url', type=str, default=None,
                        help='OpenAI-compatible API base URL (e.g., http://localhost:8000/v1 for vLLM)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name override (e.g., Qwen/Qwen3-8B)')

    parser.add_argument('--storage_prefix', type=str, default=None,
                        help='Storage prefix to isolate memory per run')
    args = parser.parse_args()

    # Selects the benchmark-specific decomposition / extraction prompts.
    os.environ.setdefault("HYPERSKILL_DOMAIN", "gaia")

    if args.storage_prefix:
        os.environ["STORAGE_PREFIX"] = args.storage_prefix
    elif not os.environ.get("STORAGE_PREFIX"):
        import uuid
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        model_name = args.model_name or os.environ.get("DEFAULT_MODEL", "unknown")
        model_short = model_name.split("/")[-1].lower()
        mem = args.memory_provider or "no_memory"
        os.environ["STORAGE_PREFIX"] = f"{model_short}/gaia/{mem}_{run_id}"

    main(args)
    