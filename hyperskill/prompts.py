"""Task-decomposition, skill-extraction and merge prompts for HyperSkill.

Three prompt families, each with one variant per benchmark domain plus a
domain-agnostic ``general`` fallback:

* ``INITIAL_PLAN_PROMPTS``      -- decompose a task into subtasks P_0 (Sec. 3.3)
* ``SUCCESS_EXTRACTION_PROMPTS``/``FAILURE_EXTRACTION_PROMPTS``
                                -- outcome-conditioned node extraction (Sec. 3.4)
* ``MERGE_PROMPT``              -- structure-informed merging (Sec. 3.5)

The domain is selected by ``HYPERSKILL_DOMAIN`` (set automatically by the
runners); unknown domains fall back to ``general``.
"""

DOMAINS = ("general", "gaia", "webwalkerqa", "xbench")


# --------------------------------------------------------------------------
# Task decomposition
# --------------------------------------------------------------------------

INITIAL_PLAN_PROMPTS = {
    "general": """\
Decompose this task into 2-4 SHORT steps. Minimize steps to save context window.

Available tools:
- web_search(query): Search with SPECIFIC keywords (names, dates, exact phrases)
- crawl_page(url): Read a URL - use ONLY when you need specific page content
- inspect_file_as_text(path): Read attached files
- inspect_file_as_image(path): View images/figures
- python code: Calculations, counting, data processing

Key principles:
- If a file is attached, read it FIRST
- Use web_search with the MOST specific terms possible
- Prefer python code for calculations over searching

Task: {query}

Return ONLY a JSON array: [{{"name": "action_name", "description": "use <tool> to <specific action>"}}]""",

    "gaia": """\
Decompose this task into 1-3 SHORT steps. Minimize steps to save context window.

Available tools:
- web_search(query): Search with SPECIFIC keywords (names, dates, exact phrases)
- crawl_page(url): Read a URL - use ONLY when you need specific page content
- inspect_file_as_text(path): Read attached files - START HERE if task has a file
- inspect_file_as_image(path): View images/figures
- python code: Calculations, counting, data processing

Key principles:
- If a file is attached, read it FIRST - the answer may be inside
- Use web_search with the MOST specific terms possible
- Prefer python code for calculations over searching
- Do NOT crawl pages unless absolutely necessary

Task: {query}

Return ONLY a JSON array: [{{"name": "action_name", "description": "use <tool> to <specific action>"}}]""",

    "webwalkerqa": """\
Decompose this web navigation task into 2-4 steps.

Available tools:
- web_search(query): Search for specific pages on the website
- crawl_page(url): Read content from a URL

Navigation strategy: ALWAYS start by crawling the root URL to understand site structure.
Then navigate to the relevant section (News, Awards, Faculty, etc.) before searching for specifics.

Example for "Find what award Prof X won in 2021 on university.edu":
[
  {{"name": "explore_site", "description": "crawl_page(root_url) to find navigation menu and identify relevant sections"}},
  {{"name": "navigate_section", "description": "crawl_page(root_url/awards) to find awards listing"}},
  {{"name": "find_target", "description": "web_search('Prof X award 2021 site:university.edu') to locate specific page"}}
]

Task: {query}

Return ONLY a JSON array.""",

    "xbench": """\
Decompose this data task into 2-4 steps. Each step must specify WHICH DATA to get and HOW to process it.

Available tools:
- web_search(query): Search for data sources
- crawl_page(url): Read data from specific URLs
- python code: Run calculations, unit conversions, comparisons

Task: {query}

Return ONLY a JSON array: [{{"name": "action_name", "description": "use <tool> to <specific data action>"}}]""",
}


# --------------------------------------------------------------------------
# Outcome-conditioned extraction
# --------------------------------------------------------------------------

_GENERALIZATION_RULES_BASE = """
CRITICAL RULES:
- Names must be SHORT reusable pattern names (3-5 words), NOT task-specific.
  Bad: "Search for Zip Codes". Good: "Targeted Database Search".
- Content must be a COMPLETE paragraph (2-4 sentences) that covers:
  WHAT to do, WHEN to apply it, and WHY it works (or fails).
  The content must NEVER be empty.
- Extract 1-3 items max. Quality over quantity.
- Do NOT include task-specific details (exact names, IDs, URLs). Make knowledge transferable."""

_DOMAIN_EXAMPLES = {
    "general": """
SKILL EXAMPLE:
"When a task requires a precise figure, identify the authoritative source that publishes it and query that source directly rather than relying on secondary summaries. This works because primary sources report exact definitions and units, which summaries frequently distort. Apply this whenever the answer is a specific number, date, or identifier."

MISTAKE EXAMPLE:
"Committing to the first plausible candidate entity without verifying it against a second independent source leads to confidently wrong answers. This happens because early search results are ranked by popularity rather than correctness. Instead, confirm the entity with one corroborating source before building further steps on it."
""",
    "gaia": """
SKILL EXAMPLE:
"When searching for academic papers, use web_search with the paper title, author name, and publication year as keywords. This approach works because specific metadata narrows results to the exact paper, avoiding irrelevant matches. Apply this whenever the task requires locating a specific document or dataset."

MISTAKE EXAMPLE:
"Relying on generic search terms like 'fish species location' instead of querying specific databases like USGS NAS leads to incomplete data. This happens because general search engines return popular results rather than authoritative databases. Instead, identify the relevant domain-specific database first, then query it directly."
""",
    "webwalkerqa": """
SKILL EXAMPLE:
"For university or organization websites, start by crawling the root URL to identify the navigation menu structure, then navigate to the relevant section (News, Faculty, Awards) before searching for specifics. This works because most institutional sites organize content hierarchically, and understanding the structure prevents random URL guessing."

MISTAKE EXAMPLE:
"Attempting to crawl specific deep URLs without first exploring the site structure leads to 404 errors and wasted steps. This happens because URL patterns vary across websites. Instead, always start from the homepage and follow internal navigation links to discover the correct path."
""",
    "xbench": """
SKILL EXAMPLE:
"For financial data like commodity prices, search the official exchange website directly rather than third-party aggregators. Official sources provide exact contract specifications and settlement prices that aggregators often misreport. Apply this whenever the task requires precise numerical data from regulated markets."

MISTAKE EXAMPLE:
"Using closing prices instead of settlement prices when calculating contract spreads leads to incorrect results. This happens because different price types are reported differently across sources. Instead, verify which price type the question asks for and confirm the data source reports that specific type."
""",
}

_TASK_LABEL = {
    "general": "task",
    "gaia": "research/reasoning task",
    "webwalkerqa": "web navigation task",
    "xbench": "data retrieval/analysis task",
}

_STRATEGY_LABEL = {
    "general": "strategies",
    "gaia": "search/reasoning strategies",
    "webwalkerqa": "navigation strategies",
    "xbench": "data strategies",
}

_SUCCESS_JSON_TEMPLATE = """
Extract as JSON:
{{
  "skills": [{{"name": "Short Skill Name", "content": "YOUR 2-4 sentence paragraph here. MUST describe what to do, when to apply it, and why it works. NEVER leave empty."}}],
  "knowledge_fragment": "ONE OR TWO sentences stating the core transferable lesson from this episode. State the reusable pattern, NOT the specific details. Example: 'When official exchange data is rate-limited, cross-verified third-party aggregators provide a reliable fallback for numerical financial data.'"
}}
Return ONLY valid JSON."""

_FAILURE_JSON_TEMPLATE = """
Extract as JSON:
{{
  "mistakes": [{{"name": "Short Mistake Name", "content": "YOUR 2-4 sentence paragraph here. MUST describe what fails, why it happens, and what to do instead. NEVER leave empty."}}],
  "knowledge_fragment": "ONE OR TWO sentences stating the core failure mode to avoid. State the reusable lesson, NOT the specific details. Example: 'Relying on generic search terms instead of querying domain-specific databases yields incomplete or irrelevant data.'"
}}
Return ONLY valid JSON."""


def _build_extraction_prompt(domain: str, success: bool) -> str:
    header = (
        f"Extract reusable knowledge from this SUCCESSFUL {_TASK_LABEL[domain]}."
        if success else
        f"Extract reusable lessons from this FAILED {_TASK_LABEL[domain]}."
    )
    instruction = (
        f"Extract skills (what {_STRATEGY_LABEL[domain]} worked well)."
        if success else
        "Extract mistakes (what went wrong and how to avoid it)."
    )
    return (
        header + "\n"
        + _GENERALIZATION_RULES_BASE + _DOMAIN_EXAMPLES[domain] + "\n"
        + "\nTask: {query}\nTrajectory: {trajectory}\nResult: {result}\n\n"
        + instruction + "\n"
        + (_SUCCESS_JSON_TEMPLATE if success else _FAILURE_JSON_TEMPLATE)
    )


SUCCESS_EXTRACTION_PROMPTS = {d: _build_extraction_prompt(d, True) for d in DOMAINS}
FAILURE_EXTRACTION_PROMPTS = {d: _build_extraction_prompt(d, False) for d in DOMAINS}


# --------------------------------------------------------------------------
# Structure-informed merging (Sec. 3.5)
# --------------------------------------------------------------------------

MERGE_PROMPT = """\
These two {type}s are structurally and semantically related. Consolidate them into ONE higher-order {type} that captures the knowledge from both.

{TYPE} A:
Name: {name_a}
Content: {content_a}

{TYPE} B:
Name: {name_b}
Content: {content_b}

Create a consolidated {type} that:
- Has a SHORT reusable name (3-5 words)
- Has content (2-4 sentences) covering BOTH strategies/patterns
- Is more general and widely applicable than either individual {type}

Reply with ONLY JSON:
{{"name": "Consolidated Name", "content": "Consolidated content covering both."}}"""


# --------------------------------------------------------------------------
# Section headers of the memory block handed to the agent
# --------------------------------------------------------------------------

EXPERIENCE_HEADER = "## Past Experience"
SKILL_HEADER = "## Relevant Skills"
MISTAKE_HEADER = "## Mistakes to Avoid"


def _resolve(domain: str) -> str:
    """Map an arbitrary domain string onto a known prompt family."""
    return domain if domain in DOMAINS else "general"


def get_initial_plan_prompt(domain: str = "general") -> str:
    return INITIAL_PLAN_PROMPTS[_resolve(domain)]


def get_success_extraction_prompt(domain: str = "general") -> str:
    return SUCCESS_EXTRACTION_PROMPTS[_resolve(domain)]


def get_failure_extraction_prompt(domain: str = "general") -> str:
    return FAILURE_EXTRACTION_PROMPTS[_resolve(domain)]


def get_merge_prompt(node_type: str, name_a: str, content_a: str,
                     name_b: str, content_b: str) -> str:
    return MERGE_PROMPT.format(
        type=node_type, TYPE=node_type.upper(),
        name_a=name_a, content_a=content_a,
        name_b=name_b, content_b=content_b,
    )
