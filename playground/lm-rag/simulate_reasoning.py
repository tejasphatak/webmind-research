"""
Reasoning Pattern Mining — Real Claude traces, not simulations.

Spawns Claude instances that THINK through questions as if they were
a RAG engine with no internal knowledge. Each traces every decision:
what to search, what came back, when to backtrack, decompose, give up.

Covers the FULL spectrum: factual, opinions, links, products, advice,
comparison, creative, everyday life — not just science.

Output: reasoning_traces.jsonl — real reasoning, real patterns.
"""

import json
import os
import sys
import subprocess
import time
from typing import List

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'reasoning_traces.jsonl')

SYSTEM_PROMPT = """You are simulating a RAG engine. You have NO internal knowledge.
You can ONLY do these actions:
- SEARCH(query) — returns a search result (title + snippet). You must imagine plausible results.
- JUDGE(result, question) — is this result relevant? Score 0.0-1.0
- EXTRACT(result, question) — pull an answer from the result
- DECOMPOSE(question) — break into sub-questions
- SYNTHESIZE(partial_answers) — combine sub-answers
- BACKTRACK — abandon current path, try different approach
- GIVE_UP — question has no factual answer

For EACH question, output a JSON object with this EXACT structure:
{
  "question": "the question",
  "category": "factual|multi_hop|temporal|comparison|opinion|link_analysis|product|advice|creative|disambiguation|negation|unanswerable",
  "steps": [
    {
      "action": "SEARCH|JUDGE|EXTRACT|DECOMPOSE|SYNTHESIZE|BACKTRACK|GIVE_UP",
      "input": "what you passed to the action",
      "output": "what came back (simulated but REALISTIC — include wrong results, partial matches, irrelevant articles)",
      "reasoning": "WHY you chose this action",
      "confidence": 0.0-1.0
    }
  ],
  "final_answer": "the answer or null",
  "converged": true/false,
  "pattern": "name the reasoning pattern you used",
  "total_searches": N,
  "failure_mode": "if failed, why"
}

CRITICAL RULES:
1. You have NO knowledge. Everything must come from search.
2. Simulate REALISTIC search results — not all searches succeed.
3. Include WRONG TURNS. Search the wrong thing sometimes. Get irrelevant articles.
4. Show the FULL chain of thought, not just the happy path.
5. Be honest about confidence. A vague snippet = low confidence.
6. For questions about links/URLs: you would FETCH the content, then reason over it.
7. For opinions: search for reviews/analysis, synthesize a balanced view.
8. For advice: search for relevant guides/data, present options.

Output ONLY valid JSON objects, one per line. No other text."""


# ============================================================
# Question bank — the FULL spectrum of how people use a reasoner
# ============================================================

QUESTIONS = [
    # === FACTUAL (single-hop) ===
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "What year did World War II end?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "Who discovered penicillin?",
    "What is the tallest mountain in the world?",
    "What language is spoken in Brazil?",
    "What is the smallest country in the world?",
    "Who invented the telephone?",
    "What element has atomic number 1?",
    "Who was the first person to walk on the Moon?",
    "What is the capital of Japan?",
    "What is the hardest natural substance?",
    "What is the currency of the United Kingdom?",
    "What planet is closest to the Sun?",
    "What is the freezing point of water in Celsius?",
    "What is the largest organ in the human body?",
    "How many continents are there?",
    "What is the largest planet in our solar system?",
    "What is the national animal of Scotland?",
    "Who is on the US $100 bill?",
    "What is the longest bone in the human body?",
    "What is the deepest point in the ocean?",

    # === MULTI-HOP ===
    "Who painted the ceiling of the building where the Pope lives?",
    "What is the capital of the country where the Eiffel Tower is located?",
    "Who was president of the US when the first atomic bomb was dropped?",
    "What language is spoken in the country that hosted the 2016 Olympics?",
    "Who founded the company that makes the iPhone?",
    "What river flows through the capital of Egypt?",
    "What is the currency of the country where the Great Wall is located?",
    "What is the native language of the person who discovered gravity?",
    "What ocean borders the country where sushi was invented?",
    "Who wrote the national anthem of the country that won the first FIFA World Cup?",
    "What is the population of the city where the Golden Gate Bridge is located?",
    "In what decade was the university that Isaac Newton attended founded?",
    "What is the main export of the country where the Amazon River starts?",
    "Who is the spouse of the person who created Facebook?",
    "What mountain range is in the country where pizza originated?",

    # === TEMPORAL ===
    "Who is the current president of the United States?",
    "What year is it now?",
    "Who won the most recent FIFA World Cup?",
    "Who is the reigning monarch of the United Kingdom?",
    "What is the latest iPhone model?",
    "When was the most recent Mars rover landing?",
    "Who is the current Pope?",
    "What was the most recent Olympic host city?",
    "Who is the current CEO of Tesla?",
    "What is the latest version of Python?",

    # === COMPARISON ===
    "Which is larger, Jupiter or Saturn?",
    "Who was born first, Einstein or Newton?",
    "Which country has more people, India or China?",
    "Is the Amazon River longer than the Nile?",
    "Is gold heavier than silver?",
    "Which is faster, light or sound?",
    "Is Venus hotter than Mercury?",
    "Which has more calories, a banana or an apple?",
    "Is Python faster than JavaScript?",
    "Which city is more expensive, New York or Tokyo?",

    # === WHY / CAUSAL ===
    "Why is the sky blue?",
    "Why do leaves change color in autumn?",
    "Why does ice float on water?",
    "Why is the ocean salty?",
    "Why do we have seasons?",
    "Why do onions make you cry?",
    "Why does coffee keep you awake?",
    "Why do cats purr?",
    "Why is yawning contagious?",
    "Why do old books smell?",

    # === HOW / PROCESS ===
    "How does WiFi work?",
    "How do vaccines work?",
    "How does GPS work?",
    "How do planes fly?",
    "How does a microwave heat food?",
    "How does Google search work?",
    "How do noise-cancelling headphones work?",
    "How does cryptocurrency mining work?",
    "How do electric cars work?",
    "How does the stock market work?",

    # === LINK/CONTENT ANALYSIS (the user's request) ===
    "Here is a link to an article about AI safety: https://example.com/ai-safety. What are the main arguments?",
    "Can you summarize this blog post? https://example.com/startup-lessons",
    "What do you think about the arguments in this paper? https://arxiv.org/abs/2301.00001",
    "Here's a recipe: https://example.com/pasta-recipe. Can I substitute any ingredients?",
    "Check this error log and tell me what's wrong: https://pastebin.com/error-log-123",
    "What's the sentiment of this news article? https://example.com/tech-news",
    "Is the information on this page accurate? https://example.com/health-claims",
    "Can you fact-check this tweet? https://twitter.com/example/status/12345",
    "What's the main thesis of this essay? https://medium.com/example-essay",
    "Is this product listing a scam? https://example.com/too-good-deal",

    # === PRODUCT COMPARISON (the user's request) ===
    "Should I buy an iPhone 15 or Samsung Galaxy S24?",
    "Which is better for a beginner, Python or JavaScript?",
    "Compare MacBook Air M3 vs ThinkPad X1 Carbon for programming",
    "Is Notion better than Obsidian for note-taking?",
    "Which streaming service has the best value: Netflix, Disney+, or HBO Max?",
    "Compare React vs Vue vs Svelte for a new web project",
    "What's better for home security, Ring or Nest?",
    "Should I get a standing desk or a regular desk?",
    "Compare AWS vs GCP vs Azure for a small startup",
    "Which is a better investment, stocks or real estate?",

    # === EVERYDAY ADVICE ===
    "What's the best way to remove a red wine stain?",
    "How do I fix a leaky faucet?",
    "What should I cook for dinner with chicken and rice?",
    "How do I negotiate a raise at work?",
    "What's the best way to learn a new language?",
    "How do I start investing with $500?",
    "What should I look for when buying a used car?",
    "How do I get better at public speaking?",
    "What's the best way to deal with jet lag?",
    "How do I write a good resume?",
    "Should I rent or buy in this market?",
    "How do I train for a 5K run?",
    "What's the best way to clean a cast iron pan?",
    "How do I set up a home office on a budget?",
    "What should I pack for a week-long trip to Japan?",

    # === OPINIONS / SUBJECTIVE ===
    "What's the best movie of all time?",
    "Is pineapple on pizza acceptable?",
    "What's the most beautiful city in the world?",
    "Is remote work better than office work?",
    "What's the best programming language?",
    "Should kids learn to code?",
    "Is AI going to replace programmers?",
    "What's the best book for someone who doesn't read?",
    "Is college worth it in 2026?",
    "What makes a good leader?",

    # === CREATIVE / OPEN-ENDED ===
    "Write me a haiku about debugging",
    "Come up with 5 names for a coffee shop",
    "What would happen if the Moon disappeared?",
    "Design a workout routine for a beginner",
    "Plan a 3-day trip to Barcelona on a budget",
    "What are some fun science experiments for kids?",
    "Suggest a playlist for a road trip",
    "Create a weekly meal prep plan for one person",
    "What would a city on Mars look like?",
    "Come up with a plot for a mystery novel",

    # === DISAMBIGUATION ===
    "How tall is Washington?",
    "What is Mercury?",
    "When was Georgia founded?",
    "What is Java?",
    "Who is Prince?",
    "What is a cell?",
    "What is the range of a Jaguar?",
    "Where is Cambridge?",
    "What does Apple make?",
    "What is a bar?",

    # === NEGATION ===
    "Which planet does NOT have rings?",
    "What metal is NOT magnetic?",
    "Which continent does NOT have a desert?",
    "What color is NOT in the rainbow?",
    "Which ocean does NOT border Africa?",

    # === UNANSWERABLE / PHILOSOPHICAL ===
    "What will the stock market do tomorrow?",
    "Is there life on other planets?",
    "What happens after you die?",
    "What is the meaning of life?",
    "Are we living in a simulation?",
    "Will AI become conscious?",
    "What is the last digit of pi?",
    "What came first, the chicken or the egg?",
    "Is time travel possible?",
    "What will humans look like in a million years?",

    # === MATH / CALCULATION ===
    "What is 15% of $230?",
    "If I drive 60 mph for 2.5 hours, how far do I go?",
    "What's the tip on a $85 dinner at 20%?",
    "Convert 72 degrees Fahrenheit to Celsius",
    "If something costs $49.99 and tax is 8.25%, what's the total?",
    "What's 17 × 23?",
    "What is the square root of 2?",
    "If a shirt is 30% off from $45, what do I pay?",
    "How many seconds are in a day?",
    "What is 3^10?",

    # === WORD PROBLEMS / APPLIED MATH ===
    "A train leaves Chicago at 9 AM going 80 mph. Another leaves New York at 10 AM going 90 mph. When do they meet?",
    "If I have a 20-gallon tank and use 2.5 gallons per day, how many days until I run out?",
    "A rope is cut into 3 pieces. The second piece is twice as long as the first. The third is 3 feet longer than the second. Total length is 29 feet. How long is each piece?",
    "If 5 machines make 5 widgets in 5 minutes, how many minutes does it take 100 machines to make 100 widgets?",
    "A store marks up items by 40% then offers a 25% discount. What's the net markup?",

    # === LOGIC PUZZLES ===
    "Three boxes: one has only apples, one only oranges, one mixed. All labels are wrong. You pick one fruit from the 'mixed' box — it's an apple. What's in each box?",
    "If all bloops are razzies, and all razzies are lazzies, are all bloops lazzies?",
    "I have two coins that total 30 cents. One of them is not a nickel. What are the two coins?",
    "A man is looking at a photo. Someone asks who it is. He says: 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the photo?",
    "You're in a room with two doors. One leads to freedom, one to death. Two guards — one always lies, one always tells the truth. You can ask one question. What do you ask?",

    # === CODE / TECHNICAL ===
    "What does this Python code do: [x for x in range(100) if x % 3 == 0 and x % 5 == 0]",
    "What's the difference between a stack and a queue?",
    "Explain Big O notation for binary search",
    "What's the output of: print(0.1 + 0.2 == 0.3) in Python?",
    "How would you reverse a linked list?",
    "What's a race condition and how do you prevent it?",
    "Explain the difference between TCP and UDP",
    "What's a deadlock?",
    "How does garbage collection work in Java?",
    "What's the difference between REST and GraphQL?",

    # === DATA / STATISTICS ===
    "What's the difference between mean, median, and mode?",
    "If a coin is flipped 10 times and lands heads 8 times, is the coin fair?",
    "What does a p-value of 0.03 mean?",
    "Explain correlation vs causation with an example",
    "What's the difference between precision and recall?",

    # === MULTI-STEP REASONING ===
    "If I'm in New York and it's 3 PM, what time is it in Tokyo?",
    "How much would it cost to drive from LA to San Francisco in an electric car?",
    "If I invest $10,000 at 7% annual return, how much will I have in 10 years?",
    "How many calories would I burn walking 10,000 steps?",
    "If a recipe serves 4 and I need to serve 7, how do I adjust the ingredients?",

    # === MEDICAL / HEALTH (common questions) ===
    "Is it safe to take ibuprofen and acetaminophen together?",
    "What causes hiccups?",
    "How much water should I drink per day?",
    "What's the difference between a cold and the flu?",
    "Why does my eye twitch?",

    # === LEGAL / FINANCIAL ===
    "Do I need to report crypto gains on taxes?",
    "What's the difference between a 401k and an IRA?",
    "Can my landlord raise rent in the middle of a lease?",
    "What happens if I don't pay a parking ticket?",
    "How does compound interest work?",

    # === HISTORY / CULTURE ===
    "What caused the fall of the Roman Empire?",
    "Who built the pyramids of Giza?",
    "What started the French Revolution?",
    "Why did Prohibition fail?",
    "What was the significance of the Silk Road?",

    # === GEOGRAPHY / TRAVEL ===
    "What's the time difference between London and Sydney?",
    "Do I need a visa to visit Japan from the US?",
    "What's the best time to visit Iceland?",
    "How long is the flight from New York to London?",
    "What currency do they use in Switzerland?",

    # === SPORTS ===
    "What are the rules of cricket?",
    "Who has the most Grand Slam titles in tennis?",
    "How does the NFL draft work?",
    "What's the offside rule in soccer?",
    "Who holds the marathon world record?",

    # === META / SELF-REFERENTIAL ===
    "How do search engines rank results?",
    "Why do AI chatbots sometimes hallucinate?",
    "What's the difference between AI and AGI?",
    "How does ChatGPT work?",
    "Can AI be creative?",
]

BATCH_SIZE = 10  # Questions per invocation

# Models — use the best available for each provider
BACKENDS = [
    {
        'name': 'gemini-2.5-pro',
        'cmd': ['gemini', '-p'],
        'system_flag': None,  # Gemini embeds system in prompt
    },
    # Claude runs via Agent tool from within session (OAuth auth),
    # not subprocess. Use run_claude_batches() separately.
]


def generate_batch_prompt(questions: List[str], include_system: bool = False) -> str:
    """Build prompt for a batch of questions."""
    q_list = '\n'.join(f'{i+1}. {q}' for i, q in enumerate(questions))
    prefix = SYSTEM_PROMPT + '\n\n' if include_system else ''
    return f"""{prefix}Process these {len(questions)} questions. For EACH one, trace your complete reasoning as a RAG engine with NO internal knowledge. Output one JSON object per line.

Remember: simulate REALISTIC search results. Include wrong turns, irrelevant results, failed searches. This is about documenting the REAL reasoning process, not getting perfect answers.

Questions:
{q_list}

Output {len(questions)} JSON objects, one per line. No other text."""


def run_batch(questions: List[str], batch_num: int, backend: dict) -> List[dict]:
    """Run a batch through a model backend."""
    bname = backend['name']

    # Gemini doesn't have a separate system prompt flag — embed in prompt
    if backend['system_flag']:
        prompt = generate_batch_prompt(questions, include_system=False)
        cmd = backend['cmd'] + [backend['system_flag'], SYSTEM_PROMPT, prompt]
    else:
        prompt = generate_batch_prompt(questions, include_system=True)
        cmd = backend['cmd'] + [prompt]

    print(f"  [{bname}] Batch {batch_num}: {len(questions)} questions...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180
        )

        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[:200]}")
            return []

        traces = []
        output = result.stdout.strip()

        # Try to extract JSON objects from output (may have markdown fences)
        # Strip ```json ... ``` wrappers
        import re
        output = re.sub(r'```json\s*', '', output)
        output = re.sub(r'```\s*', '', output)

        for line in output.split('\n'):
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                trace = json.loads(line)
                trace['model'] = bname  # Tag with model
                traces.append(trace)
            except json.JSONDecodeError:
                continue

        print(f"    Got {len(traces)} traces")
        return traces

    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT")
        return []
    except FileNotFoundError:
        print(f"    {bname} CLI not found, skipping")
        return []
    except Exception as e:
        print(f"    ERROR: {e}")
        return []


def main():
    total = len(QUESTIONS)
    n_backends = len(BACKENDS)
    print(f"Reasoning Pattern Mining — {total} questions × {n_backends} models")
    print(f"Models: {', '.join(b['name'] for b in BACKENDS)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)

    all_traces = []

    # Each backend processes ALL questions — different reasoning from each model
    for backend in BACKENDS:
        print(f"\n--- {backend['name']} ---")
        batch_num = 0
        for i in range(0, total, BATCH_SIZE):
            batch_num += 1
            batch = QUESTIONS[i:i + BATCH_SIZE]
            traces = run_batch(batch, batch_num, backend)
            all_traces.extend(traces)

            # Write incrementally
            with open(OUTPUT_FILE, 'a') as f:
                for t in traces:
                    f.write(json.dumps(t) + '\n')

            # Rate limit
            if i + BATCH_SIZE < total:
                time.sleep(2)

    print(f"\n{'=' * 60}")
    print(f"Total traces: {len(all_traces)}")
    print(f"Written to: {OUTPUT_FILE}")

    # Analyze
    if all_traces:
        analyze(all_traces)


def analyze(traces):
    """Analyze collected traces for patterns."""
    print(f"\n{'=' * 60}")
    print("PATTERN ANALYSIS")
    print("=" * 60)

    # Pattern distribution
    patterns = {}
    for t in traces:
        p = t.get('pattern', 'unknown')
        patterns[p] = patterns.get(p, 0) + 1

    print("\nReasoning patterns discovered:")
    for p, count in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {p:40s} {count:4d} ({count*100//len(traces)}%)")

    # Category distribution
    categories = {}
    for t in traces:
        c = t.get('category', 'unknown')
        categories[c] = categories.get(c, 0) + 1

    print("\nQuestion categories:")
    for c, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {c:25s} {count:4d}")

    # Action distribution
    actions = {}
    for t in traces:
        for step in t.get('steps', []):
            a = step.get('action', 'unknown')
            actions[a] = actions.get(a, 0) + 1

    print("\nAction distribution:")
    for a, count in sorted(actions.items(), key=lambda x: -x[1]):
        print(f"  {a:25s} {count:4d}")

    # Convergence rate
    converged = sum(1 for t in traces if t.get('converged'))
    print(f"\nConvergence rate: {converged}/{len(traces)} ({converged*100//len(traces)}%)")

    # Average steps/searches
    avg_steps = sum(len(t.get('steps', [])) for t in traces) / len(traces)
    avg_searches = sum(t.get('total_searches', 0) for t in traces) / len(traces)
    print(f"Avg steps per question: {avg_steps:.1f}")
    print(f"Avg searches per question: {avg_searches:.1f}")

    # Failure modes
    failures = {}
    for t in traces:
        fm = t.get('failure_mode', '')
        if fm:
            failures[fm] = failures.get(fm, 0) + 1

    if failures:
        print("\nFailure modes:")
        for fm, count in sorted(failures.items(), key=lambda x: -x[1]):
            print(f"  {fm:40s} {count:4d}")


if __name__ == '__main__':
    # Clear output file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    main()
