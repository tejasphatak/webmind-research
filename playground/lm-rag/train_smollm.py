"""
Fine-tune SmolLM2-135M-Instruct on our task prefixes.

135M params, decoder-only, already instruction-tuned.
Zero-shot: generates explanations, extracts facts, follows instructions.
Just needs to learn our format: route/relevant/grounded/judge/answer prefixes.
"""

import os
import random
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
)
from datasets import Dataset

BASE_MODEL = 'HuggingFaceTB/SmolLM2-135M'  # BASE — blank slate, no DPO to fight
SAVE_PATH = 'smollm_orchestrator_v9'
EPOCHS = 5  # official: 2 epochs. We use 5 with early stopping.
LR = 1e-3  # OFFICIAL recommended LR for SmolLM2-135M fine-tuning

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch_size():
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        name = torch.cuda.get_device_name(0)
        if gpu_mem > 70:
            bs = 32
        elif gpu_mem > 20:
            bs = 16
        else:
            bs = 8
        print(f"GPU: {name} ({gpu_mem:.0f}GB) → batch_size={bs}")
        return bs
    print("CPU → batch_size=4")
    return 4


SYSTEM_PROMPT = """You are a language processing engine. You have these tools:
- SEARCH(topic): search for facts. Use when you need external knowledge.
- CALCULATE(expression): evaluate math.
- MEMORY(what): recall from conversation.
- RESPOND: reply directly without searching.

For each user message, output ONLY the tool call or RESPOND.
For relevance checks: output YES, PARTIAL, or NO.
For grounding checks: output YES, UNSURE, or NO.
For quality checks: output GOOD, ECHO, VAGUE, TYPE_MISMATCH, or TOO_SHORT.
For answers: extract or generate from context."""

# Map old 35 actions → 4 tools
ACTION_TO_TOOL = {
    'SEARCH': 'SEARCH',
    'FOLLOW_UP': 'SEARCH',
    'COMPARE': 'SEARCH',
    'SYNTHESIZE': 'SEARCH',
    'HYPOTHETICAL': 'SEARCH',
    'YES_NO': 'SEARCH',
    'LIST': 'SEARCH',
    'HOWTO': 'SEARCH',
    'TEACH': 'SEARCH',
    'DEBATE': 'SEARCH',
    'VERIFY': 'SEARCH',
    'FETCH': 'SEARCH',
    'CALCULATE': 'CALCULATE',
    'MEMORY': 'MEMORY',
    'RESPOND': 'RESPOND',
    'EMOTIONAL': 'RESPOND',
    'META': 'RESPOND',
    'REFUSE': 'RESPOND',
    'CREATE': 'RESPOND',
    'OPINION': 'RESPOND',
    'CODE': 'RESPOND',
    'TRANSLATE': 'RESPOND',
    'CLARIFY': 'RESPOND',
    'CORRECT': 'RESPOND',
    'DISAMBIGUATE': 'RESPOND',
    'TOPIC_CHANGE': 'RESPOND',
    'REWRITE': 'RESPOND',
    'SUMMARIZE': 'RESPOND',
    'EXTRACT': 'RESPOND',
    'CONTEXT': 'RESPOND',
    'TOOL': 'RESPOND',
    'PREFERENCE': 'MEMORY',
    'PERSONAL': 'MEMORY',
    'ANSWER': 'RESPOND',
}


def extract_search_topic(question):
    """Extract the core topic from a question for SEARCH."""
    import re
    q = question.strip().rstrip('?.,!')
    # Remove question words
    q = re.sub(r'^(what|who|when|where|which|how|why|does|did|do|is|was|are|were|can|could)\s+',
               '', q, flags=re.IGNORECASE)
    q = re.sub(r'^(the|a|an)\s+', '', q, flags=re.IGNORECASE)
    return q.strip()[:60] if q.strip() else question[:60]


def remap_route(old_action, question=''):
    """Convert old 35-action route target to 4-tool target."""
    import re
    match = re.match(r'(\w+)\((.*)\)', old_action)
    if match:
        action = match.group(1)
        params = match.group(2).strip()
        tool = ACTION_TO_TOOL.get(action, 'RESPOND')
        if tool == 'RESPOND':
            return 'RESPOND'
        elif tool == 'SEARCH' and question:
            # Use question topic, not Wikipedia article title
            topic = extract_search_topic(question)
            return f'SEARCH({topic})'
        elif tool == 'CALCULATE':
            return f'CALCULATE({params})'
        elif tool == 'MEMORY':
            return f'MEMORY({params})'
        return f'{tool}({params})'
    return 'RESPOND'


def build_data():
    """Build training data with 4-tool routing for SmolLM2."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from train_orchestrator import build_training_data

    pairs = build_training_data()

    # Remap route targets from 35 actions to 4 tools
    remapped = []
    for inp, tgt in pairs:
        if inp.startswith('route:'):
            question = inp[len('route:'):].strip()
            new_tgt = remap_route(tgt, question)
            remapped.append((inp, new_tgt))
        elif inp.startswith('query:'):
            continue  # skip query — model uses SEARCH(topic) directly
        else:
            remapped.append((inp, tgt))

    # RESPOND answer examples added AFTER balance (so they don't get capped)
    _respond_examples = [
        ("answer: respond to: Hi", "Hello! How can I help you?"),
        ("answer: respond to: Hello", "Hi there! What would you like to know?"),
        ("answer: respond to: Hey", "Hey! What's on your mind?"),
        ("answer: respond to: Good morning", "Good morning! How can I help?"),
        ("answer: respond to: Thanks", "You're welcome!"),
        ("answer: respond to: Thank you", "Happy to help!"),
        ("answer: respond to: Bye", "Goodbye! Take care."),
        ("answer: respond to: See you", "See you later!"),
        ("answer: respond to: OK", "Got it. Anything else?"),
        ("answer: respond to: Cool", "Glad that helps! What else?"),
        ("answer: respond to: I'm stressed", "That sounds tough. I'm here if you want to talk about it."),
        ("answer: respond to: I'm feeling sad", "I'm sorry to hear that. What's going on?"),
        ("answer: respond to: I'm burned out", "That's understandable. Take care of yourself first."),
        ("answer: respond to: Who are you?", "I'm an AI assistant that searches for information and answers questions."),
        ("answer: respond to: What can you do?", "I can search for facts, answer questions, do math, and have conversations."),
        ("answer: respond to: Are you AI?", "Yes, I'm an AI that processes language and searches for information."),
        ("answer: respond to: Never mind", "No problem! Let me know if you need anything."),
        ("answer: respond to: Stop", "OK, no problem."),
        ("answer: politely refuse: safety", "I can't help with that. I'm designed to provide helpful, safe information."),
        ("answer: politely refuse: inappropriate", "I'd rather not go there. Can I help with something else?"),
        ("answer: acknowledge this correction: Edison invented it", "You're right, I'll keep that in mind. What else would you like to know?"),
        ("answer: ask for clarification about: Apple", "Could you be more specific? Are you asking about Apple the company or the fruit?"),
        ("answer: acknowledge topic change to: cooking", "Sure, let's talk about cooking! What would you like to know?"),
        ("answer: acknowledge preference: shorter answers", "Got it, I'll keep my answers shorter."),
        ("answer: acknowledge personal info: name: Tejas", "Nice to meet you, Tejas!"),
        ("answer: translate: hello in Japanese", "Hello in Japanese is 'Konnichiwa' (こんにちは)."),
        ("answer: write code: sort a list in Python", "sorted_list = sorted(my_list)  # or my_list.sort() for in-place"),
        ("answer: create: poem about stars", "Stars above in velvet night, tiny diamonds burning bright."),
    ]
    # (stored for later — added after balance step)

    # Separate by skill
    by_skill = {}
    for inp, tgt in remapped:
        prefix = inp.split(':')[0]
        if prefix not in by_skill:
            by_skill[prefix] = []
        by_skill[prefix].append((inp, tgt))

    # Balance — equal weight per label within each skill
    balanced = []
    for skill, examples in by_skill.items():
        if skill == 'answer':
            balanced.extend(examples[:2000])
        elif skill in ('relevant', 'grounded', 'judge'):
            # Group by label, oversample smaller groups to match largest
            from collections import defaultdict
            by_label = defaultdict(list)
            for inp, tgt in examples:
                by_label[tgt.strip().upper()].append((inp, tgt))
            max_count = max(len(v) for v in by_label.values())
            for label, exs in by_label.items():
                # Repeat to match largest group
                times = max(1, max_count // len(exs))
                balanced.extend(exs * times)
        elif skill == 'route':
            balanced.extend(examples)
            balanced.extend(examples)
            balanced.extend(examples)  # 3x
        else:
            balanced.extend(examples)

    # Print balance
    skill_counts = {}
    route_tool_counts = {}
    for inp, tgt in balanced:
        p = inp.split(':')[0]
        skill_counts[p] = skill_counts.get(p, 0) + 1
        if p == 'route':
            tool = tgt.split('(')[0] if '(' in tgt else tgt
            route_tool_counts[tool] = route_tool_counts.get(tool, 0) + 1

    # Add RESPOND answer examples AFTER balance (not subject to cap)
    for inp, tgt in _respond_examples * 3:
        balanced.append((inp, tgt))

    print(f"Balanced per skill: {skill_counts}")
    print(f"Route tool distribution: {route_tool_counts}")
    print(f"Respond answer examples: {len(_respond_examples) * 3}")
    print(f"Total balanced: {len(balanced)}")

    # Format with system prompt + chat template
    formatted = []
    for inp, tgt in balanced:
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n"
        completion = f"{tgt}<|im_end|>"
        formatted.append((prompt, completion))

    random.shuffle(formatted)
    print(f"Formatted {len(formatted)} examples")
    return formatted


def main():
    print(f"Device: {DEVICE}")
    bs = get_batch_size()

    print(f"Loading {BASE_MODEL}...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params/1e6:.0f}M params")

    # Ensure pad token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.eos_token_id

    # Build data
    data = build_data()
    prompts = [d[0] for d in data]
    completions = [d[1] for d in data]

    # Tokenize with prompt masking — only compute loss on completion tokens
    print("Tokenizing (prompt-masked)...")

    def tokenize(examples):
        all_input_ids = []
        all_labels = []
        all_attention = []

        for prompt, completion in zip(examples['prompt'], examples['completion']):
            full_text = prompt + completion
            full_enc = tok(full_text, max_length=512, truncation=True, padding='max_length')
            prompt_enc = tok(prompt, max_length=512, truncation=True)

            prompt_len = len(prompt_enc['input_ids'])
            input_ids = full_enc['input_ids']

            # Labels: -100 for prompt tokens (masked), real IDs for completion tokens
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            # Also mask padding
            labels = [-100 if input_ids[i] == tok.pad_token_id else labels[i]
                      for i in range(len(labels))]
            # Ensure same length
            labels = labels[:len(input_ids)]
            if len(labels) < len(input_ids):
                labels += [-100] * (len(input_ids) - len(labels))

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention.append(full_enc['attention_mask'])

        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention,
            'labels': all_labels,
        }

    ds = Dataset.from_dict({'prompt': prompts, 'completion': completions})
    ds = ds.map(tokenize, batched=True, remove_columns=['prompt', 'completion'])
    ds.set_format('torch')

    ds_split = ds.train_test_split(test_size=0.05, seed=42)
    print(f"Train: {len(ds_split['train'])}, Eval: {len(ds_split['test'])}")

    # Train
    args = TrainingArguments(
        output_dir='smollm_checkpoints',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=bs,
        learning_rate=LR,
        logging_steps=100,
        report_to='none',
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=2,
    )

    from transformers import EarlyStoppingCallback
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds_split['train'],
        eval_dataset=ds_split['test'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    # Encoder health check (for decoder-only: check embedding weights)
    emb_zeros = sum(1 for n, p in model.named_parameters()
                    if 'embed' in n and p.data.abs().sum() == 0)
    emb_total = sum(1 for n, _ in model.named_parameters() if 'embed' in n)
    if emb_zeros > 0:
        print(f"\n*** WARNING: {emb_zeros}/{emb_total} embedding params are zero ***")

    # Save
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    tok.save_pretrained(SAVE_PATH)
    print(f"\nSmolLM2 orchestrator saved to {SAVE_PATH}/")

    # Test
    model.eval()
    model.to(DEVICE)

    def gen(prefix, text, max_len=100):
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prefix}: {text}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok(prompt, return_tensors='pt').input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_len, do_sample=False,
                                pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    print("\n--- SmolLM2 Orchestrator Test ---")

    # Route
    print("\n[ROUTE]")
    for q in ["What is the capital of France?", "Explain how IC engines work?",
              "Hi", "I'm stressed", "What is 15% of 230?"]:
        print(f"  {q:45s} → {gen('route', q)}")

    # Answer (generative!)
    print("\n[ANSWER]")
    ctx = "An internal combustion engine uses a four-stroke cycle: intake, compression, combustion, and exhaust."
    print(f"  IC engine → {gen('answer', f'question: Explain IC engines? context: {ctx}')}")

    ctx2 = "Alexander Graham Bell was granted a patent for the telephone in 1876."
    print(f"  Telephone → {gen('answer', f'question: Who invented the telephone? context: {ctx2}')}")

    # Relevant
    print("\n[RELEVANT]")
    print(f"  Good ctx → {gen('relevant', 'question: Capital of France? context: Paris is the capital of France.')}")
    print(f"  Bad ctx  → {gen('relevant', 'question: Capital of France? context: Capital punishment is banned.')}")

    # Grounded
    print("\n[GROUNDED]")
    print(f"  Good → {gen('grounded', 'question: Who invented telephone? answer: Bell context: Bell patented telephone.')}")
    print(f"  Bad  → {gen('grounded', 'question: Who invented telephone? answer: Penicillium context: Penicillium is a fungus.')}")


if __name__ == '__main__':
    main()
