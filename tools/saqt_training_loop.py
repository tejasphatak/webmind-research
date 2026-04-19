#!/usr/bin/env python3
"""
SAQT Training Loop — Teacher (Gemini) trains Student (SAQT)
=============================================================
1. Generate questions
2. Ask SAQT → get answer
3. Ask Gemini → get correct answer
4. If SAQT wrong → store (question, correct_answer)
5. Re-encode
6. Repeat

Runs locally. No public exposure.
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import json, time, os, random

DEVICE = "cpu"
HOME = os.environ.get("HOME", "/home/tejasphatak")
QA_PATH = os.path.join(HOME, "webmind-research/trained_model/qa_pairs.jsonl")
EMB_PATH = os.path.join(HOME, "webmind-research/trained_model/qa_embeddings.pt")


class SAQTStudent:
    """The student — retrieves from Q&A pairs."""

    def __init__(self):
        print("[student] Loading...", flush=True)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        self.pairs = []
        with open(QA_PATH) as f:
            for line in f:
                self.pairs.append(json.loads(line))
        self.embeddings = torch.load(EMB_PATH, map_location=DEVICE, weights_only=True)
        print(f"[student] {len(self.pairs)} Q&A pairs loaded", flush=True)

    def ask(self, question):
        q_emb = self.encoder.encode([question], convert_to_tensor=True,
                                   show_progress_bar=False)[0]
        sims = F.cosine_similarity(q_emb.unsqueeze(0), self.embeddings)
        top_val, top_idx = sims.topk(1)
        best = self.pairs[top_idx[0].item()]
        return best.get("answer", ""), top_val[0].item()

    def learn(self, question, answer, source="training_loop"):
        """Add a new Q&A pair and encode it."""
        new_pair = {"question": question, "answer": answer, "source": source}
        self.pairs.append(new_pair)
        # Encode and append
        new_emb = self.encoder.encode([question], convert_to_tensor=True,
                                     show_progress_bar=False)
        self.embeddings = torch.cat([self.embeddings, new_emb.cpu()], dim=0)
        # Save
        with open(QA_PATH, 'a') as f:
            f.write(json.dumps(new_pair) + "\n")
        torch.save(self.embeddings, EMB_PATH)
        return len(self.pairs)


class GeminiTeacher:
    """The teacher — generates correct answers."""

    def __init__(self):
        import google.generativeai as genai
        key = json.load(open(os.path.join(HOME, ".claude/secrets/gemini.json")))["api_key"]
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        print("[teacher] Gemini ready", flush=True)

    def answer(self, question):
        try:
            resp = self.model.generate_content(
                f"Answer this question concisely in 1-2 sentences. Be factual and direct.\n\nQuestion: {question}",
                generation_config={"max_output_tokens": 150, "temperature": 0.2})
            return resp.text.strip() if resp.text else None
        except Exception as e:
            print(f"[teacher] Error: {e}", flush=True)
            return None

    def judge(self, question, student_answer, teacher_answer):
        """Does the student's answer match the teacher's?"""
        try:
            resp = self.model.generate_content(
                f"Does Answer A correctly answer the question? Compare with Answer B. Reply ONLY 'correct' or 'wrong'.\n\nQuestion: {question}\nAnswer A (student): {student_answer[:200]}\nAnswer B (reference): {teacher_answer[:200]}",
                generation_config={"max_output_tokens": 10, "temperature": 0})
            return "correct" in resp.text.lower() if resp.text else False
        except:
            return False


def generate_test_questions():
    """Generate diverse test questions."""
    return [
        # Facts
        "What is the tallest building in the world?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical formula for water?",
        "How many continents are there?",
        "What is the largest ocean?",
        # Reasoning
        "If it's 30 degrees Celsius, is it hot or cold?",
        "If a train travels 100km in 2 hours, what is its speed?",
        "Why do boats float?",
        "What happens when you mix baking soda and vinegar?",
        # Common sense
        "Can humans breathe underwater?",
        "Is fire hot or cold?",
        "Do fish live in water or on land?",
        # Math-adjacent
        "What is the perimeter of a square with side 5?",
        "How many sides does a hexagon have?",
        # Current
        "Who is the president of the United States?",
        "What country hosts the Olympics in 2028?",
        # Programming
        "What does HTML stand for?",
        "What is a variable in programming?",
        "What is the difference between a list and a dictionary in Python?",
        # Multi-step
        "If all cats are animals, and Whiskers is a cat, is Whiskers an animal?",
    ]


def run_training_loop(n_rounds=3):
    print("=== SAQT TRAINING LOOP ===\n", flush=True)

    student = SAQTStudent()
    teacher = GeminiTeacher()

    questions = generate_test_questions()
    total_learned = 0

    for round_num in range(n_rounds):
        print(f"\n--- Round {round_num + 1}/{n_rounds} ---\n", flush=True)
        random.shuffle(questions)

        correct, wrong, learned = 0, 0, 0

        for q in questions:
            # Student answers
            s_answer, s_sim = student.ask(q)

            # Teacher answers
            t_answer = teacher.answer(q)
            if not t_answer:
                continue

            # Judge
            is_correct = teacher.judge(q, s_answer, t_answer)

            if is_correct:
                correct += 1
                print(f"  [OK] Q: {q[:50]}", flush=True)
            else:
                wrong += 1
                # LEARN: add teacher's answer to student's knowledge
                count = student.learn(q, t_answer)
                learned += 1
                total_learned += 1
                print(f"  [LEARN] Q: {q[:50]}", flush=True)
                print(f"          Student said: {s_answer[:60]}", flush=True)
                print(f"          Teacher said: {t_answer[:60]}", flush=True)
                print(f"          Now has {count} pairs", flush=True)

        pct = correct / max(correct + wrong, 1)
        print(f"\n  Round {round_num + 1}: {correct}/{correct+wrong} = {pct:.0%} "
              f"(learned {learned} new pairs)", flush=True)

        # If student got >90%, questions are too easy — would need harder ones
        if pct > 0.9:
            print(f"  Student scoring >90% — needs harder questions", flush=True)

    print(f"\n=== TRAINING COMPLETE ===", flush=True)
    print(f"  Total learned: {total_learned} new Q&A pairs", flush=True)
    print(f"  Final knowledge base: {len(student.pairs)} pairs", flush=True)


if __name__ == "__main__":
    run_training_loop(n_rounds=3)
