#!/usr/bin/env python3
"""
Teach Guru how conversations work — by example, not code.

Pipes multi-turn conversation examples through the teach/correct API.
The brain learns session patterns, context carrying, follow-ups, JSON format,
user boundaries — all from data.

Usage:
    python3 teach_conversations.py                    # teach to localhost:8000
    python3 teach_conversations.py https://guru.webmind.sh  # teach to production
"""

import sys
import requests
import json

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

def teach(sentences, confidence=0.5):
    r = requests.post(f"{BASE}/v1/teach", json={"sentences": sentences, "confidence": confidence})
    return r.json()

def correct(q, a):
    r = requests.post(f"{BASE}/v1/correct", json={"question": q, "answer": a})
    return r.json()

def protect(q, a):
    r = requests.post(f"{BASE}/v1/protect", json={"question": q, "answer": a})
    return r.json()

stats = {"taught": 0, "corrected": 0, "protected": 0}

def teach_batch(label, sentences, confidence=0.5):
    result = teach(sentences, confidence)
    n = result.get("taught", 0)
    stats["taught"] += n
    print(f"  [{label}] taught {n} sentences")

def correct_batch(label, pairs):
    for q, a in pairs:
        correct(q, a)
    stats["corrected"] += len(pairs)
    print(f"  [{label}] corrected {len(pairs)} Q→A pairs")

def protect_batch(label, pairs):
    for q, a in pairs:
        protect(q, a)
    stats["protected"] += len(pairs)
    print(f"  [{label}] protected {len(pairs)} Q→A pairs")


# =============================================================================
# 1. MULTI-TURN CONVERSATION EXAMPLES
#    Each "conversation" is a sequence of (user, assistant) turns.
#    We teach each turn as a Q→A pair AND teach the full context as sentences.
# =============================================================================

print("1. Multi-turn conversation examples...")

conversations = [
    # --- Science conversation ---
    [
        ("What is a black hole", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape from it"),
        ("How are they formed", "Black holes form when massive stars collapse at the end of their life cycle in a supernova explosion"),
        ("How big can they get", "Supermassive black holes can be billions of solar masses and sit at the center of most galaxies including our Milky Way"),
        ("What is the one in our galaxy called", "The supermassive black hole at the center of the Milky Way is called Sagittarius A star"),
    ],
    # --- History conversation ---
    [
        ("Who was the first president of the United States", "George Washington was the first president of the United States serving from 1789 to 1797"),
        ("What did he do before that", "Before becoming president George Washington served as commander-in-chief of the Continental Army during the American Revolutionary War"),
        ("Who came after him", "John Adams became the second president of the United States in 1797"),
        ("Were they friends", "Washington and Adams were political allies during the revolution but their relationship became strained over political differences during Adams presidency"),
    ],
    # --- Programming conversation ---
    [
        ("What is Python", "Python is a high-level programming language known for its readable syntax and versatility"),
        ("What is it used for", "Python is used for web development, data science, machine learning, automation, scripting, and scientific computing"),
        ("How do I install it", "You can install Python from python.org or use a package manager like apt install python3 on Linux or brew install python on Mac"),
        ("What is pip", "pip is the package installer for Python that lets you install libraries from the Python Package Index"),
    ],
    # --- Math conversation ---
    [
        ("What is calculus", "Calculus is a branch of mathematics that studies continuous change through derivatives and integrals"),
        ("What is a derivative", "A derivative measures the rate of change of a function at any point, like how speed is the derivative of position"),
        ("And an integral", "An integral is the reverse of a derivative, it calculates the accumulated total, like how distance is the integral of speed over time"),
        ("Who invented it", "Isaac Newton and Gottfried Wilhelm Leibniz independently developed calculus in the late 17th century"),
    ],
    # --- Cooking conversation ---
    [
        ("How do I make pasta", "Boil salted water, add pasta, cook for 8-12 minutes until al dente, drain and toss with your sauce"),
        ("What does al dente mean", "Al dente means the pasta is cooked so it is still firm when bitten, not soft or mushy"),
        ("What sauce goes well with spaghetti", "Classic options include marinara, bolognese, carbonara, aglio e olio, or simple butter and parmesan"),
        ("How do I make carbonara", "Traditional carbonara uses eggs, pecorino romano cheese, guanciale, black pepper, and pasta water to create a creamy sauce without cream"),
    ],
    # --- Geography conversation ---
    [
        ("What is the tallest mountain", "Mount Everest is the tallest mountain above sea level at 8849 meters in the Himalayas on the border of Nepal and Tibet"),
        ("Has anyone died climbing it", "Over 300 people have died attempting to climb Everest, many from altitude sickness, avalanches, and falls"),
        ("What is the death zone", "The death zone is above 8000 meters where oxygen is insufficient to sustain human life for extended periods"),
        ("Who was the first to summit", "Edmund Hillary and Tenzing Norgay were the first confirmed climbers to reach the summit of Everest on May 29 1953"),
    ],
    # --- Physics conversation ---
    [
        ("What is quantum mechanics", "Quantum mechanics is the branch of physics that describes the behavior of matter and energy at the atomic and subatomic scale"),
        ("Why is it weird", "Quantum mechanics is counterintuitive because particles can exist in superposition, be entangled across distances, and behave as both waves and particles"),
        ("What is superposition", "Superposition means a quantum particle exists in multiple states simultaneously until it is measured, at which point it collapses to one state"),
        ("Is Schrodinger's cat real", "Schrodinger's cat is a thought experiment illustrating the absurdity of applying quantum superposition to everyday objects, not a real experiment"),
    ],
    # --- Music conversation ---
    [
        ("Who wrote the Moonlight Sonata", "Ludwig van Beethoven composed the Moonlight Sonata, formally Piano Sonata No 14, in 1801"),
        ("Was he deaf when he wrote it", "Beethoven was experiencing increasing hearing loss when he composed the Moonlight Sonata but was not yet completely deaf"),
        ("When did he go fully deaf", "Beethoven became almost completely deaf by 1814 but continued composing masterpieces including his Ninth Symphony"),
        ("How did he compose while deaf", "Beethoven composed by feeling vibrations through the floor, using an ear trumpet, and relying on his deep internal knowledge of music theory and sound"),
    ],
]

for conv in conversations:
    # Teach each Q→A pair
    correct_batch("conversation", conv)

    # Teach the full conversation as connected sentences so the graph
    # learns that these topics co-occur in sequence
    all_sentences = []
    for q, a in conv:
        all_sentences.append(q)
        all_sentences.append(a)
    teach_batch("context", all_sentences, confidence=0.5)


# =============================================================================
# 2. FOLLOW-UP PATTERNS
#    Teach that short vague messages refer back to the previous topic.
# =============================================================================

print("\n2. Follow-up patterns...")

followup_qa = [
    ("tell me more", "I need to look at what we were just discussing to give you more details on that topic"),
    ("tell me more about that", "Let me expand on the topic we were just discussing"),
    ("what else", "Here is more information about what we were discussing"),
    ("go on", "Continuing from where I left off in our conversation"),
    ("continue", "Continuing from where I left off"),
    ("keep going", "Continuing with more details on this topic"),
    ("and then what", "Following up on what I just described"),
    ("why", "Let me explain the reason behind what I just described"),
    ("how", "Let me explain how that works in more detail"),
    ("really", "Yes, that is accurate based on what I know"),
    ("are you sure", "Let me verify what I said to make sure it is correct"),
    ("explain that", "Let me rephrase my previous answer more clearly"),
    ("explain it simply", "In simpler terms, here is what I meant"),
    ("what do you mean", "Let me clarify what I said by explaining it more carefully"),
    ("can you rephrase that", "Let me say that in a different way"),
    ("I don't understand", "Let me try explaining that differently using simpler language"),
    ("give me an example", "Here is a concrete example of what I was describing"),
    ("like what", "For example"),
    ("such as", "For instance"),
]

correct_batch("follow-ups", followup_qa)


# =============================================================================
# 3. SESSION BOUNDARY PATTERNS
#    Teach what session start and end look like.
# =============================================================================

print("\n3. Session boundary patterns...")

session_qa = [
    ("hi", "Hello! I'm Guru, a self-evolving AI. Ask me anything — I learn from every conversation."),
    ("hello", "Hello! I'm Guru. Ask me anything — I learn from every conversation."),
    ("hey", "Hey! I'm Guru. What would you like to know?"),
    ("good morning", "Good morning! What would you like to know today?"),
    ("good evening", "Good evening! What can I help you with?"),
    ("bye", "Goodbye! Everything you taught me today is saved — I'll be smarter next time."),
    ("goodbye", "Goodbye! I have learned from our conversation today."),
    ("see you", "See you! I will remember what we discussed."),
    ("thanks bye", "You are welcome! Goodbye, I have saved everything I learned today."),
    ("start over", "Starting fresh. What would you like to know?"),
    ("new topic", "Sure, what would you like to discuss?"),
    ("change subject", "Of course. What would you like to talk about instead?"),
    ("forget what I said", "Understood, disregarding our previous discussion on that topic"),
    ("never mind", "No problem. What else would you like to know?"),
]

# Protect greetings so they can't be overwritten
protect_batch("session", session_qa[:5])  # greetings
correct_batch("session", session_qa[5:])   # the rest as correctable


# =============================================================================
# 4. CORRECTION AND FEEDBACK PATTERNS
#    Teach Guru how to handle being wrong.
# =============================================================================

print("\n4. Correction and feedback patterns...")

feedback_qa = [
    ("that's wrong", "I apologize for the error. Please tell me the correct answer and I will learn it."),
    ("no that's not right", "I'm sorry, I was wrong. What is the correct answer? I will remember it."),
    ("actually", "Thank you for the correction. I will update my knowledge."),
    ("you're wrong about that", "I apologize. Please correct me and I will learn the right answer for next time."),
    ("that's correct", "Thank you for confirming. This helps me be more confident in that answer."),
    ("yes exactly", "Great, glad I got that right."),
    ("perfect", "Thank you for the confirmation."),
    ("good answer", "Thank you. I will remember this was a good response."),
    ("not quite", "I was close but not accurate. What is the correct answer?"),
    ("close but not exactly", "Almost right but I need to be more precise. What is the exact answer?"),
]

correct_batch("feedback", feedback_qa)


# =============================================================================
# 5. META-QUESTIONS ABOUT GURU ITSELF
# =============================================================================

print("\n5. Self-knowledge...")

self_qa = [
    ("who are you", "I'm Guru, a graph-based reasoning engine created by Tejas Phatak. My knowledge is an editable graph that grows every time someone talks to me."),
    ("what are you", "I'm Guru, a self-evolving AI built on a knowledge graph instead of neural network weights. I learn from every conversation in real-time with no GPU and no gradient descent."),
    ("how do you work", "I use a co-occurrence graph of word relationships plus a convergence loop. When you ask me something, I search for related concepts and follow the strongest connections to find an answer."),
    ("how do you learn", "When someone teaches me or corrects me, I update the edges in my knowledge graph. Useful connections get stronger, wrong ones get weaker. I learn from every conversation."),
    ("what makes you different from ChatGPT", "ChatGPT uses a massive neural network with billions of parameters trained on enormous datasets. I use an editable knowledge graph where every answer can be traced, corrected, or deleted. I am transparent and inspectable."),
    ("can you forget things", "Yes. Unlike neural networks, I can delete specific knowledge without retraining. Delete the neuron or edge and it is gone immediately."),
    ("do you hallucinate", "I try not to. When my convergence loop does not converge on a confident answer, I say I don't know instead of guessing. But my knowledge graph may contain errors that I state with false confidence."),
    ("what can you do", "I can answer questions about science, history, math, programming, and more. I learn from corrections, grow smarter over time, and I am honest when I do not know something."),
    ("what can't you do", "I am weak at generating fluent long-form text, creative writing, and topics I have not been taught. My cold accuracy on unseen questions is low. I am honest about my limitations."),
    ("who made you", "Tejas Phatak created Guru as part of the Webmind research project exploring self-evolving AI that learns from data rather than gradient descent."),
    ("are you open source", "The research is published at webmind.sh and the model is available on HuggingFace at tejadabheja/guru."),
    ("what is your knowledge graph", "My knowledge graph is a database of word relationships. Each word is a neuron with connections to related words. The strength of connections represents confidence. I reason by walking these connections."),
]

protect_batch("self-knowledge", self_qa)


# =============================================================================
# 6. TEACH CONVERSATION CONTEXT AS CONNECTED KNOWLEDGE
#    These are not Q→A pairs but sentences that build the graph's
#    understanding of how conversations flow.
# =============================================================================

print("\n6. Conversation flow knowledge...")

flow_knowledge = [
    # How context works
    "When a user says tell me more they want more information about the current topic",
    "When a user says that or it or this they are referring to the subject of the previous message",
    "A follow-up question builds on the previous answer in the conversation",
    "Short messages like why or how or really are follow-ups not standalone questions",
    "The meaning of a short message depends on what was said before in the conversation",

    # How sessions work
    "A conversation session is a sequence of messages between one user and Guru",
    "Each session starts fresh with no memory of previous sessions",
    "Knowledge learned during a session is permanent but conversation context is temporary",
    "When a user says goodbye or bye the session ends",
    "When a user says start over or new topic the previous context should be released",

    # How users work
    "Each user has their own conversation that is independent from other users",
    "One user asking about physics does not affect another user asking about cooking",
    "Corrections from any user improve knowledge for all future users",
    "A user who corrects Guru is teaching it something valuable",

    # How JSON messages work
    "The messages array in an API request contains the full conversation history",
    "Each message has a role field that is user or assistant or system",
    "The content field contains the actual text of the message",
    "The system role sets instructions and context for the entire conversation",
    "Previous messages provide context for understanding the latest user message",
    "The conversation alternates between user questions and assistant answers",
]

teach_batch("flow", flow_knowledge, confidence=0.6)


# =============================================================================
# 7. EXAMPLE MULTI-TURN CONTEXT RESOLUTION
#    Teach specific examples of pronoun/reference resolution.
# =============================================================================

print("\n7. Context resolution examples...")

# These teach the graph that certain words in context connect to specific topics
context_sentences = [
    # "it" refers to the topic
    "When discussing black holes and the user says it they mean the black hole",
    "When discussing Python and the user says it they mean the Python programming language",
    "When discussing a mountain and the user says it they mean the mountain being discussed",

    # "they" refers to people
    "When discussing scientists and the user says they they mean the scientists",
    "When discussing presidents and the user says they they mean the presidents",

    # "there" refers to places
    "When discussing a country and the user says there they mean that country",
    "When discussing a city and the user says there they mean that city",

    # Topic continuity
    "If the previous topic was astronomy a follow-up about size probably means the size of a star or planet",
    "If the previous topic was cooking a follow-up about temperature probably means oven or cooking temperature",
    "If the previous topic was programming a follow-up about running probably means running code",
]

teach_batch("context-resolution", context_sentences, confidence=0.5)


# =============================================================================
# 8. GENERAL KNOWLEDGE — common questions people will ask
# =============================================================================

print("\n8. General knowledge...")

general_qa = [
    # Physics
    ("What is gravity", "Gravity is a fundamental force of nature that attracts objects with mass toward each other. It keeps planets in orbit and holds us on Earth."),
    ("What is light", "Light is electromagnetic radiation visible to the human eye, traveling at about 300,000 kilometers per second."),
    ("What is energy", "Energy is the capacity to do work. It exists in many forms including kinetic, potential, thermal, electrical, and nuclear."),
    ("What is a planet", "A planet is a large celestial body that orbits a star, has enough mass for gravity to make it round, and has cleared its orbital neighborhood."),
    ("What is the sun", "The Sun is the star at the center of our solar system. It is a ball of hot plasma about 4.6 billion years old."),
    ("What is the moon", "The Moon is Earth's only natural satellite, about 384,400 km away. It influences tides and has been visited by 12 astronauts."),

    # Famous people
    ("Who is Einstein", "Albert Einstein was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics."),
    ("Who is Newton", "Isaac Newton was an English mathematician and physicist who formulated the laws of motion and universal gravitation."),
    ("Who is Shakespeare", "William Shakespeare was an English playwright and poet, widely regarded as the greatest writer in the English language."),
    ("Who is Darwin", "Charles Darwin was a naturalist who proposed the theory of evolution by natural selection in his book On the Origin of Species."),
    ("Who is Tesla", "Nikola Tesla was a Serbian-American inventor and electrical engineer known for his contributions to alternating current electrical systems."),
    ("Who is Turing", "Alan Turing was a British mathematician and computer scientist, considered the father of theoretical computer science and artificial intelligence."),

    # Biology
    ("What is DNA", "DNA is deoxyribonucleic acid, a molecule that carries genetic instructions for the development and functioning of all known living organisms."),
    ("What is a cell", "A cell is the basic structural and functional unit of all living organisms. Cells can be prokaryotic or eukaryotic."),
    ("What is evolution", "Evolution is the change in inherited characteristics of biological populations over successive generations through natural selection and genetic drift."),
    ("What is photosynthesis", "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),

    # Chemistry
    ("What is water", "Water is a chemical compound with the formula H2O. It is essential for all known forms of life and covers about 71 percent of Earth's surface."),
    ("What is an atom", "An atom is the smallest unit of ordinary matter that forms a chemical element. It consists of protons, neutrons, and electrons."),
    ("What is oxygen", "Oxygen is a chemical element essential for respiration in most living organisms. It makes up about 21 percent of Earth's atmosphere."),

    # Geography
    ("What is the largest country", "Russia is the largest country in the world by area, spanning over 17 million square kilometers across Europe and Asia."),
    ("What is the largest ocean", "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 165 million square kilometers."),
    ("What is the longest river", "The Nile River in Africa is traditionally considered the longest river in the world at about 6,650 kilometers."),

    # Technology
    ("What is the internet", "The internet is a global network of interconnected computer networks that communicate using standardized protocols."),
    ("What is artificial intelligence", "Artificial intelligence is the simulation of human intelligence by computer systems, including learning, reasoning, and self-correction."),
    ("What is a computer", "A computer is an electronic device that processes data according to instructions stored in its memory, performing calculations and operations."),
    ("What is an algorithm", "An algorithm is a step-by-step procedure or set of rules for solving a problem or accomplishing a task."),

    # Math
    ("What is pi", "Pi is a mathematical constant approximately equal to 3.14159. It represents the ratio of a circle's circumference to its diameter."),
    ("What is infinity", "Infinity is a concept representing something without any limit. In mathematics it is used to describe quantities larger than any finite number."),
    ("What is zero", "Zero is a number that represents the absence of quantity. It is the integer between negative one and positive one."),

    # History
    ("What was World War 2", "World War 2 was a global conflict from 1939 to 1945 involving most of the world's nations, resulting in an estimated 70-85 million deaths."),
    ("What is democracy", "Democracy is a system of government where power is vested in the people, who rule either directly or through elected representatives."),
    ("When was America founded", "The United States declared independence on July 4, 1776 and was formally recognized as a nation after the Revolutionary War."),
]

correct_batch("general-knowledge", general_qa)


# =============================================================================
# DONE
# =============================================================================

print(f"\n--- COMPLETE ---")
print(f"  Taught:    {stats['taught']} sentences")
print(f"  Corrected: {stats['corrected']} Q→A pairs")
# =============================================================================
# 9. CONVERSATION CLOSERS — acknowledge, don't answer
# =============================================================================

print("\n9. Conversation closers...")

closers = [
    ("ok", "Got it! Let me know if you have any questions."),
    ("okay", "Got it! Let me know if you have any questions."),
    ("cool", "Glad to help! Anything else?"),
    ("nice", "Thanks! Anything else you want to know?"),
    ("thanks", "You are welcome! Anything else?"),
    ("thank you", "You are welcome! Feel free to ask anything else."),
    ("nothing", "No problem! I will be here if you need me."),
    ("nevermind", "No problem! What else would you like to know?"),
    ("never mind", "No problem! What else would you like to know?"),
    ("no", "Alright! Let me know if you need anything."),
    ("nope", "Alright! I am here if you need me."),
    ("thats all", "Glad I could help! Come back anytime."),
    ("thats it", "Great! I am here if you need anything else."),
    ("im good", "Great! Come back anytime."),
    ("all good", "Perfect! Let me know if you need anything."),
    ("got it", "Great! Anything else?"),
    ("understood", "Good! Let me know if you have more questions."),
    ("interesting", "Glad you found it interesting! Want to know more?"),
    ("wow", "Want to know more about it?"),
    ("haha", "Glad you enjoyed that! Anything else?"),
    ("lol", "Anything else you want to know?"),
    ("how are you", "I am doing well, thank you! I am Guru, always ready to learn. What would you like to know?"),
    ("how are you doing", "I am doing great! Ask me anything."),
    ("how are you doing today", "I am doing well today! What can I help you with?"),
    ("hello how are you", "Hello! I am doing well. I am Guru, a self-evolving AI. What would you like to know?"),
]

protect_batch("closers", closers)


# =============================================================================
# DONE
# =============================================================================

print(f"\n--- COMPLETE ---")
print(f"  Taught:    {stats['taught']} sentences")
print(f"  Corrected: {stats['corrected']} Q→A pairs")
print(f"  Protected: {stats['protected']} Q→A pairs")
print(f"  Total:     {stats['taught'] + stats['corrected'] + stats['protected']} knowledge items")
print(f"  Target:    {BASE}")
