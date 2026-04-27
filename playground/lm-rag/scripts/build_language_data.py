"""
Build all language data files: idioms, grammar, normalize, registers, templates, constructions.

Generates JSON data for ULI from curated knowledge + existing resources.
Run once to populate the data/ directory.
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def save(subdir, filename, data):
    path = os.path.join(DATA_DIR, subdir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    size = os.path.getsize(path) / 1024
    print(f"  {subdir}/{filename}: {size:.0f} KB")


# ============================================================
# IDIOMS — fixed expressions per language
# ============================================================

def build_idioms():
    print("\n=== IDIOMS ===")

    save('idioms', 'en.json', {
        # Death / endings
        "kick the bucket": {"meaning": "die", "literal": False},
        "bite the dust": {"meaning": "die or fail", "literal": False},
        "pushing up daisies": {"meaning": "dead and buried", "literal": False},
        # Good / positive
        "piece of cake": {"meaning": "very easy", "literal": False},
        "break a leg": {"meaning": "good luck", "literal": False},
        "hit the nail on the head": {"meaning": "exactly right", "literal": False},
        "on cloud nine": {"meaning": "extremely happy", "literal": False},
        "blessing in disguise": {"meaning": "good thing that seemed bad at first", "literal": False},
        # Bad / negative
        "cost an arm and a leg": {"meaning": "very expensive", "literal": False},
        "under the weather": {"meaning": "feeling ill", "literal": False},
        "the last straw": {"meaning": "final provocation", "literal": False},
        "barking up the wrong tree": {"meaning": "pursuing a mistaken approach", "literal": False},
        "adding insult to injury": {"meaning": "making a bad situation worse", "literal": False},
        # Action / effort
        "burn the midnight oil": {"meaning": "work late into the night", "literal": False},
        "bite off more than you can chew": {"meaning": "take on too much", "literal": False},
        "go the extra mile": {"meaning": "make extra effort", "literal": False},
        "pull someone's leg": {"meaning": "joke with someone", "literal": False},
        "let the cat out of the bag": {"meaning": "reveal a secret", "literal": False},
        "spill the beans": {"meaning": "reveal a secret", "literal": False},
        "beat around the bush": {"meaning": "avoid getting to the point", "literal": False},
        "cut to the chase": {"meaning": "get to the point", "literal": False},
        # Knowledge / understanding
        "see eye to eye": {"meaning": "agree", "literal": False},
        "miss the boat": {"meaning": "miss an opportunity", "literal": False},
        "once in a blue moon": {"meaning": "very rarely", "literal": False},
        "the tip of the iceberg": {"meaning": "small part of larger problem", "literal": False},
        "read between the lines": {"meaning": "understand the hidden meaning", "literal": False},
        # Time
        "at the drop of a hat": {"meaning": "instantly", "literal": False},
        "in the nick of time": {"meaning": "just barely in time", "literal": False},
        "time flies": {"meaning": "time passes quickly", "literal": False},
        # Money
        "break the bank": {"meaning": "cost too much", "literal": False},
        "penny for your thoughts": {"meaning": "what are you thinking", "literal": False},
        "raining cats and dogs": {"meaning": "raining heavily", "literal": False},
        # Locale-dependent
        "tabling a motion": {"meaning_us": "postpone", "meaning_uk": "bring forward", "locale_dependent": True},
        "first floor": {"meaning_us": "ground level", "meaning_uk": "one level up", "locale_dependent": True},
    })

    save('idioms', 'mr.json', {
        "डोक्यावर बसणे": {"meaning": "dominate someone", "literal": False},
        "तोंडावर पडणे": {"meaning": "fail publicly", "literal": False},
        "हातावर तुरी देणे": {"meaning": "deceive or trick", "literal": False},
        "कानावर हात ठेवणे": {"meaning": "refuse to listen", "literal": False},
        "डोळे उघडणे": {"meaning": "realize the truth", "literal": False},
        "पाणी पाजणे": {"meaning": "defeat someone", "literal": False},
        "नाकी नऊ आणणे": {"meaning": "exhaust or frustrate", "literal": False},
        "आकाशाला गवसणी घालणे": {"meaning": "attempt the impossible", "literal": False},
    })

    save('idioms', 'hi.json', {
        "आँखों का तारा": {"meaning": "most beloved person", "literal": False},
        "नौ दो ग्यारह होना": {"meaning": "run away", "literal": False},
        "अंगूठा दिखाना": {"meaning": "refuse bluntly", "literal": False},
        "हवा में बातें करना": {"meaning": "talk nonsense", "literal": False},
        "दाल में काला": {"meaning": "something suspicious", "literal": False},
        "आँखें चुराना": {"meaning": "avoid eye contact from guilt", "literal": False},
        "तिल का ताड़ बनाना": {"meaning": "exaggerate", "literal": False},
        "लोहे के चने चबाना": {"meaning": "do something very difficult", "literal": False},
    })


# ============================================================
# NORMALIZE — abbreviations, contractions, slang per language
# ============================================================

def build_normalize():
    print("\n=== NORMALIZE ===")

    # English already exists, but let's expand it
    save('normalize', 'en.json', {
        "abbreviations": {
            "ppl": "people", "bc": "because", "rn": "right now", "ngl": "not gonna lie",
            "tbh": "to be honest", "imo": "in my opinion", "fyi": "for your information",
            "btw": "by the way", "afaik": "as far as I know", "iirc": "if I recall correctly",
            "smh": "shaking my head", "lol": "laughing out loud", "brb": "be right back",
            "idk": "I don't know", "omg": "oh my god", "tho": "though", "thru": "through",
            "abt": "about", "w": "with", "b4": "before", "2day": "today", "2morrow": "tomorrow",
            "u": "you", "ur": "your", "r": "are", "n": "and", "pls": "please", "thx": "thanks",
            "msg": "message", "info": "information", "govt": "government", "dept": "department",
            "fr": "for real", "ong": "on god", "istg": "I swear to god", "wya": "where you at",
            "hmu": "hit me up", "lmk": "let me know", "ily": "I love you", "ty": "thank you",
            "np": "no problem", "jk": "just kidding", "ofc": "of course", "ikr": "I know right",
            "nvm": "never mind", "ttyl": "talk to you later", "bff": "best friend forever",
            "dm": "direct message", "rt": "retweet", "tl": "timeline", "irl": "in real life",
            "smth": "something", "sth": "something", "sb": "somebody", "esp": "especially",
            "aka": "also known as", "asap": "as soon as possible", "eta": "estimated time of arrival",
            "fomo": "fear of missing out", "goat": "greatest of all time", "tba": "to be announced",
        },
        "contractions": {
            "dont": "don't", "cant": "can't", "wont": "won't", "im": "I'm",
            "ive": "I've", "id": "I'd", "ill": "I'll", "isnt": "isn't",
            "arent": "aren't", "wasnt": "wasn't", "werent": "weren't",
            "doesnt": "doesn't", "didnt": "didn't", "hasnt": "hasn't",
            "havent": "haven't", "hadnt": "hadn't", "shouldnt": "shouldn't",
            "wouldnt": "wouldn't", "couldnt": "couldn't", "lets": "let's",
            "thats": "that's", "whats": "what's", "whos": "who's",
            "hes": "he's", "shes": "she's", "theyre": "they're",
            "youre": "you're", "weve": "we've", "theyd": "they'd",
            "yall": "you all", "gonna": "going to", "wanna": "want to",
            "gotta": "got to", "kinda": "kind of", "sorta": "sort of",
            "dunno": "don't know", "lemme": "let me", "gimme": "give me",
        },
        "substitutions": {}
    })

    save('normalize', 'hi.json', {
        "abbreviations": {
            "acha": "अच्छा", "nahi": "नहीं", "kya": "क्या", "hai": "है",
            "haan": "हाँ", "theek": "ठीक", "sahi": "सही",
        },
        "contractions": {},
        "substitutions": {}
    })

    save('normalize', 'mr.json', {
        "abbreviations": {
            "nay": "नाही", "ho": "होय", "kay": "काय", "kasa": "कसे",
            "bare": "बरे", "chala": "चला",
        },
        "contractions": {},
        "substitutions": {}
    })

    save('normalize', 'es.json', {
        "abbreviations": {
            "tb": "también", "xq": "porque", "q": "que", "d": "de",
            "x": "por", "tmb": "también", "pq": "porque",
        },
        "contractions": {},
        "substitutions": {}
    })

    save('normalize', 'fr.json', {
        "abbreviations": {
            "slt": "salut", "pk": "pourquoi", "pcq": "parce que",
            "bcp": "beaucoup", "tt": "tout", "tjs": "toujours",
            "stp": "s'il te plaît", "mdr": "mort de rire",
        },
        "contractions": {},
        "substitutions": {}
    })

    save('normalize', 'de.json', {
        "abbreviations": {
            "vllt": "vielleicht", "bzgl": "bezüglich", "bzw": "beziehungsweise",
            "usw": "und so weiter", "zb": "zum Beispiel", "dh": "das heißt",
        },
        "contractions": {},
        "substitutions": {}
    })


# ============================================================
# REGISTERS — slang, dialect, jargon overlays
# ============================================================

def build_registers():
    print("\n=== REGISTERS ===")

    save('registers/en', 'gen_z.json', {
        "register": "gen_z",
        "parent": "english",
        "words": {
            "slay": {"senses": ["excel", "impress"], "formal": "did excellently"},
            "bussin": {"senses": ["very good", "delicious"], "formal": "excellent"},
            "no cap": {"senses": ["truthfully", "seriously"], "formal": "honestly"},
            "bet": {"senses": ["agreement", "okay"], "formal": "agreed"},
            "rizz": {"senses": ["charisma", "charm"], "formal": "charisma"},
            "fire": {"senses": ["excellent", "amazing"], "formal": "excellent"},
            "mid": {"senses": ["mediocre", "average"], "formal": "average"},
            "sus": {"senses": ["suspicious", "questionable"], "formal": "suspicious"},
            "vibe": {"senses": ["feeling", "atmosphere"], "formal": "atmosphere"},
            "lowkey": {"senses": ["somewhat", "secretly"], "formal": "somewhat"},
            "highkey": {"senses": ["very", "obviously"], "formal": "very much"},
            "salty": {"senses": ["bitter", "upset"], "formal": "bitter"},
            "shade": {"senses": ["disrespect", "subtle insult"], "formal": "disrespect"},
            "tea": {"senses": ["gossip", "truth"], "formal": "gossip"},
            "lit": {"senses": ["exciting", "excellent"], "formal": "exciting"},
            "fam": {"senses": ["close friend", "family"], "formal": "close friend"},
            "extra": {"senses": ["over the top", "dramatic"], "formal": "excessive"},
            "flex": {"senses": ["show off", "boast"], "formal": "boast"},
            "ghosting": {"senses": ["ignoring someone"], "formal": "ignoring someone"},
            "simp": {"senses": ["someone overly attentive"], "formal": "overly attentive person"},
            "ratio": {"senses": ["getting more replies than likes"], "formal": "negative response"},
            "based": {"senses": ["true to oneself", "admirable opinion"], "formal": "authentic"},
            "w": {"senses": ["win"], "formal": "win"},
            "l": {"senses": ["loss"], "formal": "loss"},
            "yeet": {"senses": ["throw", "discard forcefully"], "formal": "throw"},
            "periodt": {"senses": ["end of discussion"], "formal": "final statement"},
            "stan": {"senses": ["obsessive fan", "strongly support"], "formal": "devoted fan"},
            "understood the assignment": {"senses": ["performed perfectly"], "formal": "executed perfectly"},
            "main character": {"senses": ["the focus", "living fully"], "formal": "protagonist"},
            "rent free": {"senses": ["constantly thinking about"], "formal": "preoccupied with"},
        },
        "constructions": [
            {"pattern": "it's giving {NOUN}", "meaning": "resembles/evokes"},
            {"pattern": "lowkey {ADJ/VERB}", "meaning": "moderate degree"},
            {"pattern": "that's so {ADJ}", "meaning": "emphasis"},
            {"pattern": "no {NOUN} was {VERB}ed", "meaning": "impressive performance"},
            {"pattern": "{NOUN} is {NOUN}", "meaning": "identity assertion"},
            {"pattern": "the way {CLAUSE}", "meaning": "emphasis on surprising element"},
        ]
    })

    save('registers/en', 'aave.json', {
        "register": "aave",
        "parent": "english",
        "description": "African American Vernacular English — systematic grammar, not slang",
        "words": {
            "finna": {"senses": ["about to", "going to"], "formal": "about to"},
            "ain't": {"senses": ["is not", "am not", "have not"], "formal": "is not"},
            "tryna": {"senses": ["trying to"], "formal": "trying to"},
            "ion": {"senses": ["I don't"], "formal": "I don't"},
            "ight": {"senses": ["alright"], "formal": "alright"},
            "fasho": {"senses": ["for sure"], "formal": "for sure"},
        },
        "grammar_features": {
            "habitual_be": "Invariant 'be' for habitual aspect: 'He be working' = he regularly works",
            "remote_past_been": "'I been knew that' = I have known that for a long time",
            "completive_done": "'I done told you' = I already told you",
            "negative_concord": "Multiple negation is standard: 'I ain't never seen nobody'",
        }
    })

    save('registers/en', 'british.json', {
        "register": "british",
        "parent": "english",
        "words": {
            "brilliant": {"senses": ["great", "excellent"], "us_equiv": "awesome"},
            "rubbish": {"senses": ["garbage", "nonsense"], "us_equiv": "trash/garbage"},
            "queue": {"senses": ["line"], "us_equiv": "line"},
            "flat": {"senses": ["apartment"], "us_equiv": "apartment"},
            "boot": {"senses": ["car trunk"], "us_equiv": "trunk"},
            "bonnet": {"senses": ["car hood"], "us_equiv": "hood"},
            "lorry": {"senses": ["truck"], "us_equiv": "truck"},
            "biscuit": {"senses": ["cookie"], "us_equiv": "cookie"},
            "chips": {"senses": ["french fries"], "us_equiv": "french fries"},
            "crisps": {"senses": ["potato chips"], "us_equiv": "potato chips"},
            "petrol": {"senses": ["gasoline"], "us_equiv": "gas"},
            "lift": {"senses": ["elevator"], "us_equiv": "elevator"},
            "mate": {"senses": ["friend"], "us_equiv": "buddy"},
            "cheers": {"senses": ["thanks", "goodbye"], "us_equiv": "thanks"},
            "innit": {"senses": ["isn't it", "right"], "us_equiv": "right?"},
            "dodgy": {"senses": ["suspicious", "unreliable"], "us_equiv": "sketchy"},
            "knackered": {"senses": ["exhausted"], "us_equiv": "exhausted"},
            "gutted": {"senses": ["very disappointed"], "us_equiv": "devastated"},
            "chuffed": {"senses": ["very pleased"], "us_equiv": "thrilled"},
        }
    })

    save('registers/en', 'tech.json', {
        "register": "tech",
        "parent": "english",
        "words": {
            "deploy": {"senses": ["release software to production"]},
            "refactor": {"senses": ["restructure code without changing behavior"]},
            "debug": {"senses": ["find and fix errors in code"]},
            "sprint": {"senses": ["short development cycle, usually 2 weeks"]},
            "standup": {"senses": ["brief daily team meeting"]},
            "MVP": {"senses": ["minimum viable product"]},
            "API": {"senses": ["application programming interface"]},
            "PR": {"senses": ["pull request — code review submission"]},
            "LGTM": {"senses": ["looks good to me — approval"]},
            "ship": {"senses": ["release to users"]},
            "bikeshed": {"senses": ["argue about trivial details"]},
            "yak shaving": {"senses": ["doing prerequisite tasks before the actual task"]},
            "rubber duck": {"senses": ["explain problem aloud to find solution"]},
            "tech debt": {"senses": ["accumulated shortcuts that need fixing"]},
            "scope creep": {"senses": ["uncontrolled expansion of project requirements"]},
        }
    })

    save('registers/en', 'medical.json', {
        "register": "medical",
        "parent": "english",
        "words": {
            "prn": {"senses": ["as needed"]},
            "bid": {"senses": ["twice daily"]},
            "tid": {"senses": ["three times daily"]},
            "stat": {"senses": ["immediately"]},
            "dx": {"senses": ["diagnosis"]},
            "tx": {"senses": ["treatment"]},
            "hx": {"senses": ["history"]},
            "rx": {"senses": ["prescription"]},
            "sx": {"senses": ["symptoms"]},
            "pt": {"senses": ["patient"]},
            "BP": {"senses": ["blood pressure"]},
            "HR": {"senses": ["heart rate"]},
        }
    })

    save('registers/en', 'legal.json', {
        "register": "legal",
        "parent": "english",
        "words": {
            "plaintiff": {"senses": ["person who brings a case"]},
            "defendant": {"senses": ["person being accused"]},
            "tort": {"senses": ["wrongful act leading to civil liability"]},
            "statute": {"senses": ["written law"]},
            "jurisdiction": {"senses": ["authority of a court"]},
            "habeas corpus": {"senses": ["right to appear before judge"]},
            "pro bono": {"senses": ["free legal work"]},
            "prima facie": {"senses": ["at first sight, on its face"]},
            "subpoena": {"senses": ["legal order to appear or produce documents"]},
            "affidavit": {"senses": ["sworn written statement"]},
        }
    })


# ============================================================
# GRAMMAR — basic rules per language
# ============================================================

def build_grammar():
    print("\n=== GRAMMAR ===")

    save('grammar', 'en.json', {
        "language": "english",
        "word_order": "SVO",
        "dependency_rules": [
            {"rel": "nsubj", "head": "verb", "dep": "noun", "position": "before_head"},
            {"rel": "dobj", "head": "verb", "dep": "noun", "position": "after_head"},
            {"rel": "amod", "head": "noun", "dep": "adj", "position": "before_head"},
            {"rel": "advmod", "head": "verb", "dep": "adv", "position": "flexible"},
        ],
        "question_formation": {
            "yes_no": "AUX SUBJ VERB OBJ",
            "wh": "WH AUX SUBJ VERB"
        },
        "agreement": [
            {"between": ["subject", "verb"], "feature": "number"},
            {"between": ["determiner", "noun"], "feature": "number"},
        ],
        "negation": {"marker": "not", "position": "after_aux"},
        "tense_markers": {
            "past": "-ed (regular), irregular forms",
            "future": "will + base",
            "progressive": "be + -ing",
            "perfect": "have + -ed/-en",
        }
    })

    save('grammar', 'mr.json', {
        "language": "marathi",
        "word_order": "SOV",
        "script": "devanagari",
        "postpositions": True,
        "gender": ["masculine", "feminine", "neuter"],
        "dependency_rules": [
            {"rel": "nsubj", "head": "verb", "dep": "noun", "position": "before_head"},
            {"rel": "dobj", "head": "verb", "dep": "noun", "position": "before_head"},
            {"rel": "amod", "head": "noun", "dep": "adj", "position": "before_head"},
        ],
        "question_formation": {
            "yes_no": "SUBJ OBJ VERB + काय/का",
            "wh": "WH(कोण/काय/कुठे/केव्हा) SUBJ OBJ VERB"
        },
        "verb_agreement": {"with": "subject", "features": ["number", "gender", "person"]},
        "case_system": "vibhakti (8 cases)",
    })

    save('grammar', 'hi.json', {
        "language": "hindi",
        "word_order": "SOV",
        "script": "devanagari",
        "postpositions": True,
        "gender": ["masculine", "feminine"],
        "dependency_rules": [
            {"rel": "nsubj", "head": "verb", "dep": "noun", "position": "before_head"},
            {"rel": "dobj", "head": "verb", "dep": "noun", "position": "before_head"},
        ],
        "question_formation": {
            "yes_no": "SUBJ OBJ VERB + क्या",
            "wh": "WH(कौन/क्या/कहाँ/कब) SUBJ OBJ VERB"
        },
        "verb_agreement": {"with": "subject", "features": ["number", "gender"]},
    })

    save('grammar', 'es.json', {
        "language": "spanish", "word_order": "SVO",
        "gender": ["masculine", "feminine"],
        "verb_agreement": {"with": "subject", "features": ["number", "person"]},
        "question_formation": {"yes_no": "¿VERB SUBJ OBJ?", "wh": "¿WH VERB SUBJ?"},
    })

    save('grammar', 'fr.json', {
        "language": "french", "word_order": "SVO",
        "gender": ["masculine", "feminine"],
        "adjective_position": "after_noun (usually)",
        "negation": {"markers": ["ne...pas"], "position": "around_verb"},
    })

    save('grammar', 'de.json', {
        "language": "german", "word_order": "SVO (main), SOV (subordinate)",
        "gender": ["masculine", "feminine", "neuter"],
        "case_system": "4 cases (nominative, accusative, dative, genitive)",
        "verb_position": "second (V2) in main clause, final in subordinate",
    })

    save('grammar', 'ja.json', {
        "language": "japanese", "word_order": "SOV",
        "scripts": ["hiragana", "katakana", "kanji"],
        "particles": True,
        "honorific_system": "keigo (sonkeigo, kenjougo, teineigo)",
    })

    save('grammar', 'zh.json', {
        "language": "chinese", "word_order": "SVO",
        "tones": True,
        "classifiers": True,
        "no_inflection": True,
    })

    save('grammar', 'ar.json', {
        "language": "arabic", "word_order": "VSO (classical), SVO (modern)",
        "script": "arabic (RTL)",
        "root_system": "triliteral roots",
        "gender": ["masculine", "feminine"],
        "case_system": "3 cases",
    })

    save('grammar', 'ko.json', {
        "language": "korean", "word_order": "SOV",
        "script": "hangul",
        "particles": True,
        "honorific_system": "7 speech levels",
    })

    save('grammar', 'ru.json', {
        "language": "russian", "word_order": "SVO (flexible)",
        "script": "cyrillic",
        "gender": ["masculine", "feminine", "neuter"],
        "case_system": "6 cases",
    })


# ============================================================
# TEMPLATES — discourse structure patterns
# ============================================================

def build_templates():
    print("\n=== TEMPLATES ===")

    save('templates', 'discourse.json', {
        "email": {
            "structure": ["greeting", "context?", "body", "closing", "signature"],
            "greeting_patterns": ["Dear {name},", "Hi {name},", "Hello {name},", "Hey {name},"],
            "closing_patterns": ["Best regards,", "Thanks,", "Sincerely,", "Cheers,", "Best,"],
        },
        "essay": {
            "structure": ["introduction", "body_paragraph+", "conclusion"],
            "introduction": {"contains": ["hook", "context", "thesis_statement"]},
            "body_paragraph": {"contains": ["topic_sentence", "evidence+", "analysis", "transition?"]},
            "conclusion": {"contains": ["restatement", "synthesis", "final_thought"]},
        },
        "research_paper": {
            "structure": ["abstract", "introduction", "related_work?", "method", "results", "discussion", "conclusion", "references"],
            "abstract": {"max_words": 300, "contains": ["problem", "approach", "key_result"]},
        },
        "chat": {
            "structure": ["short_turn+"],
            "features": ["contractions", "emoji", "abbreviations", "informal"],
            "max_sentence_length": 15,
        },
        "poem_haiku": {"lines": 3, "syllables": [5, 7, 5]},
        "poem_sonnet": {"lines": 14, "meter": "iambic_pentameter", "rhyme": "ABAB_CDCD_EFEF_GG"},
        "poem_limerick": {"lines": 5, "rhyme": "AABBA"},
        "poem_free_verse": {"lines": "variable", "meter": "none", "rhyme": "none"},
        "legal_contract": {
            "structure": ["parties", "recitals?", "definitions", "terms+", "conditions+", "signatures"],
        },
        "math_proof": {
            "structure": ["given", "to_prove", "proof_steps+", "qed"],
        },
        "code_review": {
            "structure": ["summary", "issues+", "suggestions?", "verdict"],
        },
        "news_article": {
            "structure": ["headline", "lead", "body+", "conclusion?"],
            "lead": {"contains": ["who", "what", "when", "where"]},
        },
        "recipe": {
            "structure": ["title", "description?", "ingredients", "steps+", "notes?"],
        },
        "cover_letter": {
            "structure": ["header", "greeting", "opening", "body+", "closing", "signature"],
        },
        "factual_answer_short": {
            "template": "{answer}.",
        },
        "factual_answer_sentence": {
            "template": "The {target} is {answer}.",
            "alternatives": ["{answer} is the {target} of {context}.", "Based on available information, {answer}."],
        },
        "factual_answer_paragraph": {
            "template": "{context_sentence} {answer_sentence} {elaboration_sentence}",
        },
    })


# ============================================================
# CONSTRUCTIONS — universal cross-language patterns
# ============================================================

def build_constructions():
    print("\n=== CONSTRUCTIONS ===")

    save('constructions', 'universal.json', {
        "constructions": [
            {"name": "comparative_correlative", "pattern": "the {COMP} the {COMP}", "meaning": "proportional_relationship", "example_en": "the bigger the better"},
            {"name": "caused_motion", "pattern": "{AGENT} {VERB} {OBJ} {DIR}", "meaning": "agent_causes_object_to_move", "example_en": "She pushed the box off the table"},
            {"name": "ditransitive", "pattern": "{AGENT} {VERB} {RECIP} {THEME}", "meaning": "agent_transfers_theme_to_recipient", "example_en": "She gave him a book"},
            {"name": "resultative", "pattern": "{SUBJ} {VERB} {OBJ} {RESULT}", "meaning": "action_causes_state_change", "example_en": "She painted the wall red"},
            {"name": "existential", "pattern": "there {BE} {NP} {LOC}", "meaning": "existence_assertion", "example_en": "There is a book on the table"},
            {"name": "passive", "pattern": "{PATIENT} {BE} {VERB_PP} (by {AGENT})", "meaning": "patient_focused", "example_en": "The cake was eaten by the children"},
            {"name": "imperative", "pattern": "{VERB} {OBJ}!", "meaning": "command", "example_en": "Open the door!"},
            {"name": "conditional", "pattern": "if {CLAUSE}, {CLAUSE}", "meaning": "condition_consequence", "example_en": "If it rains, we stay inside"},
            {"name": "rhetorical_question", "pattern": "isn't {CLAIM}?", "meaning": "assertion_as_question", "example_en": "Isn't that obvious?"},
            {"name": "serial_verb", "pattern": "{SUBJ} {V1} {V2} {OBJ}", "meaning": "sequential_actions", "example_en": "Go get the book"},
            {"name": "cleft", "pattern": "it is {FOCUS} that {CLAUSE}", "meaning": "emphasis", "example_en": "It was John who broke the vase"},
        ]
    })


# ============================================================
# INFERENCE RULES — for DMRSM reasoning
# ============================================================

def build_inference():
    print("\n=== INFERENCE RULES ===")

    save('', 'inference_rules.json', {
        "rules": [
            {"name": "transitivity", "if": ["A causes B", "B causes C"], "then": "A causes C"},
            {"name": "modus_ponens", "if": ["if P then Q", "P"], "then": "Q"},
            {"name": "comparison_transitive", "if": ["A > B", "B > C"], "then": "A > C"},
            {"name": "temporal_order", "if": ["A before B", "B before C"], "then": "A before C"},
            {"name": "set_membership", "if": ["X is a Y", "all Y are Z"], "then": "X is a Z"},
            {"name": "contradiction", "if": ["P", "not P"], "then": "CONTRADICTION"},
            {"name": "negation_scope", "if": ["not (A and B)"], "then": "(not A) or (not B)"},
            {"name": "superlative", "if": ["A > B for all B in set"], "then": "A is the most/largest/first in set"},
        ]
    })


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Building ULI Language Data")
    print("=" * 60)

    build_idioms()
    build_normalize()
    build_registers()
    build_grammar()
    build_templates()
    build_constructions()
    build_inference()

    print("\n" + "=" * 60)
    print("Done. All language data files populated.")

    # Summary
    for subdir in ['idioms', 'normalize', 'grammar', 'registers/en', 'templates', 'constructions']:
        path = os.path.join(DATA_DIR, subdir)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.json')]
            print(f"  {subdir}/: {len(files)} files")


if __name__ == '__main__':
    main()
