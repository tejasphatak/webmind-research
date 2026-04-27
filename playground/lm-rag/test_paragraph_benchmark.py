"""
Paragraph-level similarity benchmark.
80 question-passage pairs: 40 relevant, 40 irrelevant.
Includes heavy paraphrases, multi-sentence passages, and tricky negatives.

Tests both structural similarity (ULI) and MiniLM for comparison.
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# BENCHMARK DATA — 80 pairs
# ============================================================

PARAGRAPH_PAIRS = [
    # === RELEVANT PAIRS (40) ===
    # --- Direct vocabulary overlap ---
    ("What is the capital of France?",
     "Paris is the capital of France. It is located along the Seine River and is known for the Eiffel Tower.",
     True),

    ("Who painted the Mona Lisa?",
     "Leonardo da Vinci painted the Mona Lisa in the early 16th century. The painting now hangs in the Louvre Museum in Paris.",
     True),

    ("When was the first moon landing?",
     "The first moon landing occurred on July 20, 1969. Neil Armstrong and Buzz Aldrin walked on the lunar surface as part of the Apollo 11 mission.",
     True),

    ("What is the largest planet in our solar system?",
     "Jupiter is the largest planet in the solar system. It has a mass more than twice that of all other planets combined and is known for its Great Red Spot.",
     True),

    ("Who wrote Romeo and Juliet?",
     "William Shakespeare wrote Romeo and Juliet around 1594-1596. The play is one of the most famous love stories ever written.",
     True),

    # --- Moderate paraphrase ---
    ("What causes earthquakes?",
     "Tectonic plates beneath the Earth's surface are constantly moving. When these massive slabs of rock collide, separate, or slide past each other, the resulting energy release produces seismic waves felt as ground shaking.",
     True),

    ("How does the internet work?",
     "Data is broken into small packets and transmitted across a global network of interconnected computers. Routers direct these packets through various paths, and protocols like TCP/IP ensure they arrive correctly at their destination.",
     True),

    ("Why do leaves change color in autumn?",
     "As days grow shorter, trees reduce chlorophyll production. The green pigment fades, revealing yellow and orange carotenoids that were masked during summer. Red anthocyanins are produced in response to trapped sugars.",
     True),

    ("What is DNA?",
     "Deoxyribonucleic acid is a molecule that carries the genetic instructions used in the growth, development, and functioning of all known living organisms. It consists of two strands forming a double helix structure.",
     True),

    ("How do vaccines work?",
     "Immunizations introduce a weakened or inactive form of a pathogen to the immune system. The body learns to recognize and fight the invader, creating memory cells that provide future protection against the actual disease.",
     True),

    # --- Heavy paraphrase (completely different vocabulary) ---
    ("How do planes fly?",
     "Aircraft generate lift through the aerodynamic shape of their wings. As the vehicle moves forward, air flowing over the curved upper surface travels faster than beneath, creating a pressure difference that pushes upward.",
     True),

    ("What is photosynthesis?",
     "Green organisms harness solar radiation to transform carbon dioxide and water into glucose and oxygen. This biochemical process relies on chloroplasts containing specialized pigments that absorb particular wavelengths.",
     True),

    ("Why is the sky blue?",
     "Shorter wavelengths of visible light scatter more readily when encountering atmospheric molecules. This phenomenon, described by Rayleigh, causes the overhead expanse to appear azure during daytime hours.",
     True),

    ("How fast is light?",
     "Electromagnetic radiation propagates through vacuum at approximately 299,792 kilometers per second. This fundamental constant governs the ultimate speed limit in the universe.",
     True),

    ("What causes tides?",
     "The gravitational pull of the moon and sun creates bulges in the ocean. As Earth rotates, coastlines experience the rhythmic rise and fall of water levels approximately twice daily.",
     True),

    # --- Multi-sentence, answer buried mid-passage ---
    ("Who discovered penicillin?",
     "The history of antibiotics is a fascinating chapter in medicine. In 1928, a Scottish researcher named Alexander Fleming noticed that mold had contaminated one of his petri dishes. The bacteria surrounding the mold had been killed. This accidental observation led to the development of the first widely used antibiotic.",
     True),

    ("What is the speed of sound?",
     "Sound behaves differently depending on the medium through which it travels. In dry air at room temperature, acoustic waves propagate at roughly 343 meters per second. This figure increases in denser media like water and steel.",
     True),

    ("Where is the Great Wall of China?",
     "Stretching across northern China, an ancient defensive fortification spans thousands of kilometers. Built over many centuries by various dynasties, it runs from Dandong in the east to Lop Lake in the west.",
     True),

    ("How do black holes form?",
     "When a massive star exhausts its nuclear fuel, it can no longer support itself against gravitational collapse. The core implodes, creating a region of space where the gravitational field is so intense that nothing, including electromagnetic radiation, can escape.",
     True),

    ("What causes diabetes?",
     "The pancreas produces a hormone called insulin that regulates blood sugar. Type 1 occurs when the immune system destroys insulin-producing cells. Type 2 develops when the body becomes resistant to this hormone or doesn't produce enough of it.",
     True),

    # --- Question about process, passage explains mechanism ---
    ("How does a refrigerator work?",
     "A compressor circulates refrigerant through coils. The substance absorbs heat from inside the unit as it evaporates, then releases that thermal energy outside as it condenses back to liquid. This continuous cycle maintains a cool interior temperature.",
     True),

    ("What makes rainbows appear?",
     "Sunlight entering water droplets is refracted, internally reflected, and dispersed into its component wavelengths. Each color exits at a slightly different angle, producing the familiar arc of spectral bands visible to observers when the sun is behind them.",
     True),

    ("How do earthquakes create tsunamis?",
     "When a submarine fault displaces the ocean floor vertically, an enormous volume of water is suddenly set in motion. These long-wavelength waves can travel across entire ocean basins at speeds approaching 800 km/h before building to devastating heights near coastlines.",
     True),

    # --- Historical/biographical ---
    ("When did World War II end?",
     "The global conflict concluded in 1945. Germany surrendered unconditionally on May 8, celebrated as V-E Day. Japan followed on September 2, after atomic bombs were dropped on Hiroshima and Nagasaki.",
     True),

    ("Who invented the printing press?",
     "Johannes Gutenberg developed movable type technology in Mainz, Germany, around 1440. His innovation revolutionized the production of books and made written knowledge accessible to a much broader audience.",
     True),

    # --- Scientific concepts ---
    ("What is gravity?",
     "Every object with mass exerts an attractive force on every other mass. According to Einstein's general relativity, this phenomenon results from the curvature of spacetime caused by the presence of mass and energy.",
     True),

    ("How do computers store information?",
     "Digital devices represent all data as sequences of binary digits — zeros and ones. These bits are physically encoded using electrical charges in semiconductor memory chips or magnetic patterns on spinning disk platters.",
     True),

    ("What is global warming?",
     "The average temperature of Earth's atmosphere and oceans has been rising since the Industrial Revolution. Greenhouse gases, primarily carbon dioxide from fossil fuel combustion, trap thermal radiation that would otherwise escape to space.",
     True),

    # --- Geography ---
    ("What is the longest river in the world?",
     "The Nile stretches approximately 6,650 kilometers through northeastern Africa. It flows through eleven countries before emptying into the Mediterranean Sea. Some geographers argue the Amazon may be longer when measured by certain criteria.",
     True),

    ("What is the deepest ocean?",
     "The Pacific basin contains the Mariana Trench, reaching a maximum known depth of nearly 11,000 meters. This abyss near the Mariana Islands was first explored by the bathyscaphe Trieste in 1960.",
     True),

    # --- Biology ---
    ("How do birds fly?",
     "Avian locomotion relies on lightweight hollow bones, powerful pectoral muscles, and feathered wings that generate both lift and thrust. During powered flight, the downstroke pushes air backward and downward, propelling the animal forward and upward.",
     True),

    ("What is evolution?",
     "Species change over successive generations through variation, inheritance, and natural selection. Individuals with traits better suited to their environment tend to survive and reproduce more successfully, gradually shifting the characteristics of populations.",
     True),

    # --- Technology ---
    ("How does GPS work?",
     "Satellites orbiting Earth continuously broadcast signals containing precise timing data. A receiver on the ground calculates its distance from multiple satellites by measuring signal travel time, then triangulates its exact position.",
     True),

    ("What is artificial intelligence?",
     "Machine learning systems are trained on vast datasets to recognize patterns and make predictions. Neural networks, inspired by biological brain structure, process information through layers of interconnected nodes that adjust their connections during training.",
     True),

    # --- Medicine ---
    ("How does anesthesia work?",
     "General anesthetic agents suppress neural activity throughout the central nervous system. They interrupt signal transmission between brain regions, producing unconsciousness, amnesia, and inability to feel pain during surgical procedures.",
     True),

    # --- Physics ---
    ("What is nuclear fusion?",
     "When hydrogen nuclei are forced together under extreme temperature and pressure, they combine to form helium. This thermonuclear reaction releases enormous amounts of energy and is the process that powers stars.",
     True),

    # --- Environment ---
    ("Why are coral reefs dying?",
     "Rising ocean temperatures cause zooxanthellae algae to abandon their coral hosts, a phenomenon called bleaching. Combined with acidification from absorbed CO2, pollution, and destructive fishing, these marine ecosystems face severe degradation worldwide.",
     True),

    # --- Social science ---
    ("What caused the French Revolution?",
     "Widespread inequality, crushing taxation of the common people, and food shortages created immense social tension. Enlightenment ideas about individual rights challenged the legitimacy of absolute monarchy. The storming of the Bastille in 1789 marked the beginning of radical political transformation.",
     True),

    # --- Math/Logic ---
    ("What is the Pythagorean theorem?",
     "In a right-angled triangle, the square of the hypotenuse equals the sum of the squares of the other two sides. This fundamental geometric relationship, expressed as a squared plus b squared equals c squared, has applications throughout mathematics and engineering.",
     True),

    ("What are prime numbers?",
     "Natural numbers greater than one that have no positive divisors other than one and themselves are considered indivisible. The sequence begins with 2, 3, 5, 7, 11, and continues infinitely, as Euclid proved over two millennia ago.",
     True),


    # === IRRELEVANT PAIRS (40) ===
    # --- Completely unrelated topics ---
    ("What is the capital of France?",
     "The migration patterns of Arctic terns span nearly the entire globe. These remarkable birds fly from Arctic breeding grounds to Antarctic waters and back each year.",
     False),

    ("Who painted the Mona Lisa?",
     "The Pacific Ocean covers more area than all of Earth's land surfaces combined. Its deepest point, the Challenger Deep, reaches nearly 11 kilometers below sea level.",
     False),

    ("How do planes fly?",
     "The stock market experienced significant volatility this quarter. Investors reacted to changing interest rate expectations and corporate earnings reports.",
     False),

    ("What is photosynthesis?",
     "The architectural style of Gothic cathedrals features pointed arches, ribbed vaults, and flying buttresses. These structural innovations allowed builders to create taller, lighter buildings with larger windows.",
     False),

    ("When was the first moon landing?",
     "Coffee cultivation originated in Ethiopia and spread through the Arabian Peninsula. Today, Brazil is the world's largest producer, followed by Vietnam and Colombia.",
     False),

    ("Why is the sky blue?",
     "Shakespeare's plays have been translated into every major language and are performed more often than those of any other playwright. His works explore timeless themes of love, power, and human nature.",
     False),

    ("What causes earthquakes?",
     "The history of jazz music traces back to New Orleans at the turn of the 20th century. Blues, ragtime, and African rhythmic traditions merged to create this uniquely American art form.",
     False),

    ("How does the internet work?",
     "Ancient Egyptian civilization flourished along the Nile for thousands of years. The pyramids at Giza remain among the most impressive architectural achievements in human history.",
     False),

    # --- Same topic area, but wrong subject ---
    ("Who discovered penicillin?",
     "Marie Curie conducted groundbreaking research on radioactivity. She was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes in two different sciences.",
     False),

    ("What is the longest river in the world?",
     "Mount Everest, located in the Himalayas, stands at 8,849 meters above sea level. It was first summited by Edmund Hillary and Tenzing Norgay in 1953.",
     False),

    ("What caused the French Revolution?",
     "The Industrial Revolution transformed British manufacturing in the late 18th century. Steam-powered machinery replaced hand production methods, leading to urbanization and profound social change.",
     False),

    ("How do vaccines work?",
     "Surgical techniques advanced dramatically during the 19th century. The introduction of antiseptic procedures by Joseph Lister reduced post-operative infections significantly.",
     False),

    # --- Tricky: shared words but different meaning ---
    ("What is a cell in biology?",
     "Prison cells in the United States have specific size requirements. Most state regulations mandate a minimum floor area of 60 square feet per inmate.",
     False),

    ("What is a virus in biology?",
     "Computer viruses are malicious software programs designed to spread from one system to another. They can corrupt data, steal information, or render systems inoperable.",
     False),

    ("How do batteries work?",
     "Battery farming confines poultry in small cages for egg production. Animal welfare organizations have campaigned against this practice for decades.",
     False),

    ("What is a bank?",
     "River banks provide crucial habitats for many species. Erosion along these edges can affect water quality and the stability of adjacent structures.",
     False),

    ("What is Mercury?",
     "The Freddie Mercury tribute concert was held at Wembley Stadium in 1992. Artists from around the world performed to raise awareness about AIDS.",
     False),

    # --- Related field, different question ---
    ("How do computers store information?",
     "The first electronic computers filled entire rooms and consumed enormous amounts of power. ENIAC, completed in 1945, used 17,468 vacuum tubes.",
     False),

    ("What is DNA?",
     "The Human Genome Project was completed in 2003. This international collaboration mapped all human genes and cost approximately 2.7 billion dollars.",
     False),

    ("What is gravity?",
     "The International Space Station orbits Earth at an altitude of approximately 408 kilometers. Astronauts aboard experience microgravity, which affects their bones and muscles.",
     False),

    # --- Completely different domains ---
    ("How does GPS work?",
     "Traditional bread baking requires flour, water, yeast, and salt. The dough must be kneaded to develop gluten, then allowed to rise before shaping and baking.",
     False),

    ("What is artificial intelligence?",
     "The art of Japanese flower arrangement, known as ikebana, emphasizes balance, harmony, and simplicity. Practitioners consider the vase, stems, leaves, and branches as elements of the composition.",
     False),

    ("How does anesthesia work?",
     "The Tour de France is an annual multiple-stage bicycle race primarily held in France. First organized in 1903, it is the most prestigious of cycling's Grand Tours.",
     False),

    ("What is nuclear fusion?",
     "Crop rotation is an agricultural practice that improves soil fertility. Farmers alternate between different types of crops in sequential seasons to prevent nutrient depletion.",
     False),

    ("Why are coral reefs dying?",
     "The history of the Olympic Games dates back to ancient Greece. The modern Olympics were revived in 1896 by Pierre de Coubertin in Athens.",
     False),

    # --- Misleading keyword overlap ---
    ("How fast is light?",
     "The movie 'Speed of Light' received mixed reviews from critics. Despite a strong opening weekend, box office returns declined sharply in subsequent weeks.",
     False),

    ("What is evolution?",
     "The evolution of fashion trends often cycles through decades. Styles from the 1970s and 1990s have made notable comebacks in recent years.",
     False),

    ("How do black holes form?",
     "The Black Hole of Calcutta was a small prison cell in Fort William. In 1756, the Nawab of Bengal reportedly imprisoned 146 British prisoners of war in the cramped space.",
     False),

    ("What is global warming?",
     "Warm-up exercises before physical activity help prevent injuries. Dynamic stretching and light cardio increase blood flow to muscles and improve flexibility.",
     False),

    ("What is the Pythagorean theorem?",
     "Pythagoras founded a philosophical and religious movement in ancient Greece. His followers believed in the transmigration of souls and practiced communal living.",
     False),

    # --- Similar structure, completely different content ---
    ("What causes diabetes?",
     "The Great Barrier Reef stretches over 2,300 kilometers along the Australian coast. It is the largest coral reef system in the world and is visible from space.",
     False),

    ("How does a refrigerator work?",
     "The Silk Road was a network of trade routes connecting East and West. It facilitated not only commerce but also cultural exchange for nearly 1,500 years.",
     False),

    ("What makes rainbows appear?",
     "The development of writing systems transformed human civilization. Cuneiform, one of the earliest forms, was used in Mesopotamia for administrative and literary purposes.",
     False),

    ("How do earthquakes create tsunamis?",
     "The Renaissance was a cultural movement that began in Italy during the 14th century. It marked a renewed interest in classical learning and produced masterpieces in art, literature, and science.",
     False),

    ("When did World War II end?",
     "The construction of the Panama Canal took over a decade to complete. This engineering marvel connects the Atlantic and Pacific oceans through the narrow Isthmus of Panama.",
     False),

    ("Who invented the printing press?",
     "Modern 3D printing technology creates three-dimensional objects from digital models. Materials ranging from plastics to metals can be deposited layer by layer.",
     False),

    ("How do birds fly?",
     "The Rosetta Stone was discovered in 1799 during Napoleon's Egyptian campaign. Its trilingual inscription proved key to deciphering ancient Egyptian hieroglyphics.",
     False),

    ("What are prime numbers?",
     "Amazon Prime offers subscribers free two-day shipping and access to streaming video content. The service has grown to over 200 million members worldwide.",
     False),

    ("How does GPS work?",
     "The compass was one of the Four Great Inventions of ancient China. Magnetic compasses were first used for navigation during the Song Dynasty around the 11th century.",
     False),

    ("What is the deepest ocean?",
     "Deep learning is a subset of machine learning that uses neural networks with many layers. These architectures have achieved remarkable results in image recognition and natural language processing.",
     False),
]


def run_benchmark():
    from uli.similarity import question_passage_relevance

    print("=" * 70)
    print("PARAGRAPH-LEVEL BENCHMARK — ULI Structural Similarity")
    print("=" * 70)
    print(f"Total pairs: {len(PARAGRAPH_PAIRS)}")
    print(f"  Relevant: {sum(1 for _, _, r in PARAGRAPH_PAIRS if r)}")
    print(f"  Irrelevant: {sum(1 for _, _, r in PARAGRAPH_PAIRS if not r)}")
    print()

    # Precompute ALL scores once
    precomputed = []
    t_start = time.time()
    for q, p, expected in PARAGRAPH_PAIRS:
        rel = question_passage_relevance(q, p)
        precomputed.append((rel, expected))
    compute_time = time.time() - t_start

    # Find optimal threshold using cached scores
    best_threshold = 0.0
    best_correct = 0
    for threshold_x10 in range(5, 40):
        threshold = threshold_x10 / 100.0
        correct = sum(1 for rel, exp in precomputed if (rel > threshold) == exp)
        if correct > best_correct:
            best_correct = correct
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:.2f} ({best_correct}/{len(PARAGRAPH_PAIRS)} = {best_correct*100//len(PARAGRAPH_PAIRS)}%)")
    print()

    # Use precomputed scores with optimal threshold
    threshold = best_threshold
    tp = fp = tn = fn = 0
    failures = []

    for i, (q, p, expected) in enumerate(PARAGRAPH_PAIRS):
        rel = precomputed[i][0]
        predicted = rel > threshold

        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1
            failures.append(('FN', rel, q, p[:60]))
        elif not expected and predicted:
            fp += 1
            failures.append(('FP', rel, q, p[:60]))
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Results (threshold={threshold:.2f}):")
    print(f"  Accuracy:  {tp+tn}/{total} ({accuracy:.1%})")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1:        {f1:.1%}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Time: {compute_time:.1f}s ({compute_time*1000/total:.0f}ms/pair)")
    print()

    if failures:
        print(f"FAILURES ({len(failures)}):")
        for ftype, score, q, p in failures:
            print(f"  [{ftype}] score={score:.3f} Q: {q[:50]:50s} P: {p}")
    print()

    return accuracy, f1


def run_minilm_comparison():
    """Run MiniLM side by side for comparison."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("sentence-transformers not installed, skipping MiniLM comparison")
        return None

    model = SentenceTransformer('all-MiniLM-L6-v2')

    best_threshold = 0.0
    best_correct = 0

    # Precompute embeddings
    qs = [q for q, _, _ in PARAGRAPH_PAIRS]
    ps = [p for _, p, _ in PARAGRAPH_PAIRS]
    q_embs = model.encode(qs)
    p_embs = model.encode(ps)

    sims = []
    for i in range(len(PARAGRAPH_PAIRS)):
        sim = float(np.dot(q_embs[i], p_embs[i]) /
                     (np.linalg.norm(q_embs[i]) * np.linalg.norm(p_embs[i])))
        sims.append(sim)

    for threshold_x10 in range(10, 80):
        threshold = threshold_x10 / 100.0
        correct = sum(1 for i, (_, _, exp) in enumerate(PARAGRAPH_PAIRS)
                       if (sims[i] > threshold) == exp)
        if correct > best_correct:
            best_correct = correct
            best_threshold = threshold

    print("=" * 70)
    print("MINILM COMPARISON")
    print("=" * 70)
    print(f"Optimal threshold: {best_threshold:.2f} ({best_correct}/{len(PARAGRAPH_PAIRS)} = {best_correct*100//len(PARAGRAPH_PAIRS)}%)")

    threshold = best_threshold
    tp = fp = tn = fn = 0
    failures = []

    for i, (q, p, expected) in enumerate(PARAGRAPH_PAIRS):
        predicted = sims[i] > threshold
        if expected and predicted: tp += 1
        elif expected and not predicted:
            fn += 1
            failures.append(('FN', sims[i], q, p[:60]))
        elif not expected and predicted:
            fp += 1
            failures.append(('FP', sims[i], q, p[:60]))
        else: tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Accuracy:  {tp+tn}/{total} ({accuracy:.1%})")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1:        {f1:.1%}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print()

    if failures:
        print(f"FAILURES ({len(failures)}):")
        for ftype, score, q, p in failures:
            print(f"  [{ftype}] score={score:.3f} Q: {q[:50]:50s} P: {p}")

    return accuracy


if __name__ == '__main__':
    uli_acc, uli_f1 = run_benchmark()
    print()
    minilm_acc = run_minilm_comparison()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  ULI Structural:  {uli_acc:.1%}")
    if minilm_acc:
        print(f"  MiniLM (22M):    {minilm_acc:.1%}")
        if uli_acc > minilm_acc:
            print(f"  ULI WINS by {(uli_acc - minilm_acc):.1%}")
        elif minilm_acc > uli_acc:
            print(f"  MiniLM wins by {(minilm_acc - uli_acc):.1%}")
        else:
            print(f"  TIE")
