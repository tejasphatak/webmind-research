"""
Guru Vision — image understanding through graph reasoning.

No neural networks. Classical CV features → concept graph → reasoning.

Architecture (mirrors the language pipeline):
    Image → Feature Extractor (edges, contours, colors, textures)
        → Visual Parse (spatial grammar: parts, relationships)
        → Concept Mapper (features → graph nodes via visual ontology)
        → Graph Reasoner (traverse is_a, has, part_of edges → classify)

Like how ULI parses "The dog sat on the mat":
    POS tag → parse → (dog, sat_on, mat) → graph lookup

Vision parses an image:
    edges → contours → spatial parse → (shape_A[fur,4legs], on, shape_B[flat])
    → graph: fur+4legs → dog, flat+rectangle → mat

Usage:
    from uli.vision import VisionEngine
    engine = VisionEngine(db_path='data/vocab/wordnet.db')
    result = engine.see('photo.jpg')
    print(result.description)   # "A brown quadruped animal, possibly a dog"
    print(result.objects)       # [{'label': 'dog.n.01', 'confidence': 0.8, 'why': ['4 legs', 'fur texture', 'tail']}]
    print(result.scene)         # "indoor, bright lighting"
"""

import logging
import math
import os
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger('uli.vision')

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data', 'vocab', 'wordnet.db'
)


# ── Data structures ────────────────────────────────────────────────

@dataclass
class VisualFeature:
    """A detected visual feature."""
    name: str           # e.g., 'vertical_line', 'circle', 'brown_region'
    feature_type: str   # 'edge', 'contour', 'color', 'texture', 'spatial'
    value: float        # magnitude/count/ratio
    bbox: tuple = ()    # (x, y, w, h) if localized
    details: dict = field(default_factory=dict)


@dataclass
class VisualObject:
    """A recognized object (or candidate)."""
    label: str          # synset ID or description
    confidence: float   # 0-1, based on feature match count
    evidence: List[str] # why we think it's this ("4 legs", "fur texture")
    bbox: tuple = ()    # bounding box
    features: List[VisualFeature] = field(default_factory=list)


@dataclass
class SceneDescription:
    """Full scene analysis."""
    objects: List[VisualObject]
    scene_type: str         # 'indoor', 'outdoor', 'document', 'diagram'
    lighting: str           # 'bright', 'dark', 'mixed'
    dominant_colors: List[str]
    description: str        # natural language summary
    features: List[VisualFeature] = field(default_factory=list)
    spatial_relations: List[Tuple[str, str, str]] = field(default_factory=list)


# ── Feature Extractors (pure math, zero NN) ────────────────────────

class FeatureExtractor:
    """Classical CV feature extraction. No neural networks."""

    def extract_all(self, img: np.ndarray) -> List[VisualFeature]:
        """Extract all visual features from an image."""
        features = []
        features.extend(self._extract_edges(img))
        features.extend(self._extract_contours(img))
        features.extend(self._extract_colors(img))
        features.extend(self._extract_texture(img))
        features.extend(self._extract_scene(img))
        return features

    def _extract_edges(self, img: np.ndarray) -> List[VisualFeature]:
        """Canny edge detection → edge density, orientation distribution."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape

        features = []

        # Overall edge density (how "busy" the image is)
        edge_ratio = np.count_nonzero(edges) / (h * w)
        features.append(VisualFeature(
            name='edge_density', feature_type='edge',
            value=edge_ratio,
            details={'description': 'high' if edge_ratio > 0.15 else 'medium' if edge_ratio > 0.05 else 'low'}
        ))

        # Hough lines — detect dominant orientations
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=min(h, w)//8, maxLineGap=10)
        if lines is not None:
            horizontal = 0
            vertical = 0
            diagonal = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.atan2(y2-y1, x2-x1) * 180 / math.pi)
                if angle < 20 or angle > 160:
                    horizontal += 1
                elif 70 < angle < 110:
                    vertical += 1
                else:
                    diagonal += 1

            features.append(VisualFeature(
                name='line_count', feature_type='edge',
                value=len(lines),
                details={'horizontal': horizontal, 'vertical': vertical, 'diagonal': diagonal}
            ))

            if vertical > horizontal * 2:
                features.append(VisualFeature(
                    name='dominant_vertical', feature_type='edge', value=vertical))
            if horizontal > vertical * 2:
                features.append(VisualFeature(
                    name='dominant_horizontal', feature_type='edge', value=horizontal))

        return features

    def _extract_contours(self, img: np.ndarray) -> List[VisualFeature]:
        """Contour detection → shape analysis."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use Canny edges → dilate → find contours (separates touching objects)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = []
        h, w = gray.shape
        img_area = h * w

        # Filter significant contours (> 1% of image area)
        significant = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > img_area * 0.01:
                significant.append(cnt)

        features.append(VisualFeature(
            name='contour_count', feature_type='contour',
            value=len(significant),
            details={'total_raw': len(contours)}
        ))

        # Analyze each significant contour
        shapes = {'circle': 0, 'rectangle': 0, 'triangle': 0, 'complex': 0}
        for cnt in significant:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Circularity: 4π × area / perimeter²  (1.0 = perfect circle)
            circularity = 4 * math.pi * area / (perimeter * perimeter)

            # Approximate polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            vertices = len(approx)

            # Bounding rectangle aspect ratio
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh > 0 else 1
            extent = area / (bw * bh) if bw * bh > 0 else 0

            if circularity > 0.7:
                shapes['circle'] += 1
                shape_name = 'circle'
            elif vertices == 3:
                shapes['triangle'] += 1
                shape_name = 'triangle'
            elif vertices == 4 and extent > 0.8:
                shapes['rectangle'] += 1
                shape_name = 'rectangle'
            else:
                shapes['complex'] += 1
                shape_name = f'complex_{vertices}v'

            features.append(VisualFeature(
                name=f'shape_{shape_name}', feature_type='contour',
                value=area / img_area,
                bbox=(x, y, bw, bh),
                details={
                    'circularity': round(circularity, 3),
                    'vertices': vertices,
                    'aspect_ratio': round(aspect, 2),
                    'extent': round(extent, 2),
                    'relative_size': round(area / img_area, 4),
                }
            ))

        # Summary
        for shape_type, count in shapes.items():
            if count > 0:
                features.append(VisualFeature(
                    name=f'has_{shape_type}', feature_type='contour',
                    value=count
                ))

        return features

    def _extract_colors(self, img: np.ndarray) -> List[VisualFeature]:
        """Color analysis — dominant colors, color distribution."""
        if len(img.shape) < 3:
            return [VisualFeature(name='grayscale', feature_type='color', value=1.0)]

        features = []
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        total_pixels = img.shape[0] * img.shape[1]

        # Named color regions in HSV space
        color_ranges = {
            'red':    ((0, 50, 50), (10, 255, 255)),
            'orange': ((10, 50, 50), (25, 255, 255)),
            'yellow': ((25, 50, 50), (35, 255, 255)),
            'green':  ((35, 50, 50), (85, 255, 255)),
            'blue':   ((85, 50, 50), (130, 255, 255)),
            'purple': ((130, 50, 50), (170, 255, 255)),
            'red2':   ((170, 50, 50), (180, 255, 255)),  # red wraps around
        }

        color_counts = {}
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            count = cv2.countNonZero(mask)
            name = 'red' if color_name == 'red2' else color_name
            color_counts[name] = color_counts.get(name, 0) + count

        # Achromatic regions
        low_sat = cv2.inRange(s, 0, 50)
        achromatic = cv2.countNonZero(low_sat)
        bright = cv2.countNonZero(cv2.inRange(v, 200, 255))
        dark = cv2.countNonZero(cv2.inRange(v, 0, 50))

        if achromatic > total_pixels * 0.5:
            if bright > dark:
                color_counts['white'] = bright
            else:
                color_counts['black'] = dark
            color_counts['gray'] = achromatic - bright - dark

        # Report dominant colors (> 5% of image)
        dominant = []
        for color_name, count in sorted(color_counts.items(), key=lambda x: -x[1]):
            ratio = count / total_pixels
            if ratio > 0.05:
                dominant.append(color_name)
                features.append(VisualFeature(
                    name=f'color_{color_name}', feature_type='color',
                    value=ratio,
                    details={'percentage': round(ratio * 100, 1)}
                ))

        # Overall brightness
        mean_brightness = float(np.mean(v))
        features.append(VisualFeature(
            name='brightness', feature_type='color',
            value=mean_brightness / 255,
            details={'level': 'bright' if mean_brightness > 170 else 'dark' if mean_brightness < 85 else 'medium'}
        ))

        # Color variety (saturation)
        mean_saturation = float(np.mean(s))
        features.append(VisualFeature(
            name='saturation', feature_type='color',
            value=mean_saturation / 255,
            details={'level': 'vivid' if mean_saturation > 150 else 'muted' if mean_saturation < 50 else 'moderate'}
        ))

        return features

    def _extract_texture(self, img: np.ndarray) -> List[VisualFeature]:
        """Texture analysis using Local Binary Patterns (LBP) — pure math."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape

        features = []

        # Simple LBP: compare each pixel to its neighbors
        # Uniform patterns indicate texture regularity
        lbp = np.zeros_like(gray)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp = (lbp << 1) | (gray >= shifted).astype(np.uint8)

        # Texture uniformity — how many unique LBP patterns?
        unique_patterns = len(np.unique(lbp))
        max_patterns = 256
        uniformity = 1.0 - (unique_patterns / max_patterns)

        features.append(VisualFeature(
            name='texture_uniformity', feature_type='texture',
            value=uniformity,
            details={
                'description': 'smooth' if uniformity > 0.7 else 'textured' if uniformity < 0.3 else 'moderate',
                'unique_patterns': unique_patterns
            }
        ))

        # Variance — high variance = rough/complex texture
        variance = float(np.var(gray.astype(float)))
        normalized_var = min(variance / 3000, 1.0)
        features.append(VisualFeature(
            name='texture_variance', feature_type='texture',
            value=normalized_var,
            details={'description': 'rough' if normalized_var > 0.5 else 'smooth' if normalized_var < 0.15 else 'moderate'}
        ))

        return features

    def _extract_scene(self, img: np.ndarray) -> List[VisualFeature]:
        """Scene-level features — lighting, composition, type."""
        features = []
        h, w = img.shape[:2]

        # Aspect ratio → landscape, portrait, square
        aspect = w / h
        if aspect > 1.3:
            scene_orient = 'landscape'
        elif aspect < 0.77:
            scene_orient = 'portrait'
        else:
            scene_orient = 'square'
        features.append(VisualFeature(
            name=f'orientation_{scene_orient}', feature_type='spatial',
            value=aspect
        ))

        # Sky detection — is the top 20% of the image blue-ish?
        if len(img.shape) == 3:
            top = img[:h//5, :, :]
            hsv_top = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv_top, np.array([85, 30, 100]), np.array([130, 255, 255]))
            sky_ratio = cv2.countNonZero(blue_mask) / (top.shape[0] * top.shape[1])
            if sky_ratio > 0.3:
                features.append(VisualFeature(
                    name='has_sky', feature_type='spatial',
                    value=sky_ratio
                ))

            # Green bottom → outdoor/nature
            bottom = img[4*h//5:, :, :]
            hsv_bottom = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv_bottom, np.array([35, 30, 30]), np.array([85, 255, 255]))
            green_ratio = cv2.countNonZero(green_mask) / (bottom.shape[0] * bottom.shape[1])
            if green_ratio > 0.3:
                features.append(VisualFeature(
                    name='has_ground_vegetation', feature_type='spatial',
                    value=green_ratio
                ))

        # Text detection — lots of horizontal edges + high contrast = likely document
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)

        # Horizontal edge ratio in center region
        center = edges[h//4:3*h//4, w//4:3*w//4]
        if center.size > 0:
            edge_density = np.count_nonzero(center) / center.size
            if edge_density > 0.2:
                features.append(VisualFeature(
                    name='text_likely', feature_type='spatial',
                    value=edge_density
                ))

        return features


# ── Concept Mapper (features → graph concepts) ─────────────────────

class ConceptMapper:
    """Map visual features to concepts in the knowledge graph.

    Uses a visual ontology stored in the graph:
        dog → has_visual → fur_texture
        dog → has_visual → four_legs
        dog → has_visual → tail
        car → has_visual → wheels
        car → has_visual → rectangular_body

    When we detect features matching 'fur_texture' + 'four_legs' + 'tail',
    we query the graph for nodes that have all three → dog.
    """

    def __init__(self, db_path: str = None):
        self._db_path = db_path or _DEFAULT_DB
        self._conn = None

    def _get_conn(self):
        if self._conn is None and os.path.exists(self._db_path):
            self._conn = sqlite3.connect(self._db_path)
        return self._conn

    def features_to_concepts(self, features: List[VisualFeature]) -> List[str]:
        """Map extracted features to concept names.

        Rules (from visual ontology in graph):
        - circles → wheel, ball, sun, eye
        - rectangles → building, vehicle, screen, book
        - fur texture + quadruped shape → animal
        - sky + vegetation → outdoor scene
        - text_likely → document
        """
        concepts = []

        # Build a feature summary for graph matching
        feature_map = {}
        for f in features:
            feature_map[f.name] = f

        # Shape-based concepts
        circles = feature_map.get('has_circle')
        rectangles = feature_map.get('has_rectangle')
        triangles = feature_map.get('has_triangle')
        complex_shapes = feature_map.get('has_complex')

        # Color-based
        colors = [f.name.replace('color_', '') for f in features
                  if f.name.startswith('color_') and f.value > 0.1]

        # Texture
        texture = feature_map.get('texture_uniformity', VisualFeature('', '', 0.5))
        tex_desc = texture.details.get('description', 'moderate') if texture.details else 'moderate'

        # Scene
        has_sky = 'has_sky' in feature_map
        has_vegetation = 'has_ground_vegetation' in feature_map
        has_text = 'text_likely' in feature_map
        brightness = feature_map.get('brightness', VisualFeature('', '', 0.5))
        bright_level = brightness.details.get('level', 'medium') if brightness.details else 'medium'

        # ── Map to concepts using graph queries ──
        conn = self._get_conn()
        if conn:
            # Try graph-based visual concept matching
            graph_concepts = self._graph_match(features, conn)
            if graph_concepts:
                concepts.extend(graph_concepts)

        # ── Fallback: rule-based mapping (until graph is trained) ──
        # These should eventually all come from graph edges

        # Scene type
        if has_sky and has_vegetation:
            concepts.append('outdoor_scene')
            concepts.append('nature')
        elif has_sky:
            concepts.append('outdoor_scene')
        elif has_text:
            concepts.append('document')

        # Lighting
        if bright_level == 'bright':
            concepts.append('well_lit')
        elif bright_level == 'dark':
            concepts.append('dark_scene')

        # Object candidates from shapes
        contour_count = feature_map.get('contour_count')
        num_objects = int(contour_count.value) if contour_count else 0

        if num_objects == 0:
            concepts.append('abstract')
        elif num_objects == 1:
            concepts.append('single_object')
        elif num_objects <= 5:
            concepts.append('few_objects')
        else:
            concepts.append('busy_scene')

        if circles and circles.value >= 2:
            concepts.append('circular_objects')
        if rectangles and rectangles.value >= 1:
            concepts.append('rectangular_objects')

        return concepts

    def _graph_match(self, features: List[VisualFeature],
                     conn: sqlite3.Connection) -> List[str]:
        """Query graph for visual concepts matching detected features.

        Looks for nodes with 'has_visual' edges matching our features.
        More matches = higher confidence.
        """
        cur = conn.cursor()

        # Get all visual concept definitions from the graph
        cur.execute(
            "SELECT from_id, to_id FROM graph_edges WHERE relation='has_visual'"
        )
        visual_defs = cur.fetchall()
        if not visual_defs:
            return []

        # Build a mapping: concept → set of required visual features
        concept_features = {}
        for concept, visual_feature in visual_defs:
            if concept not in concept_features:
                concept_features[concept] = set()
            concept_features[concept].add(visual_feature.lower())

        # Match: which concepts have the most features present?
        feature_names = set(f.name.lower() for f in features)
        # Also add derived names
        for f in features:
            if f.details:
                desc = f.details.get('description', '')
                if desc:
                    feature_names.add(desc.lower())

        matches = []
        for concept, required in concept_features.items():
            overlap = required & feature_names
            if overlap:
                score = len(overlap) / len(required)
                matches.append((concept, score, list(overlap)))

        # Sort by match score, return concepts above threshold
        matches.sort(key=lambda x: -x[1])
        return [m[0] for m in matches if m[1] >= 0.3]


# ── Vision Engine (orchestrator) ───────────────────────────────────

class VisionEngine:
    """
    See an image → understand it through graph reasoning.

    Pipeline:
    1. Feature extraction (classical CV)
    2. Concept mapping (features → graph nodes)
    3. Description generation (concepts → natural language)
    """

    def __init__(self, db_path: str = None):
        self._extractor = FeatureExtractor()
        self._mapper = ConceptMapper(db_path)
        self._db_path = db_path or _DEFAULT_DB

    def see(self, image_path: str) -> SceneDescription:
        """Analyze an image and return a scene description."""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        return self.see_array(img)

    def see_array(self, img: np.ndarray) -> SceneDescription:
        """Analyze an image array."""
        # Step 1: Extract features (pure math)
        features = self._extractor.extract_all(img)
        log.info("Extracted %d features", len(features))

        # Step 2: Map to concepts (graph reasoning)
        concepts = self._mapper.features_to_concepts(features)
        log.info("Mapped to %d concepts: %s", len(concepts), concepts)

        # Step 3: Build scene description
        # Determine scene type
        if 'document' in concepts:
            scene_type = 'document'
        elif 'outdoor_scene' in concepts:
            scene_type = 'outdoor'
        elif 'nature' in concepts:
            scene_type = 'nature'
        else:
            scene_type = 'indoor'

        # Lighting
        if 'well_lit' in concepts:
            lighting = 'bright'
        elif 'dark_scene' in concepts:
            lighting = 'dark'
        else:
            lighting = 'medium'

        # Dominant colors
        dominant_colors = [f.name.replace('color_', '') for f in features
                          if f.name.startswith('color_') and f.value > 0.1]

        # Build objects from contour features
        objects = []
        for f in features:
            if f.name.startswith('shape_') and f.bbox:
                obj = VisualObject(
                    label=f.name.replace('shape_', ''),
                    confidence=f.value,
                    evidence=[
                        f"shape: {f.details.get('vertices', '?')} vertices",
                        f"circularity: {f.details.get('circularity', '?')}",
                        f"aspect: {f.details.get('aspect_ratio', '?')}",
                    ],
                    bbox=f.bbox,
                    features=[f]
                )
                objects.append(obj)

        # Generate description
        description = self._describe(scene_type, lighting, dominant_colors,
                                     objects, concepts, features)

        return SceneDescription(
            objects=objects,
            scene_type=scene_type,
            lighting=lighting,
            dominant_colors=dominant_colors,
            description=description,
            features=features,
        )

    def _describe(self, scene_type, lighting, colors, objects, concepts,
                  features) -> str:
        """Generate natural language description from analysis."""
        parts = []

        # Scene
        parts.append(f"A {lighting} {scene_type} scene")

        # Colors
        if colors:
            parts.append(f"with dominant {', '.join(colors[:3])} tones")

        # Objects
        if objects:
            shape_summary = {}
            for obj in objects:
                shape_summary[obj.label] = shape_summary.get(obj.label, 0) + 1
            shape_parts = [f"{count} {name}{'s' if count > 1 else ''}"
                          for name, count in shape_summary.items()]
            if shape_parts:
                parts.append(f"containing {', '.join(shape_parts)}")

        # Scene-specific details
        if 'nature' in concepts:
            parts.append("(outdoor nature scene with sky and vegetation)")
        if 'document' in concepts:
            parts.append("(appears to be a document or text)")

        # Texture
        for f in features:
            if f.name == 'texture_uniformity' and f.details:
                tex = f.details.get('description', '')
                if tex in ('smooth', 'rough'):
                    parts.append(f"with {tex} texture")

        # Edge complexity
        for f in features:
            if f.name == 'edge_density' and f.details:
                density = f.details.get('description', '')
                if density == 'high':
                    parts.append("(complex/detailed)")
                elif density == 'low':
                    parts.append("(simple/minimal)")

        return '. '.join([' '.join(parts[:3])] + parts[3:]) + '.'

    def teach_visual_concept(self, concept: str, visual_features: List[str]):
        """Teach the vision system what something looks like.

        Usage:
            engine.teach_visual_concept('dog', ['fur_texture', 'four_legs', 'tail', 'snout'])
            engine.teach_visual_concept('car', ['wheels', 'rectangular_body', 'windows', 'metal_texture'])

        Stores as graph edges: concept → has_visual → feature
        """
        conn = self._mapper._get_conn()
        if not conn:
            return

        cur = conn.cursor()
        for feature in visual_features:
            cur.execute(
                'INSERT OR IGNORE INTO graph_edges (from_id, relation, to_id) VALUES (?,?,?)',
                (concept, 'has_visual', feature)
            )
            # Also ensure the feature node exists
            cur.execute(
                'INSERT OR IGNORE INTO graph_nodes (node_id, label, node_type) VALUES (?,?,?)',
                (feature, feature.replace('_', ' '), 'visual_feature')
            )

        # Ensure concept node exists
        cur.execute(
            'INSERT OR IGNORE INTO graph_nodes (node_id, label, node_type) VALUES (?,?,?)',
            (concept, concept.replace('_', ' '), 'visual_concept')
        )
        conn.commit()
        log.info("Taught visual concept: %s → %s", concept, visual_features)
