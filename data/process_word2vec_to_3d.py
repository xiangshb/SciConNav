#!/usr/bin/env python3
"""
Script to process ALL Word2Vec models and convert them to 3D coordinates for visualization.
This script reads multiple Word2Vec models (different dimensions), applies dimensionality reduction (UMAP),
and saves the results in a single binary format compatible with the visualization system.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import umap
import struct
from pathlib import Path
import re

def load_word2vec_model(model_path):
    """Load Word2Vec model from the given path."""
    print(f"Loading Word2Vec model from {model_path}...")
    model = Word2Vec.load(model_path)
    print(f"Model loaded successfully. Vocabulary size: {len(model.wv)}")
    return model

def load_concept_data(csv_path):
    """Load concept data from CSV file."""
    print(f"Loading concept data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} concepts from CSV")
    return df

def extract_vectors(model, concept_df, target_concepts=None):
    """
    Extract vectors for concepts that are in the Word2Vec model.
    If target_concepts is provided, extracts vectors in that specific order.
    """
    print("Extracting vectors for concepts...")
    vectors = []
    concept_ids = []
    concept_names = []
    levels = []
    missing_concepts = []
    
    # Create a mapping from display_name to concept info
    display_name_to_info = {}
    for _, row in concept_df.iterrows():
        display_name = row['display_name']
        if display_name not in display_name_to_info:
            display_name_to_info[display_name] = []
        display_name_to_info[display_name].append({
            'id': row['id'],
            'level': row.get('level', 0),
            'level_0_ancestor': row.get('level_0_ancestor', ''),
            'llm_annotation': row.get('llm_annotation', '')
        })

    # Determine which concepts to process and in what order
    if target_concepts is not None:
        print(f"Using provided target list of {len(target_concepts)} concepts.")
        concepts_to_process = target_concepts
    else:
        print("Using model vocabulary as concept list.")
        concepts_to_process = list(model.wv.key_to_index.keys())
    
    # Process each concept
    for concept_name in concepts_to_process:
        # Check if concept exists in model AND in our CSV metadata
        if concept_name in model.wv and concept_name in display_name_to_info:
            # Use the first ID for this display name
            concept_info = display_name_to_info[concept_name][0]
            vectors.append(model.wv[concept_name])
            concept_ids.append(concept_info['id'])
            concept_names.append(concept_name)
            levels.append(concept_info['level'])
        else:
            if target_concepts is not None:
                # If we are enforcing a target list, missing a concept is a problem for alignment
                # We could fill with zeros, but for now let's warn and skip (which will fail the length check later)
                # Or better: Fill with zeros to maintain alignment?
                # Given the user wants strict alignment, if a model is missing a concept that others have,
                # that model is incompatible unless we impute.
                # For now, we'll skip adding it, which will trigger the length mismatch warning in the main loop.
                pass
            missing_concepts.append(concept_name)
            
    if target_concepts is not None and len(vectors) != len(target_concepts):
        print(f"Warning: Target list had {len(target_concepts)} concepts, but only found {len(vectors)} in this model.")
    
    return np.array(vectors), concept_ids, concept_names, levels, display_name_to_info

def reduce_dimensions(vectors, n_components=3):
    """Reduce dimensionality of vectors using UMAP."""
    print(f"Reducing dimensions from {vectors.shape[1]} to {n_components} using UMAP...")
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_vectors = reducer.fit_transform(vectors)
    print(f"Dimensionality reduction completed. Shape: {reduced_vectors.shape}")
    return reduced_vectors

def save_multi_dim_binary_format(all_reduced_vectors_3d, all_reduced_vectors_2d, concept_ids, concept_names, levels, display_name_to_info, output_path):
    """Save reduced vectors for multiple dimensions (both 3D and 2D) and concept information in binary format."""
    print(f"Saving multi-dimension data to {output_path}...")
    
    # Ensure all dimensions have the same number of points and same concept order
    # We assume the first model processed sets the standard for concept_ids and names
    n_points = len(concept_ids)
    dimensions = sorted(all_reduced_vectors_3d.keys())
    n_dims = len(dimensions)
    
    print(f"Dimensions to save: {dimensions}")
    
    # Create a mapping from concept_name to llm_annotation
    name_to_annotation = {}
    for name, info_list in display_name_to_info.items():
        if info_list:
            name_to_annotation[name] = info_list[0]['llm_annotation']
    
    # Map annotations to discipline IDs
    unique_annotations = list(set(name_to_annotation.values()))
    unique_annotations.sort(key=lambda x: (x != "Art", x))
    annotation_to_discipline = {ann: idx for idx, ann in enumerate(unique_annotations)}
    
    # Get unique levels for metadata
    unique_levels = sorted(list(set(levels)))
    
    # Prepare binary data
    # Format:
    # Header: n_points (uint32), n_dims (uint32), dim_1 (uint32), dim_2 (uint32), ...
    # Body: For each point:
    #   discipline_id (uint32)
    #   level (uint32)
    #   For each dimension:
    #     x3 (float32), y3 (float32), z3 (float32)  <-- 3D coordinates
    #     x2 (float32), y2 (float32)                <-- 2D coordinates
    
    binary_data = bytearray()
    
    # Header
    binary_data.extend(struct.pack('<II', n_points, n_dims))
    for dim in dimensions:
        binary_data.extend(struct.pack('<I', dim))
        
    # Body
    for i in range(n_points):
        concept_name = concept_names[i]
        level = levels[i]
        
        # Get discipline ID
        annotation = name_to_annotation.get(concept_name, "Unknown")
        discipline_id = annotation_to_discipline.get(annotation, 0)
        
        # Write static info for point
        binary_data.extend(struct.pack('<II', discipline_id, level))
        
        # Write coordinates for each dimension
        for dim in dimensions:
            # 3D Point
            point3d = all_reduced_vectors_3d[dim][i]
            binary_data.extend(struct.pack('<fff', point3d[0], point3d[1], point3d[2]))
            
            # 2D Point
            point2d = all_reduced_vectors_2d[dim][i]
            binary_data.extend(struct.pack('<ff', point2d[0], point2d[1]))
            
    # Write binary data
    with open(output_path, 'wb') as f:
        f.write(binary_data)
    
    print(f"Binary data saved successfully. {n_points} points, {n_dims} dimensions.")
    
    # Create and save metadata
    metadata = {
        "n_points": n_points,
        "n_disciplines": len(unique_annotations),
        "discipline_names": unique_annotations,
        "discipline_to_annotation": {idx: ann for ann, idx in annotation_to_discipline.items()},
        "n_levels": len(unique_levels),
        "level_names": [str(level) for level in unique_levels],
        "level_to_value": {idx: level for idx, level in enumerate(unique_levels)},
        "available_dimensions": dimensions,
        "format": "multi_dim_binary_v2",
        "header_format": "n_points(uint32), n_dims(uint32), [dim_val(uint32)...]",
        "point_format": "discipline_id(uint32), level(uint32), [x3(float32), y3(float32), z3(float32), x2(float32), y2(float32) per dim]",
        "point_size_bytes": 8 + (20 * n_dims)  # 4+4 + ((3+2)*4 * n_dims) = 8 + 20*n_dims
    }
    
    metadata_path = output_path.replace('.bin', '_meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ALL Word2Vec models to multi-dim 3D coordinates')
    parser.add_argument('--models_dir', type=str, default='models/',
                        help='Directory containing Word2Vec model files')
    parser.add_argument('--csv', type=str, default='data/All_concepts_with_ancestors_llm_annotation.csv',
                        help='Path to the concept CSV file')
    parser.add_argument('--output', type=str, default='data/concept_coordinates.bin',
                        help='Output path for the binary file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.models_dir):
        print(f"Error: Models directory {args.models_dir} not found.")
        sys.exit(1)
        
    if not os.path.exists(args.csv):
        print(f"Error: CSV file {args.csv} not found.")
        sys.exit(1)
        
    # Find all model files
    model_files = [f for f in os.listdir(args.models_dir) if f.startswith('Word2Vec_dim_') and f.endswith('.model')]
    
    if not model_files:
        print("No model files found matching pattern 'Word2Vec_dim_*.model'")
        sys.exit(1)
        
    # Sort by dimension
    def get_dim(filename):
        match = re.search(r'dim_(\d+)', filename)
        return int(match.group(1)) if match else 0
        
    model_files.sort(key=get_dim)
    
    print(f"Found {len(model_files)} models: {model_files}")
    
    # Load concept data once
    concept_df = load_concept_data(args.csv)
    
    all_reduced_vectors_3d = {}
    all_reduced_vectors_2d = {}
    base_concept_ids = None
    base_concept_names = None
    base_levels = None
    base_display_info = None
    
    for model_file in model_files:
        dim = get_dim(model_file)
        model_path = os.path.join(args.models_dir, model_file)
        
        print(f"\nProcessing dimension {dim}...")
        
        try:
            model = load_word2vec_model(model_path)
            
            # If we already have a base list of concepts, enforce that order
            target_list = base_concept_names if base_concept_names is not None else None
            
            vectors, concept_ids, concept_names, levels, display_name_to_info = extract_vectors(model, concept_df, target_concepts=target_list)
            
            # Ensure consistency across models
            if base_concept_ids is None:
                # First model sets the standard
                base_concept_ids = concept_ids
                base_concept_names = concept_names
                base_levels = levels
                base_display_info = display_name_to_info
                print(f"Set base concept list with {len(base_concept_names)} concepts.")
            else:
                # Verify alignment
                if len(concept_ids) != len(base_concept_ids):
                    print(f"Error: Dimension {dim} has {len(concept_ids)} concepts, expected {len(base_concept_ids)} to match base model.")
                    print("This model might be missing concepts present in the first model.")
                    print("Skipping this dimension to preserve alignment.")
                    continue
                
                # Double check names match exactly (paranoia check)
                if concept_names != base_concept_names:
                    print(f"Error: Concept order mismatch in dimension {dim} despite same count!")
                    print("Skipping this dimension.")
                    continue
            
            # Reduce to 3D
            print("  -> Generating 3D coordinates...")
            reduced_3d = reduce_dimensions(vectors, n_components=3)
            all_reduced_vectors_3d[dim] = reduced_3d
            
            # Reduce to 2D
            print("  -> Generating 2D coordinates...")
            reduced_2d = reduce_dimensions(vectors, n_components=2)
            all_reduced_vectors_2d[dim] = reduced_2d
            
        except Exception as e:
            print(f"Error processing {model_file}: {e}")
            continue
            
    if not all_reduced_vectors_3d:
        print("No data processed successfully.")
        sys.exit(1)
        
    # Save combined data
    save_multi_dim_binary_format(all_reduced_vectors_3d, all_reduced_vectors_2d, base_concept_ids, base_concept_names, base_levels, base_display_info, args.output)