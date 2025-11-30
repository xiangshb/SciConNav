import os
import json
import pandas as pd
from gensim.models import Word2Vec
import argparse

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

def extract_ordered_metadata(model, concept_df):
    """
    Extract concept names and metadata in the exact same order as the binary generation script.
    This mirrors the logic in process_word2vec_to_3d.py's extract_vectors function.
    """
    print("Extracting ordered metadata...")
    
    # Get all concepts from the model (these are the unique display names)
    # The order of keys() is deterministic for a loaded model
    model_concepts = list(model.wv.key_to_index.keys())
    
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
    
    ordered_names = []
    ordered_ids = []
    ordered_annotations = []
    
    # Process each concept in the model - MUST MATCH process_word2vec_to_3d.py logic exactly
    for concept_name in model_concepts:
        if concept_name in display_name_to_info:
            # Use the first ID for this display name
            concept_info = display_name_to_info[concept_name][0]
            
            ordered_names.append(concept_name)
            ordered_ids.append(concept_info['id'])
            ordered_annotations.append(concept_info['llm_annotation'])
            
    return ordered_names, ordered_ids, ordered_annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate metadata JSON for visualization')
    parser.add_argument('--model', type=str, default='sciconnav-gemini/models/Word2Vec_dim_128_epoch_100.model',
                        help='Path to the Word2Vec model file')
    parser.add_argument('--csv', type=str, default='sciconnav-gemini/data/All_concepts_with_ancestors_llm_annotation.csv',
                        help='Path to the concept CSV file')
    parser.add_argument('--output', type=str, default='sciconnav-gemini/data/concept_metadata.json',
                        help='Output path for the metadata JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        # Try relative path if run from sciconnav-gemini folder
        alt_model = args.model.replace('sciconnav-gemini/', '')
        if os.path.exists(alt_model):
            args.model = alt_model
            print(f"Found model at {args.model}")
        else:
            exit(1)

    if not os.path.exists(args.csv):
        print(f"Error: CSV file {args.csv} not found.")
        # Try relative path
        alt_csv = args.csv.replace('sciconnav-gemini/', '')
        if os.path.exists(alt_csv):
            args.csv = alt_csv
            print(f"Found CSV at {args.csv}")
        else:
            exit(1)
            
    model = load_word2vec_model(args.model)
    concept_df = load_concept_data(args.csv)
    
    names, ids, annotations = extract_ordered_metadata(model, concept_df)
    
    print(f"Extracted {len(names)} concepts.")
    
    # Save as a simple list of names to save space, or a list of objects if we need more info
    # For now, let's save a dict with parallel arrays to be efficient
    output_data = {
        "names": names,
        "ids": ids,
        # annotations are already in the binary file (as discipline IDs), but names might be useful for search
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f)
        
    print(f"Saved metadata to {args.output}")