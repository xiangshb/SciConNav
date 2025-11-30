from model import Embedding
from visualization import Visualizer
import argparse
from concept import Concept

level_0_concepts = ['Art', 'Biology', 'Business', 'Chemistry', 'Computer science',
       'Economics', 'Engineering', 'Environmental science', 'Geography',
       'Geology', 'History', 'Materials science', 'Mathematics',
       'Medicine', 'Philosophy', 'Physics', 'Political science',
       'Psychology', 'Sociology']

def get_args():
    parser = argparse.ArgumentParser(description='OpenAlex Parameters')
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--sub', type=int, default=0)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--cmd', type=bool, default=False)
    parser.add_argument('--mode', type=int, default=0) # 1 for weight length, be careful
    return parser.parse_args()

if __name__=='__main__':
    parser = get_args()
    # subgraph = Concept.Tree(sub_graph_attribute_values = ['display_name', ['Computer science', 'Tree kernel']], show = True, fig_size=(8, 8), font_size = 26, bbox_inches = 'tight', pad_inches = 0.2, save = True, x_scale_ratio = 0.4)
    # subgraph = Concept.Tree(sub_graph_attribute_values = ['display_name', ['Engineering', 'Cascade amplifier']], show = True, fig_size=(7, 7), font_size = 26, bbox_inches = 'tight', pad_inches = 0.2, save = True, x_scale_ratio = 0.3)
    # Concept.sample_concepts_with_ancestor_paths()
    
    # Embedding.outside_pairs_similarity_verify()
    # Embedding.concept_prerequisites_verify()
    # Embedding.concept_embedding_map(k_th =  parser.k)
    # Visualizer.concept_embedding_plot(n_areas = 2, level = 3, less_than=True)
    # Visualizer.concept_embedding_plot(n_areas = 2, level = 2, less_than=False, colors = ['lime', 'orangered', 'violet'])
    # Embedding.concept_embedding_map(subset = 0, vec_dim = 24, metric = 'cosine', tsne_lib = 'op', save=True, label_refined = True, emphasize_discipline = 'Interdiscipline')
    # Embedding.analogy_inference(k_th = parser.k, probe_depth = parser.d, level = 0, multi_level = True, show = False)
    # Embedding.analogy_inference(show  = True, probe_depth=2)
    # Embedding.concept_dists()
    # Embedding.functional_axis_similarity()
    # Embedding.ancestor_inclination(concept_discipline = 'All disciplines')
    # Embedding.ancestor_inclination(concept_discipline = 'interdiscipline') # concept_discipline = 'interdiscipline'
    # Embedding.all_pairs_shortest_path()
    # while True:
    #     Embedding.concept_embedding_map(vec_dim = 24, metric = 'cosine', with_random_route=True, tsne_lib = 'op', label_refined = True, figsize = (6,6), save=True, draw_route_region = True) # closeness
    # route_s_es=[['Geodesic', 'Walkability'],['Markov process', 'Image processing'],['Statistical inference', 'Internet Protocol'],['Statistical inference', 'Face detection'],['Mathematical optimization', 'Web Accessibility Initiative'],['Likelihood principle',  'Deep neural networks'],['Hessian matrix',  'Robot control'],['Internet privacy',  'Pure mathematics']]
    # route_s_es=[['Statistical inference', 'Face detection']]
    # route_s_es = [['Statistics', 'Biological activity'], ['Statistics', 'Cell biology'], ['Statistics', 'Plant growth'], ['Artificial intelligence', 'Nursing'], ['Artificial intelligence', 'Propensity score matching'], ['Artificial intelligence', 'Physical exercise'], ['Artificial intelligence', 'Vascular surgery']]

    # route_s_es = [['Mathematical analysis', 'Decision analysis'], ['Discrete mathematics', 'Bayes estimator'], ['Mathematical analysis', 'Optimization algorithm'], ['Pure mathematics', 'Data envelopment analysis']]
    
    # route_s_es = [['Pure mathematics', 'Bioinformatics'], ['Cryptography', 'Bioinformatics'], ['Combinatorics', 'Radar'], ['Mathematical analysis', 'Radar'], ['Mathematical optimization', 'Digital control']]
    
    # for route_s_e in route_s_es:
    #     Embedding.concept_path_embedding_map(vec_dim = 24, metric = 'cosine', route_s_e = route_s_e, tsne_lib = 'op', label_refined = True, figsize = (6,6), save=True, draw_route_region = True, path_len_gt_5 = False) # closeness
    # Embedding.concept_embedding_map(vec_dim = 24, subset=0, metric = 'cosine', tsne_lib = 'op', label_refined = True, figsize = (6,5), save=True, no_inter=False) # betweenness
    # Embedding.concept_embedding_map(vec_dim = 24, subset=0, metric = 'cosine', tsne_lib = 'op', label_refined = True, figsize = (9,6), save=True, no_inter=False, centrality_type = 'betweenness', point_size=3) # betweenness
    # Embedding.interdisciplinary_centrality_analysis()
    # Embedding.functional_axis_similarity(discipline_groups = ['theoretical', 'applied']) # ['chemistry', 'biomedical'], ['applied', 'biomedical']
    # Embedding.shortest_path_length_distribution(start_area = 'Mathematics', target_split = True, mat_type = 'path_weight_mat', with_multiple_ancestor=False, save=True)
    # Embedding.shortest_path_length_distribution(start_area = 'Mathematics', target_split = False, mat_type = 'step_size_mat', with_multiple_ancestor=False, save=True)
    # Embedding.shortest_path_length_distribution(start_area = 'Mathematics', mat_type = 'path_weight_mat', return_accessibility=False, target_split=True) # path_weight_mat step_size_mat
    # print(5)
    # Embedding.ancestor_inclination(concept_type = 'with_domain')
    # Embedding.concepts_reachable_route_search()
    # Embedding.concept_dists(k_th = parser.k)
    # Embedding.search_central_popular_concepts(k_th = parser.k)
    # Embedding.all_route_dists()
    # mat = Embedding.all_pairs_path_length_mat(mode = parser.mode) 
# python concept_embedding.py --k 1 --sub 0

# python -m pip install --global-option=build_ext  --global-option="-IC:\Program Files\Graphviz\include"  --global-option="-LC:\Program Files\Graphviz\lib" pygraphviz
# pip install --global-option=build_ext --global-option="-L/opt/lib/mygviz/" --global-option="-R/opt/lib/mygviz/" pygraphviz

