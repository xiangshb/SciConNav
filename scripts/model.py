from gensim.models.callbacks import CallbackAny2Vec
from scipy.spatial import KDTree
import networkx as nx
import os.path as op
from dataset import OpenAlexData, Process, is_in_server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools, os, tqdm
from scipy.spatial import distance
import datetime
import gc

# from scipy.spatial.distance import euclidean as eudist
# from scipy.spatial import Delaunay, delaunay_plot_2d, tsearch

class CONFIG():
    level = 1
    dims = [96, 64, 48, 32, 24]

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''
    def __init__(self):
        self.epoch = 0
        self._test_words = ['Mathematics', 'Sociology', 'Physics']

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        for word in self._test_words:print(word,': ',model.wv.most_similar(word, topn=3))
        self.epoch += 1

class Embedding(Process):
    # @classmethod
    # def train_model(cls, vec_dim, training = True, level = None, args = CONFIG):
    #     level = level if level else args.level
    #     model_path = op.join(OpenAlexData.model_path, 'embedding_model_level_' + str(level) + '_dim_' + str(vec_dim) + '.model')
    #     from gensim.models import Word2Vec
    #     if training and not op.exists(model_path):
    #         print('current level ', level)
    #         lists = OpenAlexData.get_concept_lists(level=level)
    #         for n_dim in args.dims:
    #             print('training for vector dim ', n_dim)
    #             model_path_n_dim = op.join(OpenAlexData.model_path, 'embedding_model_level_' + str(level) + '_dim_' + str(n_dim) + '.model')
    #             model = Word2Vec(vector_size = n_dim, min_count=0, sg = 0, hs = 1, alpha=0.03, workers = 1, compute_loss=True, callbacks=[callback()])
    #             model.build_vocab(lists, progress_per=100000)
    #             model.train(lists, total_examples=model.corpus_count, epochs=100)
    #             model.save(model_path_n_dim)
    #     else: model = Word2Vec.load(model_path)
    #     return model
    @classmethod
    def train_model(cls, vec_dim, training = False, level = None, args = CONFIG, multi_level = False, multi_dim = False):
        # level 3 means for all levels from level 2 to level 5 concepts
        level = level if level is not None else args.level
        if multi_level: level_adder = '_level_' + str(level) + '_to_5' if level else '_level_0_to_5'
        else: level_adder = '_level_' + str(level) if level in range(6) else ''
        model_path = op.join(OpenAlexData.model_path, f'embedding_model_dim_{vec_dim}.model') # 
        from gensim.models import Word2Vec
        if training and not op.exists(model_path):
            print('current level ', level, ' multi_level ', 'True' if multi_level else 'False')
            lists = OpenAlexData.get_concept_lists(level=level, multi_level = multi_level)
            if multi_dim:
                for n_dim in args.dims:
                    print('training for vector dim ', n_dim)
                    model_path_n_dim = op.join(OpenAlexData.model_path, 'embedding_model' + level_adder + '_dim_' + str(n_dim) + '.model')
                    model = Word2Vec(vector_size = n_dim, min_count=0, sg = 0, hs = 1, alpha=0.03, workers = 1, compute_loss=True, callbacks=[callback()])
                    model.build_vocab(lists, progress_per=100000)
                    model.train(lists, total_examples=model.corpus_count, epochs=100)
                    model.save(model_path_n_dim)
                    print(model_path_n_dim,' cached')
            else:
                print('training for vector dim ', vec_dim)
                model = Word2Vec(vector_size = vec_dim, min_count=0, sg = 0, hs = 1, alpha=0.03, workers = 1, compute_loss=True, callbacks=[callback()])
                model.build_vocab(lists, progress_per=100000)
                model.train(lists, total_examples=model.corpus_count, epochs=100)
                model.save(model_path)
                print(model_path,' cached')
        else: model = Word2Vec.load(model_path)
        return model
    
    @classmethod
    def concept_prerequisites_verify(cls):
        file_path = op.join(cls.external_data_dir, 'conceptnet-assertions-5.7.0', 'assertions.csv')
        import csv
        with open(file_path, 'rt', newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            print(type(reader))
            for row in reader:
                print(row[:3])
        # assertions = pd.read_csv(file_path)
        print(5)

    @classmethod
    def outside_pairs_similarity_verify(cls, model_pars = [24, 0, True]):
        dim_vec, level_model, multi_level = model_pars
        model = cls.train_model(vec_dim = dim_vec, level = level_model, multi_level = multi_level)
        pair_path = op.join(os.path.dirname(__file__), 'drug-related-pairs\DataS4_disease_pairs.csv')
        pairs_table = pd.read_csv(pair_path)
        pairs_table.disease_A = pairs_table.disease_A.map(lambda s:s.replace(',','').capitalize())
        pairs_table.disease_B = pairs_table.disease_B.map(lambda s:s.replace(',','').capitalize())
        disease_A = pairs_table.disease_A.drop_duplicates().reset_index(drop=True)
        disease_B = pairs_table.disease_B.drop_duplicates().reset_index(drop=True)
        disease_A_B = list(set(disease_A).union(disease_B))
        df_disease_A_B = pd.Series(disease_A_B)
        concepts_valid = df_disease_A_B.loc[df_disease_A_B.isin(model.wv.index_to_key)]
        pairs_table_valid = pairs_table.loc[pairs_table.disease_A.isin(concepts_valid) & pairs_table.disease_B.isin(concepts_valid)]
        pairs_table_valid['sim_AB'] = [model.wv.similarity(*pair) for pair in pairs_table_valid[['disease_A','disease_B']].values.tolist()]
        sim_data = pairs_table_valid[['s_AB(observed)','sim_AB']]
        sim_data.plot.scatter(x='s_AB(observed)',y='sim_AB')
        print(5)

    def search_space_nearest_analogy(self, model_wv, inputs, probe_depth, marker, depth = 0, distinct = False):
        word_a, word_b, word_seed = inputs
        for word in inputs: marker[model_wv.key_to_index[word]] = True 
        axis_vector = model_wv.get_vector(word_b) - model_wv.get_vector(word_a)
        edges = []
        if depth < probe_depth:
            forward, backword = model_wv.get_vector(word_seed) + axis_vector, model_wv.get_vector(word_seed) - axis_vector
            if distinct:
                unmarked_words = np.array(model_wv.index_to_key)[~marker]
                word_forward = unmarked_words[KDTree(model_wv.vectors[~marker]).query(forward)[1]]
                marker[model_wv.key_to_index[word_forward]] = True
                unmarked_words = np.array(model_wv.index_to_key)[~marker]
                word_backward = unmarked_words[KDTree(model_wv.vectors[~marker]).query(backword)[1]]
                marker[model_wv.key_to_index[word_backward]] = True
            else:
                candidate_indexes = [word not in inputs for word in model_wv.index_to_key]
                candidate_words = np.array(model_wv.index_to_key)[candidate_indexes]
                kdtree = KDTree(model_wv.vectors[candidate_indexes])
                word_forward = candidate_words[kdtree.query(forward)[1]]
                word_backward = candidate_words[kdtree.query(backword)[1]]

            edges += [[word_seed, word_backward, 0], [word_seed, word_forward, 1]]
            edges += self.search_space_nearest_analogy(self, model_wv, [word_a, word_b, word_forward], probe_depth, marker, depth+1, distinct)
            edges += self.search_space_nearest_analogy(self, model_wv, [word_a, word_b, word_backward], probe_depth, marker, depth+1, distinct)
        return edges
    
    def search_most_similarity_analogy(self, model_wv, inputs, probe_depth, marker, depth = 0, distinct = False):
        # a->b: c->d, b-a=d-c, d=c+b-a, positive: c, b, netative: a
        # b->a: c->d, a-b=d-c, d=c+a-b, positive: c, a, netative: b
        word_a, word_b, word_seed = inputs
        edges = []
        try: marker[model_wv.key_to_index[word_seed]] = True # mark the searched words
        except: return edges
        if depth < probe_depth:
            if distinct:
                similar_words = np.array(model_wv.most_similar(negative=[word_a], positive=[word_seed, word_b], topn=2**(probe_depth+1))).T[0]
                marked_words = np.array(model_wv.index_to_key)[marker].tolist()
                word_forward = [word for word in similar_words if word not in marked_words][0] # make sure different from inputs
                marker[model_wv.key_to_index[word_forward]] = True

                similar_words = np.array(model_wv.most_similar(negative=[word_a], positive=[word_seed, word_b], topn=2**(probe_depth+1))).T[0]
                marked_words = np.array(model_wv.index_to_key)[marker].tolist()
                word_backward = [word for word in similar_words if word not in marked_words] # make sure different from inputs
                marker[model_wv.key_to_index[word_backward]] = True
            else:
                similar_words = np.array(model_wv.most_similar(negative=[word_a], positive=[word_seed, word_b])).T[0]
                word_forward = [word for word in similar_words if word not in inputs][0] # make sure different from inputs
                similar_words = np.array(model_wv.most_similar(negative=[word_b], positive=[word_seed, word_a])).T[0]
                word_backward = [word for word in similar_words if word not in inputs][0]

            edges += [[word_seed, word_backward, 0], [word_seed, word_forward, 1]]
            edges += self.search_most_similarity_analogy(self, model_wv, [word_a, word_b, word_forward], probe_depth, marker, depth+1, distinct = distinct)
            edges += self.search_most_similarity_analogy(self, model_wv, [word_a, word_b, word_backward], probe_depth, marker, depth+1, distinct = distinct)
        return edges

    def analogy_inference_network(self, inputs, model, probe_depth, ancestor_map_df, level_map, mode = 'distance', colors = ['lime', 'dodgerblue', 'gray', 'red'], precise = False): # dodgerblue orange lightsteelblue
        word_a, word_b, word_seed = inputs
        ancestor_map = dict(ancestor_map_df[['display_name','level_0_ancestor_refined']].drop_duplicates().values)
        multiple_ancestor_map = dict(ancestor_map_df[['display_name','multiple_level_0_ancestors']].drop_duplicates().values)
        ances_a, ances_b = ancestor_map[word_a], ancestor_map[word_b]
        G, marker = nx.DiGraph(), np.full(len(model.wv), False)
        for word in inputs[:2]: marker[model.wv.key_to_index[str(word)]] = True
        if mode == 'similarity': edges = self.search_most_similarity_analogy(self, model.wv, inputs, probe_depth, marker = marker)
        elif mode == 'distance': edges = self.search_space_nearest_analogy(self, model.wv, inputs, probe_depth, marker = marker)
        else: raise ValueError('Undefined mode type')
        edges_df = pd.DataFrame(edges, columns=['edge_start','edge_end','direction']).drop_duplicates()
        edges_df.index = range(edges_df.index.shape[0])
        G.add_weighted_edges_from(edges_df.values)
        edges_df['edge_color'] = edges_df.direction.map( dict( {0:colors[0], 1:colors[1]} ) )
        edge_colors = dict([tuple([edge_start, edge_end]), edge_color] for edge_start, edge_end, edge_color in edges_df[['edge_start','edge_end','edge_color']].values)
        nx.set_edge_attributes(G, edge_colors, 'edge_color')
        nx.set_node_attributes(G, ancestor_map, 'ancestor')
        nx.set_node_attributes(G, multiple_ancestor_map, 'multiple_ancestor')
        nx.set_node_attributes(G, level_map, 'level')
        node_colors, multiple_ancestor_node_colors = dict(), dict()
        for node in G.nodes:
            if node == word_seed: 
                node_colors[node] = colors[0] if ancestor_map[word_seed]==ancestor_map[word_a] else (colors[1] if ancestor_map[word_seed]==ancestor_map[word_b] else colors[2])
                multiple_ancestor_node_colors[node] = node_colors[node]
            else: 
                values, counts = np.unique([G[u][v]['weight'] for u,v in G.in_edges(node)], return_counts=True)
                direction = values[np.argmax(counts)]
                majority_voting_type, counter_majority_type = ancestor_map[inputs[direction]], ancestor_map[inputs[1-direction]] 

                multiple_type = multiple_ancestor_map[node].split(',') 
                if majority_voting_type in multiple_type:
                    multiple_ancestor_node_colors[node] = colors[direction]
                elif counter_majority_type in multiple_type:
                    multiple_ancestor_node_colors[node] = colors[1-direction]
                else: multiple_ancestor_node_colors[node] = colors[3]

                actual_type = ancestor_map[node] 
                if actual_type == majority_voting_type: node_colors[node] = colors[direction]
                elif precise: node_colors[node] = colors[3]
                elif actual_type ==  counter_majority_type: node_colors[node] = colors[1-direction]
                else: node_colors[node] = colors[3]
        nx.set_node_attributes(G, node_colors, 'node_color')
        nx.set_node_attributes(G, multiple_ancestor_node_colors, 'ma_node_color') 
        edges_df['analogy_end_ancestor'] = edges_df.direction.map({0:ances_a, 1:ances_b})
        
        edges_df['edge_end_ancestor'] = edges_df.edge_end.map(ancestor_map)
        edges_df['eeac'] = edges_df.analogy_end_ancestor == edges_df.edge_end_ancestor
        edges_df['eearc'] = edges_df.edge_end_ancestor.isin([ances_a, ances_b])

        edges_df['edge_end_mutiple_ancestor'] = edges_df.edge_end.map(multiple_ancestor_map)
        edges_df['eemac'] = [edges_df.analogy_end_ancestor[i] in edges_df.edge_end_mutiple_ancestor[i] for i in range(edges_df.shape[0])]
        edges_df['eemarc'] = [(ances_a in edges_df.edge_end_mutiple_ancestor[i]) or (ances_b in edges_df.edge_end_mutiple_ancestor[i]) for i in range(edges_df.shape[0])]
        
        def set_edge_attribute(correction_type, attribute_name, columns = []):
            if len(columns) == 2:
                correct_edges = edges_df.loc[edges_df[correction_type]][columns].values
                target_edge_map = dict([[tuple(correct_edges[i]), i+1] for i in range(len(correct_edges))])
            elif len(columns) == 0:
                edges_df[attribute_name] = edges_df[correction_type].map(dict({True:'solid', False:'dashed'}))
                target_edge_map = dict([[tuple([edge_start, edge_end]), style] for edge_start, edge_end, style in edges_df[['edge_start','edge_end', attribute_name]].values])
            else: raise ValueError('undefined columns type')
            nx.set_edge_attributes(G, target_edge_map, attribute_name)
        set_edge_attribute('eeac', 'ce', ['edge_start','edge_end']) 
        set_edge_attribute('eemac', 'mace', ['edge_start','edge_end']) 
        
        set_edge_attribute('eeac', 'edge_style')
        set_edge_attribute('eemac', 'ma_edge_style') 
        
        return G, [edges_df.eeac.sum(), edges_df.eemac.sum(), edges_df.eearc.sum(), edges_df.eemarc.sum(), edges_df.shape[0]]
    
    @classmethod
    def get_triplet(cls, model, selection_requirements = ['Mathematics', 1, 'Computer science', 1, 'Computer science', 4], top_num = 20, equal = True):
        concept_a_area, level_a, concept_b_area, level_b, concept_seed_area, level_seed = selection_requirements
        concept_table = pd.read_csv(op.join(OpenAlexData.concept_tree_path, 'All_concepts_with_ancestors.csv'))
        concept_table = concept_table.loc[concept_table.display_name.isin(model.wv.index_to_key)]
        get_condition = lambda area, level: ((concept_table.level_0_ancestor_refined==area) & (concept_table.level==level)) if level >= 0 else concept_table.level_0_ancestor_refined==area
        sub_table_a = concept_table.loc[get_condition(concept_a_area, level_a)].sort_values('works_count', ascending=False)
        sub_table_b = concept_table.loc[get_condition(concept_b_area, level_b)].sort_values('works_count', ascending=False)
        sub_table_seed = concept_table.loc[get_condition(concept_seed_area, level_seed)].sort_values('works_count', ascending=False)
        table_min = min([sub_table_a.shape[0], sub_table_b.shape[0], sub_table_seed.shape[0]])
        if equal: top_num = table_min if table_min < top_num else top_num
        sub_table_a = sub_table_a[['display_name', 'works_count']][:top_num].reset_index(drop=True)
        sub_table_b = sub_table_b[['display_name', 'works_count']][:top_num].reset_index(drop=True)
        sub_table_seed = sub_table_seed[['display_name', 'works_count']][:top_num].reset_index(drop=True)
        sub_table = pd.concat([sub_table_a, sub_table_b, sub_table_seed], axis=1)
        sub_table.columns = [concept_a_area.replace(' ','_')+'_level_'+str(level_a),'work_count_a', concept_b_area.replace(' ','_')+'_level_'+str(level_b),'work_count_b', concept_seed_area.replace(' ','_')+'_level_'+str(level_seed),'work_count_seed']
        return sub_table

    @classmethod
    def analogy_inference(cls, k_th = -1, level = 0, dim = 24, probe_depth = 5, multi_level = True, show = False):
        print('depth ', probe_depth)
        model = cls.train_model(vec_dim = dim, level = level, multi_level = multi_level)

        table = pd.read_csv(op.join(OpenAlexData.concept_tree_path, 'All_concepts_with_ancestors.csv'))
        level_map = dict(table[['display_name','level']].drop_duplicates().values)
        ancestor_map = dict(table[['display_name','level_0_ancestor_refined']].drop_duplicates().values)
        ancestor_map_df = table[['display_name','level_0_ancestor_refined','multiple_level_0_ancestors']].drop_duplicates()
        
        if k_th < 0:
            inference_df, dfs = [], []
            accuracy_file_all_path = op.join(OpenAlexData.inference_path, 'analogy_inference_accuracy_v2_depth_'+str(probe_depth) + '.csv')
            if not os.path.exists(accuracy_file_all_path):
                for k_th in range(15):
                    accuracy_file_path = op.join(OpenAlexData.inference_path, 'analogy_inference_accuracy_v2_depth_'+str(probe_depth) + '_range_' + str(k_th) +'.csv')
                    inference_df.append(pd.read_csv(accuracy_file_path))
                inference_df = pd.concat(inference_df)
                inference_df.index = range(inference_df.shape[0])
                inference_df['ances_concept_a'] = inference_df.concept_a.map(ancestor_map)
                inference_df['concept_a_level'] = inference_df.concept_a.map(level_map)
                inference_df['ances_concept_b'] = inference_df.concept_b.map(ancestor_map)
                inference_df['concept_b_level'] = inference_df.concept_b.map(level_map)
                inference_df['ances_concept_seed'] = inference_df.concept_seed.map(ancestor_map)
                inference_df['concept_seed_level'] = inference_df.concept_seed.map(level_map)
                for column in ['n_sdnc','n_mdnc','n_sdnrc','n_mdnrc']:
                    inference_df[column[2:]] = inference_df[column] / inference_df.n_edges
                inference_df.to_csv(accuracy_file_all_path, index=False)
            else: inference_df = pd.read_csv(accuracy_file_all_path)
            if probe_depth == 5:
                sub_df = inference_df.loc[(inference_df.sdnc<0.6)&(inference_df.mdnc>0.98)&(inference_df.ances_concept_a.isin(['Mathematics','Computer science']))&(inference_df.ances_concept_b.isin(['Mathematics','Computer science']))&(inference_df.n_edges>=30)]
            elif probe_depth == 2:
                sub_df = inference_df.loc[(inference_df.n_sdnc<2)&(inference_df.n_mdnc>5)&(inference_df.ances_concept_a.isin(['Mathematics','Computer science']))&(inference_df.ances_concept_b.isin(['Mathematics','Computer science']))]
            else: raise ValueError('undefined probe_depth')

            sub_table = cls.get_triplet(model, selection_requirements = ['Mathematics', 1, 'Computer science', 1, 'Computer science', 5], equal=False)
            from visualization import Visualizer
            for triplet in sub_df[['concept_a', 'concept_b', 'concept_seed']].values: 
                title = ' : '.join([triplet[0]+'('+ancestor_map[triplet[0]]+')', triplet[1]+'('+ancestor_map[triplet[1]]+')'])
                plt.close('all')
                G, accuracy = cls.analogy_inference_network(cls, triplet, model, probe_depth, ancestor_map_df, level_map, mode = 'similarity')
                if show: 
                    titles = [title + '\n' + triplet[2]+'('+ancestor_map[triplet[2]]+')' + '\nsdnc = ' + str(accuracy[0]) + ' / ' + str(accuracy[4]) + ' = ' + str(round(accuracy[0]/accuracy[4],4)),
                              title + '\n' + triplet[2]+'('+ancestor_map[triplet[2]]+')'  + '\nmdnc = ' + str(accuracy[1]) + ' / ' + str(accuracy[4]) + ' = ' + str(round(accuracy[1]/accuracy[4],4))]
                    Visualizer.draw_analogy_graph([G,G], fig_size=(8,8), layout='spiral', title = titles, combine = False, discipline_abbrev = True, save = False) # layout='spring', 'spectral', ''
        else: 
            accuracy_file_path = op.join(OpenAlexData.inference_path, 'analogy_inference_accuracy_v2_depth_'+str(probe_depth) + '_range_' + str(k_th) +'.csv')
            print(accuracy_file_path)
            concept_triplet_path = op.join(OpenAlexData.concept_tree_path, 'concept_analogy_triplets.csv')
            if not os.path.exists(concept_triplet_path):
                inputs_all = []
                level_0_concepts = table.level_0_ancestor.drop_duplicates().values
                area_pairs = np.array(list(itertools.combinations(level_0_concepts, 2)))
                for area_1, area_2 in tqdm.tqdm(area_pairs):
                    table_1, table_2 = table[(table.level_0_ancestor_refined==area_1) & (table.level<=1)].display_name.drop_duplicates(), table[(table.level_0_ancestor_refined==area_2) & (table.level<=1)].display_name.drop_duplicates()
                    target_concepts = table_1.values.tolist() + table_2.values.tolist()
                    probe_word_vectors = list(itertools.product(table_1.values, table_2.values))
                    for word_a, word_b in probe_word_vectors:
                        word_seeds = list(set(target_concepts) - set([word_a, word_b])) # 在剩下的concepts中选取seed
                        for word_seed in word_seeds: 
                            inputs_all.append([word_a, word_b, word_seed])
                inputs_all_df = pd.DataFrame(inputs_all, columns=['concept_a','concept_b','concept_seed'])
                inputs_all_df.to_csv(concept_triplet_path, index=False)
            else: inputs_all_df = pd.read_csv(concept_triplet_path)
            ranges = np.array_split(np.arange(inputs_all_df.shape[0]), 15)
            inputs_all_df = inputs_all_df.loc[ranges[k_th]]
            print('processing for split range ', k_th,' : ', str(ranges[k_th][0]), ' to ', str(ranges[k_th][-1]))
            data = [] 
            for triplet in tqdm.tqdm(inputs_all_df.values.tolist()):
                title = '-'.join([triplet[0]+'('+ancestor_map[triplet[0]]+')', triplet[1]+'('+ancestor_map[triplet[1]]+')']) + '\n'
                G, accuracy = cls.analogy_inference_network(cls, triplet, model, probe_depth, ancestor_map_df, level_map, mode = 'similarity')
                data.append(triplet + accuracy)
                if show: 
                    from visualization import Visualizer
                    titles = [title + 'similarity analogy (depth='+ str(probe_depth)+') for ' + triplet[2] +\
                                '\nabsolute accuracy = ' + str(accuracy[0]) + ' / ' + str(accuracy[4]) + ' = ' + str(round(accuracy[0]/accuracy[4],4)) +\
                                '\nrelative accuracy = ' + str(accuracy[2]) + ' / ' + str(accuracy[4]) + ' = ' + str(round(accuracy[2]/accuracy[4],4)),
                            title + 'similarity analogy (depth='+ str(probe_depth)+') for ' + triplet[2]  +\
                                '\nmultiple ancestor accuracy = ' + str(accuracy[1]) + ' / ' + str(accuracy[4]) + ' = ' + str(round(accuracy[1]/accuracy[4],4)) +\
                                '\nrelative accuracy = ' + str(accuracy[3]) + ' / ' + str(accuracy[4]) + ' = ' + str(round(accuracy[3]/accuracy[4],4))]
                    Visualizer.draw_analogy_graph([G,G], fig_size=(16,8), layout='graphviz', title = titles)
            data_df = pd.DataFrame(data, columns = ['concept_a', 'concept_b', 'concept_seed', 'n_eeac', 'n_eemac', 'n_eearc', 'n_eemarc', 'n_edges'])
            data_df.to_csv(accuracy_file_path, index = False)
            print('file cashed')
            return G

    @classmethod
    def functional_axis_similarity(cls, dim = 24, level = 0, multi_level=True, discipline_groups = ['chemistry', 'biomedical']):
        model = cls.train_model(vec_dim = dim, level = level, multi_level = multi_level)
        embedding_vectors = model.wv.index_to_key
        all_concepts = pd.read_csv(op.join(OpenAlexData.concept_tree_path, 'All_concepts_with_ancestors.csv'))
        concepts_1_level_0_ancestor = all_concepts.loc[all_concepts.n_ancestors==1][['display_name','level_0_ancestor_refined']].drop_duplicates()
        area_center, area_group_center = dict(), dict()

        for discipline_group in discipline_groups:
            table_group = concepts_1_level_0_ancestor.loc[concepts_1_level_0_ancestor.level_0_ancestor_refined.isin(OpenAlexData.discipline_group[discipline_group])]
            area_group_vectors = [model.wv.get_vector(concept_name) for concept_name in set(table_group.display_name).intersection(embedding_vectors)]
            area_group_center[discipline_group] = np.mean(area_group_vectors, axis=0)
        axis_vector = area_group_center[discipline_groups[1]] - area_group_center[discipline_groups[0]]
        dfs,data_sim = [], []
        for area, table_area in concepts_1_level_0_ancestor.groupby('level_0_ancestor_refined'):
            data = [[area, 1 - distance.cosine(axis_vector, model.wv.get_vector(concept_name))] for concept_name in set(table_area.display_name).intersection(embedding_vectors)]
            data_sim.append(np.mean([1 - distance.cosine(axis_vector, model.wv.get_vector(concept_name)) for concept_name in set(table_area.display_name).intersection(embedding_vectors)]))
            df = pd.DataFrame(data, columns=['area','similarity'])
            dfs.append(df)
        dfs_all = pd.concat(np.array(dfs, dtype=object)[np.argsort(data_sim)], axis = 0, ignore_index = True)
        from visualization import Visualizer
        title = 'interdisciplinary distribution from ' + ' to '.join(discipline_groups)
        file_name = 'functional_axis_' + '_'.join(discipline_groups) 
        Visualizer.similarity_spectrum(dfs_all, attr_1 = 'area', attr_2 = 'similarity', title = title, file_name=file_name) # yet to upgrade

    @classmethod
    def ancestor_inclination(cls,concept_discipline = 'with_domain', model_pars = [24, 0, True, 5], level = 1):
        dim_vec, level_model, multi_level, top_n = model_pars
        model = cls.train_model(vec_dim = dim_vec, level = level_model, multi_level = multi_level)
        all_concepts = pd.read_csv(op.join(OpenAlexData.concept_tree_path, 'All_concepts_with_ancestors.csv'))
        all_concepts = all_concepts.loc[all_concepts.display_name.isin(model.wv.index_to_key)]
        concepts_0 = np.sort(all_concepts.level_0_ancestor.drop_duplicates().values).tolist()
        from visualization import Visualizer

        def calculate_similarity(concept_ancestors_table):
            concepts, ancestors = concept_ancestors_table.values.T
            data_similarity = np.array([[1 - distance.cosine(model.wv.get_vector(concept), model.wv.get_vector(area)) for area in concepts_0] for concept in concepts])
            ancestor_coordinates = np.concatenate([[(i,concepts_0.index(concept)) for concept in ancestors_i.split(',')] for i, ancestors_i in enumerate(ancestors)])
            non_ancestor_coordinates = np.array(list(set(map(lambda x:tuple(x), np.argwhere(np.ones(data_similarity.shape)))).difference(map(lambda x:tuple(x), ancestor_coordinates))))
            non_ancestor_coordinates = non_ancestor_coordinates[np.lexsort(non_ancestor_coordinates.T[::-1])].astype('int')
            ancestor_similarity, non_ancestor_similarity = data_similarity[tuple(ancestor_coordinates.T)], data_similarity[tuple(non_ancestor_coordinates.T)]
            return ancestor_similarity, non_ancestor_similarity
        
        if concept_discipline == 'interdiscipline':
            similarity_df_path, title = op.join(OpenAlexData.embedding_path, 'dists', 'ances_vs_nonances_similarity_df_all.csv'), 'ances_vs_nonances_similarity'
            if not op.exists(similarity_df_path):
                interdisciplines = all_concepts.loc[(all_concepts.n_max_ancestors>1)][['display_name','multiple_level_0_ancestors']].drop_duplicates()
                ancestor_similarity, non_ancestor_similarity = calculate_similarity(interdisciplines)

                non_interdisciplines = all_concepts.loc[all_concepts.n_max_ancestors==1][['display_name','level_0_ancestor_refined']].drop_duplicates()
                ancestor_similarity_non_iter, non_ancestor_similarity_non_iter = calculate_similarity(non_interdisciplines)

                df_1 = pd.DataFrame([['inter-ancestor',simi] for simi in ancestor_similarity], columns=['type','similarity'])
                df_2 = pd.DataFrame([['inter-non-ancestor',simi] for simi in non_ancestor_similarity], columns=['type','similarity'])
                df_3 = pd.DataFrame([['mono-ancestor',simi] for simi in ancestor_similarity_non_iter], columns=['type','similarity'])
                df_4 = pd.DataFrame([['mono-non-ancestor',simi] for simi in non_ancestor_similarity_non_iter], columns=['type','similarity'])
                similarity_df = pd.concat([df_1, df_2, df_3, df_4]).reset_index()
                similarity_df.to_csv(similarity_df_path, index=False)
            else: similarity_df = pd.read_csv(similarity_df_path)
            Visualizer.similarity_pdf_ances_vs_nonances(similarity_df, title = title, save=True)
        elif (concept_discipline in concepts_0) or (concept_discipline == 'All disciplines'):
            def get_concept_similaritys(level = level):
                sub_concepts = all_concepts.loc[(all_concepts.level==level) & (all_concepts.n_ancestors==1) & (all_concepts.display_name.isin(model.wv.index_to_key))]
                sub_concepts = sub_concepts.loc[sub_concepts.display_name.drop_duplicates().index]
                concept_names = sub_concepts.display_name.values
                sim_mapping = [[1 - distance.cosine(model.wv.get_vector(area), model.wv.get_vector(concept_name)) for area in concepts_0] for concept_name in concept_names]
                df = pd.DataFrame(sim_mapping, columns = concepts_0, index = concept_names)
                df['ancestor'] = sub_concepts.display_name.map(dict(sub_concepts[['display_name','level_0_ancestor_refined']].values)).values
                return df
            similarity_df_1 = get_concept_similaritys(level = 1) 
            similarity_df_2 = get_concept_similaritys(level = 2) 
            df_2_mean = np.mean(similarity_df_2.drop(['ancestor'], axis=1).values, axis=0) 
            df_2_area_inclination_quantile_small = np.quantile(similarity_df_2.drop(['ancestor'], axis=1).values, 0.05, axis=0) 
            df_2_area_inclination_quantile_large = np.quantile(similarity_df_2.drop(['ancestor'], axis=1).values, 0.95, axis=0) 
            similarity_statistic_df_2 = pd.DataFrame(np.vstack([df_2_area_inclination_quantile_small, df_2_mean, df_2_area_inclination_quantile_large]), columns=similarity_df_2.columns[:-1], index=['q=0.05','mean','q=0.95'])
            # 
            similarity_statistic_df_2['ancestor'] = 'All disciplines'
            similarity_df_2 = pd.concat([similarity_df_2, similarity_statistic_df_2])
            candidate_areas = concepts_0 if concept_discipline == 'All disciplines' else [concept_discipline]
            sigle_label_ = True if concept_discipline == 'All disciplines' else False
            for concept_discipline_ in candidate_areas:
                print('Generating radar map for discipline: {}'.format(concept_discipline_))
                Visualizer.mapping_radar([similarity_df_1, similarity_df_2], area = concept_discipline_, save=True, sigle_label = sigle_label_)
        else: raise ValueError('Undefined concept type')

    @classmethod
    def concepts_reachable_route_search(cls, model=None, concepts = ['Mathematics', 'Natural language processing'], model_pars = [24, 0, True, 5], mode = 'cosine_similarity'):
        # mode = cosine_similarity
        dim, level, multi_level, top_n = model_pars
        if model==None: model = cls.train_model(vec_dim = dim, level = level, multi_level = multi_level)
        edges, marker = [], np.full(len(model.wv), False)
        embedding_words = np.array(model.wv.index_to_key)
        start, end = concepts
        while start != end:
            marked_words = embedding_words[marker]
            candidates = [concept[0] for concept in model.wv.most_similar(start, topn=top_n) if concept[0] not in marked_words]
            similarities = [1 - distance.cosine(model.wv.get_vector(end)-model.wv.get_vector(start), model.wv.get_vector(candidate)-model.wv.get_vector(start)) for candidate in candidates]
            next_node = candidates[np.argmax(similarities)]
            marker[model.wv.key_to_index[start]] = True
            edge = [start, next_node]
            edges.append(edge)
            start = next_node
        return edges

    @classmethod
    def concept_dists(cls, model_pars = [24, 0, True], k_th = 0, combine = False, mode = 'cosine', return_dist_df = False):
        external_data_dir = op.join(OpenAlexData.embedding_path, 'dists') if is_in_server else OpenAlexData.other_data_path
        dists_path = op.join(external_data_dir, 'concepts_dists_'+ mode +'.npy')
        if not op.exists(dists_path):
            if combine:
                dists_all = []
                for i in range(20):
                    file_path = op.join(OpenAlexData.embedding_path, 'dists', 'concepts_dists_'+ mode +'_k_'+ str(i) +'.npy')
                    dists_all.append(np.load(file_path))
                np.save(dists_path, np.concatenate(dists_all, axis=0))
            else:
                file_path = op.join(OpenAlexData.embedding_path, 'dists', 'concepts_dists_'+ mode +'_k_'+ str(k_th) +'.npy')
                print(file_path)
                dim, level, multi_level = model_pars
                model = cls.train_model(vec_dim = dim, level = level, multi_level = multi_level)
                vecs = model.wv.vectors
                vecs_ranges = np.array_split(np.arange(len(vecs)), 20)
                if mode == 'euclidean':
                    dists_k = [[np.linalg.norm(a-b) for a in vecs] for b in vecs[vecs_ranges[k_th]]] # euclidean distance
                elif mode == 'cosine':
                    dists_k = [[distance.cdist(a, b, mode) for a in vecs] for b in vecs[vecs_ranges[k_th]]] # cosine distance
                else: raise ValueError('distance mode undefined')
                np.save(file_path, np.array(dists_k))
                print('cashed')
        else: dists = np.load(dists_path)
        dists_triu = np.triu(dists, k = 1)
        if return_dist_df:
            import vaex
            dist_df = vaex.from_pandas(pd.DataFrame(dists_triu[np.nonzero(dists_triu)], columns=['dist']))
            quantile_path = op.join(OpenAlexData.embedding_path, 'dists', 'concept_dists_'+ mode +'_quantiles.csv')
            if not op.exists(quantile_path):
                quantiles_1 = [[np.round(i/100, 4), np.round(dist_df.percentile_approx(dist_df.dist, percentage=i),4)] for i in np.arange(0.05,5,0.5)]
                quantiles_2 = [[i/100, np.round(dist_df.percentile_approx(dist_df.dist, percentage=i),4)] for i in range(5,100,5)]
                quantiles = quantiles_1 + quantiles_2
                quantiles_df = pd.DataFrame(quantiles, columns=['ratio','quantile'])
                quantiles_df.to_csv(quantile_path, index=False)
        return (dists, dist_df) if return_dist_df else dists

    @classmethod
    def search_central_popular_concepts(cls, model_pars = [24, 0, True, 5], quantile_ratio = 0.0105, k_th = 0, mode='cosine'):
        dim, level, multi_level, top_n = model_pars
        # all_concepts = pd.read_csv(op.join(OpenAlexData.concept_tree_path, 'All_concepts_with_ancestors.csv'))
        dist_k_path = op.join(OpenAlexData.embedding_path, 'dists', 'concept_average_dist_k_'+ str(k_th) +'.csv')
        print(dist_k_path)
        quantile_path = op.join(OpenAlexData.embedding_path, 'dists', 'concept_dists_'+ mode +'_quantiles.csv')
        quantiles = dict(pd.read_csv(quantile_path).values)
        model = cls.train_model(vec_dim = dim, level = level, multi_level = multi_level)
        dists = np.load(op.join(OpenAlexData.embedding_path, 'dists', 'concepts_dists.npy'))
        neighbor_dists, unrachable_pairs, isolate_words = [], [], []
        concept_with_vecs = np.array(model.wv.index_to_key)
        index_ranges = np.array_split(np.arange(len(concept_with_vecs)), 10)
        for index, concept in zip(index_ranges[k_th], concept_with_vecs[index_ranges[k_th]]):
            contacts = concept_with_vecs[np.argwhere(dists[index] <= quantiles[quantile_ratio]).flatten()]
            edge_lens = []
            for contact in contacts:
                try: edges = cls.concepts_reachable_route_search(model, concepts = [concept, contact], model_pars = model_pars)
                except:
                    try: edges = cls.concepts_reachable_route_search(model, concepts = [concept, contact], model_pars = [dim, level, multi_level, 2*top_n])
                    except:
                        unrachable_pairs.append([concept, contact])
                        continue
                edge_lens.append(len(edges))
            if len(edge_lens)>0: neighbor_dists.append([concept, np.round(np.mean(edge_lens), 5)])
            else: isolate_words.append(concept)
        if len(isolate_words)>0:
            df_isolate_words = pd.DataFrame(isolate_words, columns=['concept_isolate'])
            df_isolate_words.to_csv(op.join(OpenAlexData.embedding_path, 'dists', 'isolate_concepts_k_'+ str(k_th) +'.csv'))
        if len(unrachable_pairs)>0:
            df_unrachable_pairs = pd.DataFrame(unrachable_pairs, columns=['concept_s','concept_e'])
            df_unrachable_pairs.to_csv(op.join(OpenAlexData.embedding_path, 'dists', 'unrachable_pairs_k_'+ str(k_th) +'.csv'))
        if len(neighbor_dists)>0:
            df_neighbor_dists = pd.DataFrame(neighbor_dists, columns = ['concept','ave_dist'])
            df_neighbor_dists.to_csv(dist_k_path, index = False)
        print('cached')

    @classmethod
    def all_pairs_shortest_length(cls, ajacency_matrix, k_th = 0, mat_type = 'path_weight_mat', weight_split = False):
        m, n = ajacency_matrix.shape
        row_s, row_e = (0, m) if k_th < 0 else (cls.equal_row_partition(m)[k_th])
        import igraph as ig
        G = ig.Graph.Weighted_Adjacency(ajacency_matrix, mode = 'undirected')
        if not weight_split:
            del ajacency_matrix
            gc.collect()
        print('matrix type: ', mat_type)
        if mat_type == 'step_size_mat':
            print('k_th ',k_th, ' row index ',row_s, ' : ', row_e)
            time_s = datetime.datetime.now()
            step_size_mat = np.zeros((row_e - row_s, n), dtype=int) 
            for v in tqdm.trange(row_s, row_e):
                paths_rest = ig.GraphBase.get_all_shortest_paths(G, v=v, to = range(v+1, n), weights=G.es["weight"], mode="out")
                step_size_mat[v-row_s, v+1:] = pd.DataFrame([[path[-1],len(path)-1] for path in paths_rest], columns=['target','length']).drop_duplicates(ignore_index=True).groupby('target', group_keys=True).apply(min).length
            time_e = datetime.datetime.now()
            print(mat_type, 'computation time', time_e - time_s)
            return step_size_mat
        elif mat_type == 'path_weight_mat':
            print('start calculating')
            time_s = datetime.datetime.now()
            print(time_s)
            if weight_split:
                path_weight = lambda path: ajacency_matrix[path[:-1],path[1:]].sum()
                path_weight_mat = np.zeros((row_e - row_s, n), dtype=float) 
                for v in tqdm.trange(row_s, row_e):
                    paths_rest = ig.GraphBase.get_all_shortest_paths(G, v=v, to = range(v+1, n), weights=G.es["weight"], mode="out")
                    path_weight_mat[v-row_s, v+1:] = pd.DataFrame([[path[-1],path_weight(path)] for path in paths_rest], columns=['target','weight_sum']).drop_duplicates().weight_sum
            else: path_weight_mat = np.array(ig.GraphBase.distances(G, weights=G.es["weight"])) # 2w nodes, 1 day, 20:38:10.434525, 160690.434525 secs
            time_e = datetime.datetime.now()
            print(mat_type, 'computation time', time_e - time_s)
            return path_weight_mat
        else: raise ValueError('undefined matrix type')
    
    def all_pairs_shortest_length_iterative(ajacency_matrix):
        path_len_mat_path = op.join(OpenAlexData.embedding_path, 'dists', 'shortest_path_len_mat.npy')
        if not op.exists(path_len_mat_path):
            m = ajacency_matrix.shape[0]
            path_len_mat = np.ones(ajacency_matrix.shape, int)
            np.fill_diagonal(path_len_mat, 0)
            for i in tqdm.trange(m):
                time_s = datetime.datetime.now()
                for j in tqdm.trange(m):
                    for k in range(m):
                        if ajacency_matrix[i,j] > ajacency_matrix[i,k] + ajacency_matrix[k,j]:
                            ajacency_matrix[i,j] = ajacency_matrix[i,k] + ajacency_matrix[k,j]
                            marker = k
                    if marker > 0: path_len_mat[i,j] = path_len_mat[i,marker] + path_len_mat[marker,j]
                    marker = -1
                time_e = datetime.datetime.now() 
                print(time_e - time_s)
            np.save(path_len_mat_path, path_len_mat)
        else: path_len_mat = np.load(path_len_mat_path)

    @classmethod
    def equal_row_partition(cls, n, split_size = 34): 
        total_sub = n * (n-1) / 2 / split_size
        sub_s, items = 0, np.arange(n,0,-1)-1
        start_ends = []
        for i in range(1,n):
            if len(start_ends) <= split_size - 2:
                if np.sum(items[sub_s:i]) >= total_sub:
                    sums, index = [abs(total_sub - np.sum(items[sub_s:i-1])), abs(np.sum(items[sub_s:i]) - total_sub)], [i-1, i]
                    sub_e = index[np.argmin(sums)]
                    start_ends.append([sub_s, sub_e])
                    sub_s = sub_e
                    if len(start_ends) == split_size - 1:
                        start_ends.append([sub_s, n])
                        break
        sub_s, sub_e = start_ends[-1]
        sub_mid = int(np.sum(start_ends[-1])/2)
        start_ends = start_ends[:split_size-1] + [[sub_s, sub_mid],[sub_mid, sub_e]]
        return start_ends
    
    @classmethod
    def all_pairs_path_length_mat_verify(cls, model_pars = [24, 0, True, 5], route_s_e = [], route_s_e_id = [], path = [], path_len_gt_5 = True):
        def print_route(path_sub_areas_df):
            path_weight_mat = cls.all_pairs_path_length_mat(subset = 2, mat_type= 'path_weight_mat')
            all_pair_dists_cosine = np.load(op.join(cls.external_data_dir, 'concepts_dists_cosine_works_count_top_2w.npy'))
            sub_area_shortest_paths = eval(path_sub_areas_df.path.values[0])
            path_s_id, path_e_id = path_sub_areas_df['path_s'].values[0], path_sub_areas_df['path_e'].values[0]
            cosine_distance, shortest_distance= all_pair_dists_cosine[path_s_id, path_e_id], path_weight_mat[path_s_id, path_e_id]
            shortest_path = sub_area_shortest_paths[0]
            shortest_from_path = all_pair_dists_cosine[tuple([shortest_path[:-1],shortest_path[1:]])].sum()
            print(sub_concepts[shortest_path].values.tolist())
            print(all_pair_dists_cosine[tuple([shortest_path[:-1],shortest_path[1:]])],shortest_from_path,shortest_distance,cosine_distance)
            return [shortest_from_path,shortest_distance,cosine_distance]
        
        dim, level, multi_level, top_n = model_pars
        model = cls.train_model(vec_dim = dim, level = level, multi_level = multi_level)
        all_concepts = pd.read_csv(op.join(OpenAlexData.concept_tree_path, 'All_concepts_with_ancestors.csv'))
        sub_concepts_table = all_concepts.loc[(all_concepts.works_count>=3433) & (all_concepts.display_name.isin(model.wv.index_to_key))].drop_duplicates(ignore_index = True)
        sub_concepts = sub_concepts_table.display_name.drop_duplicates()
        sub_concepts.index=range(20001)
        
        if path_len_gt_5:
            all_paths_gt_5 = pd.read_csv(op.join(cls.external_data_dir, 'path_df_gt_5.csv'))
            if len(route_s_e_id) == 2:
                path_sub_area_df = all_paths_gt_5.loc[(all_paths_gt_5.path_s==route_s_e_id[0]) & (all_paths_gt_5.path_e==route_s_e_id[1])]
            elif len(route_s_e) == 2:
                level_map = dict(sub_concepts_table[['display_name','level']].drop_duplicates().values)
                ancestor_map = dict(all_concepts.loc[all_concepts.display_name.isin(sub_concepts)][['display_name','level_0_ancestor_refined']].drop_duplicates().values)
                # all_paths = pd.read_csv(op.join(cls.embedding_path, 'dists', 'path_df_all.csv'))
                all_paths_gt_5['path_s_concept'] = all_paths_gt_5.path_s.map(sub_concepts)
                all_paths_gt_5['path_s_level'] = all_paths_gt_5.path_s_concept.map(level_map)
                all_paths_gt_5['path_s_ancestor'] = all_paths_gt_5.path_s_concept.map(ancestor_map)
                all_paths_gt_5['path_e_concept'] = all_paths_gt_5.path_e.map(sub_concepts)
                all_paths_gt_5['path_e_level'] = all_paths_gt_5.path_e_concept.map(level_map)
                all_paths_gt_5['path_e_ancestor'] = all_paths_gt_5.path_e_concept.map(ancestor_map)
                path_sub_area_df = all_paths_gt_5.loc[((all_paths_gt_5.path_s_concept==route_s_e[0]) & (all_paths_gt_5.path_e_concept==route_s_e[1])) | ((all_paths_gt_5.path_s_concept==route_s_e[1]) & (all_paths_gt_5.path_e_concept==route_s_e[0]))]
            else: raise ValueError('No path information')
        else:
            df_paths_interested = pd.read_csv(op.join(cls.external_data_dir, 'path_df_interested_with_ids.csv'))
            path_sub_area_df = df_paths_interested.loc[((df_paths_interested.path_s_concept==route_s_e[0]) & (df_paths_interested.path_e_concept==route_s_e[1])) | ((df_paths_interested.path_s_concept==route_s_e[1]) & (df_paths_interested.path_e_concept==route_s_e[0]))]
        distances = print_route(path_sub_area_df)
        
        return distances, eval(path_sub_area_df.path.values[0])

    @classmethod
    def all_pairs_path_length_mat(cls, k_th = -1, model_pars = [24, 0, True, 5], test_N = -1, subset = 2, compressed = False, mat_type = 'path_weight_mat'):
        # https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
        # mat_type = 'step_size_mat' # or 'path_weight_mat'
        file_extension = '.npz' if compressed else '.npy'
        if test_N > 0: file_extension = '_test_N_' + str(test_N) + file_extension
        if k_th < 0 and test_N < 0: # default k_th = -1, means the whole matrix, not sub
            step_size_mat_whole = []
            if subset == 0: file_name = op.join(OpenAlexData.embedding_path, 'dists', mat_type + '_all')
            elif subset == 1: mat_file_name = op.join(OpenAlexData.embedding_path, 'dists', 'shortest_length_mat_only_1_ancestor')
            elif subset == 2: mat_file_name = op.join(cls.external_data_dir, mat_type+'_works_count_top_2w')
            else: raise ValueError('undefined subset type')
            target_matrix_with_mode_path = mat_file_name + file_extension
            if not op.exists(target_matrix_with_mode_path):
                for k_th in range(35):
                    if subset == 1: file_name = op.join(OpenAlexData.embedding_path, 'dists', 'shortest_length_mat_only_1_ancestor_k_' + str(k_th))
                    elif subset == 2: file_name = op.join(OpenAlexData.embedding_path, 'dists', mat_type+'_works_count_top_2w_k_' + str(k_th))
                    mat_path_k = file_name + file_extension
                    if compressed: step_size_mat_whole.append(np.load(mat_path_k)['step_size_mat'])
                    else: step_size_mat_whole.append(np.load(mat_path_k))
                target_matrix_with_mode = np.concatenate(step_size_mat_whole, axis=0)
                np.save(target_matrix_with_mode_path, target_matrix_with_mode)
            else: target_matrix_with_mode = np.load(target_matrix_with_mode_path)
        else: 
            if k_th < 0: raise ValueError('k_th must be greater than 0')
            if test_N > 0: file_name = op.join(OpenAlexData.embedding_path, 'dists', mat_type)
            elif subset == 0: file_name = op.join(OpenAlexData.embedding_path, 'dists', mat_type + '_all')
            elif subset == 1: file_name = op.join(OpenAlexData.embedding_path, 'dists', 'shortest_length_mat_only_1_ancestor_k_' + str(k_th))
            elif subset == 2: file_name = op.join(OpenAlexData.embedding_path, 'dists', mat_type + ('' if test_N>0 else '_works_count_top_2w_k_' + str(k_th)))
            else: raise ValueError('undefined subset value')
            mat_path_k = file_name + file_extension
            print(mat_path_k)
            if not op.exists(mat_path_k):
                if test_N > 0:
                    b = np.random.random_integers(1,10,size=(test_N, test_N))
                    dists_adj = (b + b.T)/2
                    np.fill_diagonal(dists_adj, 0)
                else:
                    dim, level, multi_level, top_n = model_pars
                    if subset == 0: dists_adj = cls.concept_dists(mode='cosine')
                    elif subset == 1:
                        concept_dists_level_0_ancestor_path = op.join(OpenAlexData.external_data_dir, 'concepts_dists_cosine_level_0_ancestor.npy')
                        if not op.exists(concept_dists_level_0_ancestor_path):
                            model = cls.train_model(vec_dim = dim, level = level, multi_level = multi_level)
                            all_concepts = pd.read_csv(op.join(OpenAlexData.concept_tree_path, 'All_concepts_with_ancestors.csv'))
                            concepts_1_level_0_ancestor = all_concepts.loc[(all_concepts.n_ancestors==1) & (all_concepts.display_name.isin(model.wv.index_to_key))][['display_name','level_0_ancestor']].drop_duplicates()
                            concepts_targets = concepts_1_level_0_ancestor['display_name'].drop_duplicates().values
                            index_targets = [model.wv.key_to_index[concept] for concept in concepts_targets]
                            dists_adj = cls.concept_dists(mode='cosine')
                            dists_adj = dists_adj[np.ix_(index_targets, index_targets)]
                            gc.collect()
                            np.save(concept_dists_level_0_ancestor_path, dists_adj)
                        else: dists_adj = np.load(concept_dists_level_0_ancestor_path)
                    elif subset == 2: 
                        concept_dists_works_count_top_2w = op.join(OpenAlexData.external_data_dir, 'concepts_dists_cosine_works_count_top_2w.npy')
                        if not op.exists(concept_dists_works_count_top_2w):
                            model = cls.train_model(vec_dim = dim, level = level, multi_level = multi_level)
                            all_concepts = pd.read_csv(op.join(OpenAlexData.concept_tree_path, 'All_concepts_with_ancestor_work_counts.csv'))
                            sub_concepts = all_concepts.loc[(all_concepts.works_count>=3433) & (all_concepts.display_name.isin(model.wv.index_to_key))][['display_name']].drop_duplicates()
                            sub_concepts.to_csv(op.join(OpenAlexData.concept_tree_path, 'concepts_works_count_top_2w.csv'), index = False)
                            index_targets = [model.wv.key_to_index[concept] for concept in sub_concepts.display_name]
                            dists_adj = cls.concept_dists(mode='cosine')
                            dists_adj = dists_adj[np.ix_(index_targets, index_targets)]
                            gc.collect()
                            np.save(concept_dists_works_count_top_2w, dists_adj)
                        else: dists_adj = np.load(concept_dists_works_count_top_2w)
                    else: raise ValueError('undefined subset value')
                target_matrix_with_mode = cls.all_pairs_shortest_length(dists_adj, k_th = k_th, mat_type = mat_type) # mode: 0 'step_size', 1 'weight'
                del dists_adj
                gc.collect()
                if compressed: np.savez(mat_path_k, step_size_mat = target_matrix_with_mode)
                else: np.save(mat_path_k, target_matrix_with_mode)
                print('cached')
            else: 
                if compressed: 
                    target_matrix_with_mode = np.load(mat_path_k)['step_size_mat']
                else: target_matrix_with_mode = np.load(mat_path_k)
        return target_matrix_with_mode

    @classmethod
    def shortest_path_length_distribution(cls, start_area = None, target_area = None, target_split = False, model_pars = [24, 0, True, 5], subset = 2, mat_type= 'step_size_mat', curve = 'kde', bw_adjust = 1, save = False, with_multiple_ancestor = False, return_accessibility = False):
        dim, level, multi_level, top_n = model_pars
        target_matrix_with_mode = cls.all_pairs_path_length_mat(subset = subset, mat_type= mat_type)
        all_concepts = pd.read_csv(op.join(OpenAlexData.concept_tree_path, 'All_concepts_with_ancestors.csv'))
        model = cls.train_model(vec_dim = dim, level = level, multi_level = multi_level)
        if subset == 0:
            sub_concepts = all_concepts
        elif subset == 1:
            sub_concepts = all_concepts.loc[(all_concepts.n_ancestors==1) & (all_concepts.display_name.isin(model.wv.index_to_key))][['display_name','level_0_ancestor_refined', 'multiple_level_0_ancestors']].drop_duplicates(ignore_index = True)
        elif subset ==2:
            sub_concepts = all_concepts.loc[(all_concepts.works_count>=3433) & (all_concepts.display_name.isin(model.wv.index_to_key))][['display_name', 'level_0_ancestor_refined', 'multiple_level_0_ancestors']].drop_duplicates(ignore_index = True)
        else: raise ValueError('undefined subset')
        from visualization import Visualizer
        concepts_non_level_0, level_0_ancestor_indes = np.load(op.join(OpenAlexData.concept_tree_path, 'n_level_0_ancestors.npy'), allow_pickle = True)
        areas = np.sort(all_concepts[['level_0_ancestor']].drop_duplicates(ignore_index = True).level_0_ancestor.values)
        if start_area == None: 
            df_path = op.join(OpenAlexData.embedding_path, 'dists','areas_dists_all_mutiple_ancestor.csv')
            if not op.exists(df_path):
                time_s = datetime.datetime.now()
                areas_dists, area_dists_mean = [], []
                for i in range(len(areas)):
                    area_descendant_concepts = np.append(concepts_non_level_0[np.nonzero(level_0_ancestor_indes[i])], areas[i])
                    targets_mat_index = sub_concepts.loc[sub_concepts.display_name.isin(area_descendant_concepts)].index.tolist()
                    area_dists = target_matrix_with_mode[targets_mat_index]
                    area_dists_list = area_dists[np.nonzero(area_dists)]
                    areas_dists.append(area_dists_list)
                    area_dists_mean.append(area_dists_list.mean())
                indexs, areas_all = np.argsort(area_dists_mean), []
                areas_sorted, areas_dists_all = areas[indexs], np.array(areas_dists, dtype = object)[indexs]
                dists_len = [len(areas_dists_all[i]) for i in range(len(areas_dists_all))]
                for i in range(len(dists_len)): areas_all += dists_len[i] * [areas_sorted[i]]
                areas_dists_all, areas_all = np.concatenate(areas_dists_all), np.array(areas_all)
                data = np.vstack((areas_all, areas_dists_all)).T
                data_df = pd.DataFrame(data, columns=['area', 'distance'])
                data_df['distance'] = data_df['distance'].astype(float)
                data_df.to_csv(df_path, index=False)
                time_e = datetime.datetime.now()
                print('computation time', time_e - time_s)
            else: 
                time_s = datetime.datetime.now()
                data_df = pd.read_csv(df_path)
                time_e = datetime.datetime.now()
                print('data loading time', time_e - time_s)
            Visualizer.multiple_pdf_plot(data_df, curve = curve, bw_adjust = bw_adjust)
        else: 
            if with_multiple_ancestor:
                start_mat_index = sub_concepts.loc[sub_concepts.multiple_level_0_ancestors.map(lambda s:start_area in s)].index.tolist()
            else: start_mat_index = sub_concepts.loc[sub_concepts.level_0_ancestor_refined==start_area].index.tolist() 
            area_dists = target_matrix_with_mode[start_mat_index]
            if target_area == None:
                if target_split: 
                    area_dist_dfs, dist_mean = [], []
                    file_path = op.join(OpenAlexData.external_data_dir, 'Shortest_distances_'+start_area + '_to_others_'+ ('multiple' if with_multiple_ancestor else 'single') +'_ancestor.csv')
                    if not op.exists(file_path):
                        for area in areas:
                            if with_multiple_ancestor:
                                targets_mat_index = sub_concepts.loc[sub_concepts.multiple_level_0_ancestors.map(lambda s:area in s)].index.tolist()
                            else: targets_mat_index = sub_concepts.loc[sub_concepts.level_0_ancestor_refined==area].index.tolist()
                            temp_mat = area_dists[:,targets_mat_index]
                            area_dists_target = temp_mat[np.nonzero(temp_mat)]
                            dist_mean.append(area_dists_target.mean())
                            area_dist_df_i = pd.DataFrame(area_dists_target, columns=['distance'])
                            area_dist_df_i['target_area'] = area
                            area_dist_dfs.append(area_dist_df_i)
                        area_dists_df = pd.concat(np.array(area_dist_dfs, dtype = object)[np.argsort(dist_mean)], axis = 0)
                        area_dists_df.to_csv(file_path, index = False)
                    else: 
                        time_s = datetime.datetime.now()
                        area_dists_df = pd.read_csv(file_path)
                        time_e = datetime.datetime.now()
                        print('data loading time', time_e - time_s)
                    file_name = ('accessibility' if return_accessibility else 'shortest_dist')+'_box_'+start_area + '_to_others_'+ ('multiple' if with_multiple_ancestor else 'single') +'_ancestor'
                    print(file_name)
                    if return_accessibility: area_dists_df.distance = 1-area_dists_df.distance/2
                    _xlabel = ('accessibility' if return_accessibility else 'global distance') # + ' distribution from Mathematics'
                    Visualizer.mutiple_box_plot(area_dists_df, title = start_area, file_name = file_name, xlabel = _xlabel, save=save, fig_size = (4, 8))
                    # Visualizer.multiple_pdf_plot(area_dists_df, title = start_area, curve = curve, bw_adjust = bw_adjust, file_name = file_name, save=save)
                    print('cached')
                else: 
                    print('shape', area_dists.shape, 'number of zero elements', np.sum(area_dists==0))
                    area_dists_list = area_dists[np.nonzero(area_dists)]
                    all_area_dists_list = target_matrix_with_mode[np.nonzero(target_matrix_with_mode)]
                    if mat_type == 'step_size_mat':
                        dist_df = pd.DataFrame(np.array(np.unique(area_dists_list, return_counts = True)).T, columns = ['step_len', 'counts'])
                        dist_df['ratio'] = np.round(dist_df.counts.values/dist_df.counts.sum(), 6)
                        dist_df['area'] = 'Mathematics'
                        dist_df.loc[5] = ['6-10', dist_df.counts[5:].sum(), dist_df.counts[5:].sum()/dist_df.counts.sum(), 'Mathematics']
                        dist_df_all = pd.DataFrame(np.array(np.unique(all_area_dists_list, return_counts = True)).T, columns = ['step_len', 'counts'])
                        dist_df_all['ratio'] = np.round(dist_df_all.counts.values/dist_df_all.counts.sum(), 6)
                        dist_df_all['area'] = 'All areas'
                        dist_df_all.loc[5] = ['6-10', dist_df_all.counts[5:].sum(), dist_df_all.counts[5:].sum()/dist_df_all.counts.sum(), 'All areas']
                        dist_df_combine = pd.concat([dist_df[:6], dist_df_all[:6]])
                        Visualizer.bar_plot(dist_df_combine, title = start_area)
                    elif mat_type == 'path_weight_mat':
                        Visualizer.pdf_plot(area_dists_list, title = start_area)
                    else: raise ValueError('undefined matrix type')
            else: pass 

    @classmethod
    def chord_calculation(cls, level = 0, area = None, model_pars = [24, 0, True, 5], subset = 2, mat_type = 'path_weight_mat', save = False):
        if level > 0 and area == None: raise ValueError('specify area when level > 0')
        all_concepts = pd.read_csv(op.join(OpenAlexData.concept_tree_path, 'All_concepts_with_ancestor_work_counts.csv'))
        dim, level_vec, multi_level, top_n = model_pars
        model = cls.train_model(vec_dim = dim, level = level_vec, multi_level = multi_level)
        target_matrix_with_mode = cls.all_pairs_path_length_mat(subset = subset, mat_type= mat_type)
        concepts_non_level_0, level_0_ancestor_indes = np.load(op.join(OpenAlexData.concept_tree_path, 'n_level_0_ancestors.npy'), allow_pickle = True)
        areas = all_concepts[['level_0_ancestor']].drop_duplicates(ignore_index = True).level_0_ancestor.values
        sub_concepts = all_concepts.loc[(all_concepts.works_count>=3433) & (all_concepts.display_name.isin(model.wv.index_to_key))][['display_name', 'level_0_ancestor']].drop_duplicates(ignore_index = True)
        areas.sort()
        if level == 0:
            matrix_df_path = op.join(OpenAlexData.embedding_path, 'dists', 'level_0_descendents_average_path_weight.csv')
            if not op.exists(matrix_df_path):
                area_descendants = []
                for i in range(len(areas)): 
                    descendents_area_i = np.append(concepts_non_level_0[np.nonzero(level_0_ancestor_indes[i])], areas[i])
                    targets_mat_index = sub_concepts.loc[sub_concepts.display_name.isin(descendents_area_i)].index.tolist()
                    area_descendants.append(targets_mat_index)
                average_weight = []
                for i in range(len(areas)):
                    average_weight_i = []
                    for j in range(len(areas)):
                        sub_matrix = target_matrix_with_mode[np.ix_(area_descendants[i], area_descendants[j])]
                        average_weight_i.append(sub_matrix[np.nonzero(sub_matrix)].mean())
                    average_weight.append(average_weight_i)
                matrix_df = pd.DataFrame(average_weight, index=areas, columns=areas)
                matrix_df.to_csv(matrix_df_path)
            else: matrix_df = pd.read_csv(matrix_df_path, index_col=0)
            from visualization import Visualizer
            Visualizer.chord_plot(matrix_df, save = save)
        elif level == 1: 
            from concept import Concept
            descendents_area = concepts_non_level_0[np.nonzero(level_0_ancestor_indes[np.argwhere(areas==area)[0][0]])]
            area_concepts_level = all_concepts.loc[(all_concepts.level==level) & (all_concepts.display_name.isin(descendents_area))].display_name.values
            area_concepts_level.sort()
            concept_tree = Concept.Tree(read_from_csv = True)
            concept_names_dict = nx.get_node_attributes(concept_tree, 'display_name')
            average_weight, concept_descendents = [], []
            for concept in area_concepts_level:
                area_concept_id = list(concept_names_dict.keys())[list(concept_names_dict.values()).index(concept)]
                descendents = [concept_names_dict[concept_i] for concept_i in nx.descendants(concept_tree, area_concept_id)] # descendent concepts of target level concept
                targets_mat_index = sub_concepts.loc[sub_concepts.display_name.isin(descendents)].index.tolist() # Corresponding to the correct index
                concept_descendents.append(targets_mat_index)
            for i in range(len(area_concepts_level)):
                average_weight_i = []
                for j in range(len(area_concepts_level)):
                    sub_matrix = target_matrix_with_mode[np.ix_(concept_descendents[i], concept_descendents[j])]
                    average_weight_i.append(sub_matrix[np.nonzero(sub_matrix)].mean())
                average_weight.append(average_weight_i)
            matrix_df = pd.DataFrame(average_weight, index=area_concepts_level, columns=area_concepts_level)
            from visualization import Visualizer
            Visualizer.chord_plot(matrix_df, save = save, level = level)
        else: raise ValueError('undefined level value')

    @classmethod
    def selected_pairs_shortest_length(cls, path_df):
        import igraph as ig
        concept_dists_works_count_top_2w = op.join(OpenAlexData.external_data_dir, 'concepts_dists_cosine_works_count_top_2w.npy')
        G = ig.Graph.Weighted_Adjacency(np.load(concept_dists_works_count_top_2w), mode = 'undirected') # 340G to construct all graph
        selected_paths = []
        for path_start, sub_table in tqdm.tqdm(path_df.groupby('path_s')):
            paths_rest = np.array(ig.GraphBase.get_all_shortest_paths(G, v=path_start, to = sub_table.path_e.values, weights=G.es["weight"], mode="out"), dtype=object)
            path_e_indexes = pd.DataFrame([path[-1] for path in paths_rest], columns=['path_e']).groupby('path_e').apply(lambda df:df.index.tolist())
            selected_paths += [paths_rest[path_e_indexes[path_e]].tolist() for path_e in sub_table.path_e]
        path_df['path'] = selected_paths
        return path_df
    
    @classmethod
    def all_pairs_shortest_path(cls, k_th = -1, n_concepts = 20001):
        all_pairs_path_df_path = op.join(OpenAlexData.external_data_dir, 'path_df_all.csv')
        if not op.exists(all_pairs_path_df_path):
            if k_th < 0:
                all_path_df = []
                for k_th in range(3):
                    all_path_df_path_k = op.join(OpenAlexData.external_data_dir, 'path_df_all_k_' + str(k_th) + '.csv')
                    all_path_df.append(pd.read_csv(all_path_df_path_k))
                path_df = pd.concat(all_path_df, axis = 0)
                path_df.to_csv(all_pairs_path_df_path, index=False)
            else:
                all_path_df_path_k = op.join(OpenAlexData.external_data_dir, 'path_df_all_k_' + str(k_th) + '.csv')
                all_indexes, selected_paths = np.arange(n_concepts), []
                row_indexs_rest = range(17146, n_concepts)
                row_indexs = np.array_split(row_indexs_rest, 3)[k_th]
                print(all_path_df_path_k, 'row index',row_indexs[0], ' : ', row_indexs[-1])
                import igraph as ig
                concept_dists_works_count_top_2w = op.join(OpenAlexData.external_data_dir, 'concepts_dists_cosine_works_count_top_2w.npy')
                G = ig.Graph.Weighted_Adjacency(np.load(concept_dists_works_count_top_2w), mode = 'undirected') # 340G to construct all graph
                for v in tqdm.tqdm(row_indexs):
                    paths_rest = np.array(ig.GraphBase.get_all_shortest_paths(G, v=v, to = np.delete(all_indexes, v), weights=G.es["weight"], mode="out"), dtype=object)
                    path_e_indexes = pd.DataFrame([path[-1] for path in paths_rest], columns=['path_e']).groupby('path_e').apply(lambda df:df.index.tolist())
                    selected_paths += [paths_rest[path_e_indexes[path_e]].tolist() for path_e in np.delete(all_indexes, v)]
                path_df = pd.DataFrame()
                path_df['path_s'] = np.repeat(row_indexs, n_concepts-1)
                path_df['path_e'] = np.concatenate([np.delete(all_indexes, v) for v in row_indexs])
                path_df['path'] = selected_paths
                path_df.to_csv(all_path_df_path_k, index = False)
                print('cached')
        else: path_df = pd.read_csv(all_pairs_path_df_path)
        return path_df

    @classmethod
    def concept_embedding_map(cls, k_th = -1, route_s_e = [], subset = 2, vec_dim = 24, mat_type= 'step_size_mat', path_len_gt = 5, figsize = (8, 8), all_dists = False, save = False, metric = 'euclidean', tsne_lib = 'sk', 
                              point_size = 2, label_refined = True, with_random_route = False, centrality_type = None, no_inter=False, emphasize_discipline = None, draw_route_region = False, path_len_gt_5 = True):
        if len(route_s_e) == 2 or with_random_route:
            if with_random_route: 
                df_type = 'path_df_all' if all_dists else 'path_df_gt_' + str(path_len_gt)
                path_df_path = op.join(OpenAlexData.external_data_dir, df_type + '.csv') # gt
                if not op.exists(path_df_path):
                    if k_th == -1:
                        path_dfs = []
                        for k_th in range(30):
                            path_df_path_k = op.join(OpenAlexData.embedding_path, 'dists', df_type + '_k_' + str(k_th) + '.csv')
                            print(path_df_path_k)
                            path_dfs.append(pd.read_csv(path_df_path_k))
                        path_df = pd.concat(path_dfs, axis = 0)
                        path_df.to_csv(path_df_path, index=False)
                    else:
                        path_df_path_k = op.join(OpenAlexData.embedding_path, 'dists', df_type + '_k_' + str(k_th) + '.csv')
                        print(path_df_path_k)
                        coordinates = np.argwhere(np.triu(cls.all_pairs_path_length_mat(subset = subset, mat_type= mat_type)) >= path_len_gt)
                        path_df_k = pd.DataFrame(coordinates, columns=['path_s','path_e'])
                        path_s_split = np.array_split(np.unique(coordinates.T[0]), 30)
                        path_df_k = path_df_k.loc[path_df_k.path_s.isin(path_s_split[k_th])] # split
                        del coordinates, path_s_split
                        gc.collect()
                        path_df_k = cls.selected_pairs_shortest_length(path_df_k)
                        path_df_k.to_csv(path_df_path_k, index=False)
                        print('cached')
                else: 
                    time_s = datetime.datetime.now()
                    path_df = pd.read_csv(path_df_path)
                    time_e = datetime.datetime.now()
                    print('data loading time', time_e - time_s)
                random_path = path_df.loc[np.random.randint(0,path_df.shape[0])]
                route_s_e_id, paths = random_path.values[:2].tolist(), eval(random_path.path)
                distances, paths_ = cls.all_pairs_path_length_mat_verify(route_s_e_id=route_s_e_id, path_len_gt_5 = path_len_gt_5)
            else: distances, paths = cls.all_pairs_path_length_mat_verify(route_s_e=route_s_e, path_len_gt_5 = path_len_gt_5) # 比较耗时
        else: paths, distances = [], []
        if isinstance(centrality_type, str):
            centrality_file_path = op.join(cls.embedding_path, 'dists', 'centrality_works_count_top_2w.npy')
            if not os.path.exists(centrality_file_path):
                shortest_path_weight_mat = Embedding.all_pairs_path_length_mat(subset = subset, mat_type= 'path_weight_mat')
                closeness_centrality = (shortest_path_weight_mat.shape[1]-1)/np.sum(shortest_path_weight_mat, axis=1)
                all_paths = pd.read_csv(op.join(cls.external_data_dir, 'path_df_all.csv'))
                betweenness_centrality = np.zeros(20001)
                for paths in tqdm.tqdm(all_paths.path):
                    paths = eval(paths)
                    n_sigma_s_t_vs, n_sigma_s_t = np.zeros(20001), len(paths)
                    for path in paths: n_sigma_s_t_vs[path[1:-1]] += 1
                    betweenness_centrality += n_sigma_s_t_vs / n_sigma_s_t
                centralities = np.vstack([closeness_centrality, betweenness_centrality])
                np.save(centrality_file_path, centralities)
            else: centralities = np.load(centrality_file_path)
            if centrality_type == 'closeness':
                centrality = centralities[0]
            elif centrality_type == 'betweenness':
                centrality = centralities[1]
            else: raise ValueError('Undefined centrality type')
        else: centrality = []
        from visualization import Visualizer
        Visualizer.concept_embedding_plot(subset = subset, model_pars = [vec_dim, 0, True, 5], metric = metric, level = 2, figsize = figsize, paths = paths, path_distance = distances, draw_route_region = draw_route_region,
                                           tsne_lib = tsne_lib, save = save, label_refined = label_refined, centrality = centrality, no_inter=no_inter, point_size = point_size, emphasize_discipline = emphasize_discipline)
        print(5)
    
    @classmethod
    def concept_path_embedding_map(cls, k_th = -1, route_s_e = [], subset = 2, vec_dim = 24, mat_type= 'step_size_mat', path_len_gt = 5, figsize = (8, 8), all_dists = False, save = False, metric = 'euclidean', tsne_lib = 'sk', 
                              point_size = 2, label_refined = True, with_random_route = False, centrality_type = None, no_inter=False, emphasize_discipline = None, draw_route_region = False, path_len_gt_5 = True):
        if len(route_s_e) == 2 or with_random_route:
            if with_random_route: 
                df_type = 'path_df_all' if all_dists else 'path_df_gt_' + str(path_len_gt)
                path_df_path = op.join(OpenAlexData.external_data_dir, df_type + '.csv') # gt
                if not op.exists(path_df_path):
                    if k_th == -1:
                        path_dfs = []
                        for k_th in range(30):
                            path_df_path_k = op.join(OpenAlexData.embedding_path, 'dists', df_type + '_k_' + str(k_th) + '.csv')
                            print(path_df_path_k)
                            path_dfs.append(pd.read_csv(path_df_path_k))
                        path_df = pd.concat(path_dfs, axis = 0)
                        path_df.to_csv(path_df_path, index=False)
                    else:
                        path_df_path_k = op.join(OpenAlexData.embedding_path, 'dists', df_type + '_k_' + str(k_th) + '.csv')
                        print(path_df_path_k)
                        coordinates = np.argwhere(np.triu(cls.all_pairs_path_length_mat(subset = subset, mat_type= mat_type)) >= path_len_gt)
                        path_df_k = pd.DataFrame(coordinates, columns=['path_s','path_e'])
                        path_s_split = np.array_split(np.unique(coordinates.T[0]), 30)
                        path_df_k = path_df_k.loc[path_df_k.path_s.isin(path_s_split[k_th])] # split
                        del coordinates, path_s_split
                        gc.collect()
                        path_df_k = cls.selected_pairs_shortest_length(path_df_k)
                        path_df_k.to_csv(path_df_path_k, index=False)
                        print('cached')
                else: 
                    time_s = datetime.datetime.now()
                    path_df = pd.read_csv(path_df_path)
                    time_e = datetime.datetime.now()
                    print('data loading time', time_e - time_s)
                random_path = path_df.loc[np.random.randint(0,path_df.shape[0])]
                route_s_e_id, paths = random_path.values[:2].tolist(), eval(random_path.path)
                distances, paths_ = cls.all_pairs_path_length_mat_verify(route_s_e_id=route_s_e_id, path_len_gt_5 = path_len_gt_5)
            else: distances, paths = cls.all_pairs_path_length_mat_verify(route_s_e=route_s_e, path_len_gt_5 = path_len_gt_5) # 比较耗时
        else: paths, distances = [], []
        if isinstance(centrality_type, str):
            centrality_file_path = op.join(cls.embedding_path, 'dists', 'centrality_works_count_top_2w.npy')
            if not os.path.exists(centrality_file_path):
                shortest_path_weight_mat = Embedding.all_pairs_path_length_mat(subset = subset, mat_type= 'path_weight_mat')
                closeness_centrality = (shortest_path_weight_mat.shape[1]-1)/np.sum(shortest_path_weight_mat, axis=1)
                all_paths = pd.read_csv(op.join(cls.external_data_dir, 'path_df_all.csv'))
                betweenness_centrality = np.zeros(20001)
                for paths in tqdm.tqdm(all_paths.path):
                    paths = eval(paths)
                    n_sigma_s_t_vs, n_sigma_s_t = np.zeros(20001), len(paths)
                    for path in paths: n_sigma_s_t_vs[path[1:-1]] += 1
                    betweenness_centrality += n_sigma_s_t_vs / n_sigma_s_t
                centralities = np.vstack([closeness_centrality, betweenness_centrality])
                np.save(centrality_file_path, centralities)
            else: centralities = np.load(centrality_file_path)
            if centrality_type == 'closeness':
                centrality = centralities[0]
            elif centrality_type == 'betweenness':
                centrality = centralities[1]
            else: raise ValueError('Undefined centrality type')
        else: centrality = []
        from visualization import Visualizer
        Visualizer.concept_path_embedding_plot(subset = subset, model_pars = [vec_dim, 0, True, 5], metric = metric, level = 2, figsize = figsize, paths = paths, path_distance = distances, draw_route_region = draw_route_region,
                                           tsne_lib = tsne_lib, save = save, label_refined = label_refined, centrality = centrality, no_inter=no_inter, point_size = point_size, emphasize_discipline = emphasize_discipline)
        print(5)

    @classmethod
    def centrality_map(cls, centrality, process_level = 3):
        if process_level == 0: return centrality
        from sklearn.preprocessing import quantile_transform
        centrality = np.log(centrality)
        if process_level == 1: return centrality
        centrality_min, centrality_max = centrality.min(), centrality.max()
        centrality = (centrality-centrality_min) / (centrality_max - centrality_min)
        if process_level == 2: return centrality
        centrality = quantile_transform(centrality.reshape(-1, 1), output_distribution='uniform').flatten()
        if process_level == 3: return centrality

    @classmethod
    def interdisciplinary_centrality_analysis(cls, model_pars = [24, 0, True, 5]):
        dim, level_vec, multi_level, top_n = model_pars
        model = Embedding.train_model(vec_dim = dim, level = level_vec, multi_level = multi_level)
        model_words = model.wv.index_to_key
        all_concepts = pd.read_csv(op.join(Process.concept_tree_path, 'All_concepts_with_ancestors.csv'))
        level_0_ancestors = np.sort(all_concepts.level_0_ancestor.drop_duplicates().values)
        duplicate_concept_table = all_concepts.loc[(all_concepts.works_count>=3433) & (all_concepts.display_name.isin(model_words))][['level', 'display_name', 'n_ancestors', 'n_max_ancestors', 'level_0_ancestor', 'level_0_ancestor_refined', 'multiple_level_0_ancestors']].drop_duplicates(ignore_index = True)
        table_subset_2 = all_concepts.loc[(all_concepts.works_count>=3433) & (all_concepts.display_name.isin(model_words))][['display_name', 'n_ancestors', 'n_max_ancestors', 'level_0_ancestor', 'level_0_ancestor_refined', 'multiple_level_0_ancestors']].drop_duplicates(ignore_index = True)
        centralities = np.load(op.join(cls.embedding_path, 'dists', 'centrality_works_count_top_2w.npy'))
        table_subset_2['closeness'] = cls.centrality_map(centralities[0], process_level = 0)
        table_subset_2['betweenness'] = cls.centrality_map(centralities[1], process_level = 0)
        table_subset_2['closeness_log'] = cls.centrality_map(centralities[0], process_level = 1)
        table_subset_2['betweenness_log'] = cls.centrality_map(centralities[1], process_level = 1)
        table_subset_2['closeness_normal'] = cls.centrality_map(centralities[0], process_level = 2)
        table_subset_2['betweenness_normal'] = cls.centrality_map(centralities[1], process_level = 2)
        table_subset_2['closeness_uniform'] = cls.centrality_map(centralities[0], process_level = 3)
        table_subset_2['betweenness_uniform'] = cls.centrality_map(centralities[1], process_level = 3)
        table_subset_2['level'] = table_subset_2.display_name.map(dict(duplicate_concept_table[['display_name', 'level']].values))

        def get_top_centrality_concepts(centrality_type = 'closeness', top = -1, bottom = -1, area = None):
            data_df = table_subset_2[['display_name', 'n_ancestors', 'n_max_ancestors', 'level', 'level_0_ancestor_refined', 'multiple_level_0_ancestors']]
            data_df.loc[range(data_df.shape[0]),'centrality'] = table_subset_2[centrality_type].values.tolist()
            data_df.loc[range(data_df.shape[0]),'centrality_size'] = (data_df.centrality >= np.quantile(data_df.centrality, 0.99)).map({True:'high',False:'low'})
            data_df.loc[range(data_df.shape[0]),'type'] = centrality_type
            data_df_sort = data_df.loc[data_df.centrality.argsort().values[::-1]]
            data_df_sort = data_df_sort.loc[data_df_sort.level_0_ancestor_refined != 'Undefined']
            if top>0 and bottom < 0: table = data_df_sort[:top]
            elif top > 0 and bottom > 0: raise ValueError('unsupported')
            elif top < 0 and bottom > 0: table = data_df_sort[:-bottom]
            else: table = data_df_sort
            if area != None:
                sub_0=table.loc[(table.multiple_level_0_ancestors.map(lambda s:area in s))].drop(['centrality_size','type'],axis=1)
                sub_1=table.loc[table.level_0_ancestor_refined==area]
                sub_2=table.loc[(table.multiple_level_0_ancestors.map(lambda s:area in s)) & (table.level_0_ancestor_refined!=area)]
                sub_3 = pd.concat([sub_1, sub_2]).drop(['centrality_size','type'],axis=1)
                if sub_0.shape == sub_3.shape:
                    return sub_3
                else: raise ValueError('shape unequal')
            return table
        
        data_df_1_sort_top200 = get_top_centrality_concepts(centrality_type = 'closeness', top = 200)
        data_df_1_sort_top500 = get_top_centrality_concepts(centrality_type = 'closeness', top = 500)
        data_df_1_sort_top1000 = get_top_centrality_concepts(centrality_type = 'closeness', top = 1000)
        data_df_1_sort_top1500 = get_top_centrality_concepts(centrality_type = 'closeness', top = 1500)
        data_df_1_sort_top2000 = get_top_centrality_concepts(centrality_type = 'closeness', top = 2000)

        data_df_2_sort_top200 = get_top_centrality_concepts(centrality_type = 'betweenness', top = 200)
        data_df_2_sort_top500 = get_top_centrality_concepts(centrality_type = 'betweenness', top = 500)
        data_df_2_sort_top1000 = get_top_centrality_concepts(centrality_type = 'betweenness', top = 1000)
        data_df_2_sort_top1500 = get_top_centrality_concepts(centrality_type = 'betweenness', top = 1500)
        data_df_2_sort_top2000 = get_top_centrality_concepts(centrality_type = 'betweenness', top = 2000)

        get_top_centrality_concepts(centrality_type = 'closeness', top = 200, area='Mathematics').to_csv(op.join(cls.external_data_dir, 'data_df_1_sort_top200_Mathematics.csv'))

        def get_ancestor_counts(table=[], multiple_ancestor = True, divide = False):
            if len(table) == 0: table = all_concepts
            if multiple_ancestor:
                if divide:
                    ancestors_ratio_count_df = pd.DataFrame([[area, 0] for area in level_0_ancestors], columns=['ancestor','counts'])
                    for ancestors in table.loc[table.level_0_ancestor_refined!='Undefined'].multiple_level_0_ancestors.map(lambda s: s.split(',')):
                        ancestors_ratio_count_df.loc[ancestors_ratio_count_df.ancestor.isin(ancestors),'counts'] += 1/len(ancestors)
                    ancestor_count_df = ancestors_ratio_count_df.loc[ancestors_ratio_count_df.counts.argsort().values[::-1]]
                else:
                    ancestors_count_df = pd.DataFrame([[area, table.multiple_level_0_ancestors.map(lambda s: area in s).sum()] for area in level_0_ancestors], columns=['ancestor', 'counts'])
                    ancestor_count_df = ancestors_count_df.loc[ancestors_count_df.counts.argsort().values[::-1]]
            else:
                ancestor_count_df=pd.DataFrame(np.unique(table.level_0_ancestor_refined, return_counts=True), index=['ancestor', 'counts']).T
                ancestor_count_df = ancestor_count_df.loc[ancestor_count_df.counts.argsort().values[::-1]]
            ancestor_count_df.index = range(ancestor_count_df.shape[0])
            return ancestor_count_df
        
        def get_top_centrality_concepts_ancestor_trend(centrality_type = 'closeness', top = 200, bottom = -1, multiple_ancestor = True, get_ratio = True):
            ancestors_count_dict = dict(get_ancestor_counts(multiple_ancestor = multiple_ancestor, divide = False).values)
            ances_top = sum(get_top_centrality_concepts(centrality_type, top, bottom).multiple_level_0_ancestors.map(lambda s:s.split(',')).values,[])
            second_column = 'counts_ratio' if get_ratio else 'counts'
            ances_ratio_df = pd.DataFrame(np.unique(ances_top, return_counts=True), index=['ancestor', second_column]).T
            if get_ratio: 
                for i in range(ances_ratio_df.shape[0]):
                    ances_ratio_df.loc[i, second_column] /= ancestors_count_dict[ances_ratio_df.loc[i, 'ancestor']]
            ances_ratio_df=ances_ratio_df.loc[ances_ratio_df[second_column].argsort()[::-1]]
            ances_ratio_df.index = range(1, ances_ratio_df.index.shape[0]+1)
            if get_ratio: ances_ratio_df.to_csv(op.join(cls.external_data_dir, centrality_type+'_multiple_ancestor_count.csv'))
            return ances_ratio_df

        ances_ratio_df = get_top_centrality_concepts_ancestor_trend(centrality_type = 'closeness', top = 200)
        
        data_df_1_sort = get_top_centrality_concepts(centrality_type = 'closeness')
        data_df_2_sort = get_top_centrality_concepts(centrality_type = 'betweenness')
        data_df_concat = pd.concat([data_df_1_sort_top2000, data_df_2_sort_top2000])
        data_df_concat.index = range(data_df_concat.shape[0])
        data_df_concat['ances_type']= data_df_concat.n_ancestors.map(dict([(i,'single' if i<=1 else 'multiple') for i in range(data_df_concat.n_ancestors.max()+1)]))
        data_df_concat['level_range']= data_df_concat.level.map(dict([(i,'0-1' if i<=1 else '2-5') for i in range(data_df_concat.level.max()+1)]))
        pd.DataFrame(np.unique(data_df_concat.loc[data_df_concat.type=='closeness'].ances_type, return_counts=True), index=['ances_type', 'count']).T
        pd.DataFrame(np.unique(data_df_concat.loc[data_df_concat.type=='betweenness'].ances_type, return_counts=True), index=['ances_type', 'count']).T
        from visualization import Visualizer
        Visualizer.corrolation_map(data_df_concat)

