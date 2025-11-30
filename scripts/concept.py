import networkx as nx
import os.path as op
import pandas as pd
from dataset import OpenAlexData
from database import DatabaseManager
import numpy as np
import os

class Path:
    filedir = '../../../data/files/'

class Concept(OpenAlexData):
    @classmethod
    def count_intersections(self, table_1, table_2, level_i, less_than=False):
        if less_than: table_1_level_i, table_2_level_i = table_1.loc[table_1.level<=level_i], table_2.loc[table_2.level<=level_i]
        else: table_1_level_i, table_2_level_i = table_1.loc[table_1.level==level_i], table_2.loc[table_2.level==level_i]
        concepts_1_l1, concepts_2_l1 = table_1_level_i.display_name.drop_duplicates(), table_2_level_i.display_name.drop_duplicates()
        return [concepts_1_l1.shape[0], concepts_2_l1.shape[0], len(set(concepts_1_l1).intersection(concepts_2_l1))]
    
    @classmethod
    def sub_intersection_df(cls, level_i, less_than=False, threshold=20, top = 10):
        area_intersection_df_path = op.join(cls.concept_tree_path, 'number_of_area_intersection_df.csv')
        if not op.exists(area_intersection_df_path): 
            table = cls.level_0_ancestors()
            concepts_0 = table.loc[table.level==0].display_name.drop_duplicates().values
            import itertools
            area_intersection = []
            for area_1, area_2 in itertools.combinations(concepts_0, 2):
                data_1_2 = [area_1, area_2]
                table_1, table_2 = cls.level_0_ancestors(table, area_1), cls.level_0_ancestors(table, area_2)
                data_1_2 += cls.count_intersections(table_1, table_2, 1)
                for level_i_in in range(2,6):
                    data_1_2 += cls.count_intersections(table_1, table_2, level_i_in)
                    data_1_2 += cls.count_intersections(table_1, table_2, level_i_in, less_than=True)
                area_intersection.append(data_1_2)
            columns = ['concept_1','concept_2','num_1_l1','num_2_l1','inter_l1']
            for level_i_in in range(2,6):
                columns += ['num_1_l' + str(level_i_in),'num_2_l' + str(level_i_in),'inter_l' + str(level_i_in)]
                columns += ['num_1_lt_l' + str(level_i_in),'num_2_lt_l' + str(level_i_in),'inter_lt_l' + str(level_i_in)]
            area_intersection_df = pd.DataFrame(area_intersection, columns=columns)
            area_intersection_df.to_csv(area_intersection_df_path, index=False)
        else: area_intersection_df = pd.read_csv(area_intersection_df_path)
        if level_i == 0 or (level_i == 1 and less_than): raise ValueError('undefined type for less than ',level_i)
        columns = ['num_1_lt_l' + str(level_i),'num_2_lt_l' + str(level_i),'inter_lt_l' + str(level_i)] if less_than else ['num_1_l' + str(level_i),'num_2_l' + str(level_i),'inter_l' + str(level_i)]
        sub_df = area_intersection_df.loc[(area_intersection_df[columns[0]]>=threshold)&(area_intersection_df[columns[1]]>=threshold)].sort_values(columns[2], ascending = False)
        return sub_df[['concept_1', 'concept_2']+columns][:top]

    @classmethod
    def Tree(self, G = nx.DiGraph(), subTree_concept_name = None, sub_graph_attribute_values = [], return_concept_table = False, same_area = True, read_from_csv = False, save = False, show = False, layout = 'graphviz', 
             fig_size=(20, 16), font_size = 18, margin =  [-0.1, 1.1, -0.1, 1.1], bbox_inches = 'tight', pad_inches = 0.1, x_scale_ratio = 0.8):
        path_concept_tree = op.join(self.filedir, 'Concept_Trees')
        if not op.exists(path_concept_tree): os.makedirs(path_concept_tree)
        all_concept_tree_file_path = op.join(path_concept_tree, 'Concept_Tree_All.gpickle')
        if len(G) == 0:
            if not op.exists(all_concept_tree_file_path) or read_from_csv:
                concepts_path = op.join(path_concept_tree, 'All_concepts.csv')
                concept_edges_path = op.join(path_concept_tree, 'All_concept_edges.csv')
                if not op.exists(concepts_path) or not op.exists(concept_edges_path):
                    table_concepts = self.read_table(table_name = ['concepts'])
                    edges = self.read_table(table_name = ['concepts_ancestors'], attributes_trim_enter = ['ancestor_id'])
                    table_concepts.to_csv(concepts_path, index=False)
                    edges.to_csv(concept_edges_path, index=False)
                else:
                    table_concepts = pd.read_csv(concepts_path)
                    edges = pd.read_csv(concept_edges_path)
                G.add_nodes_from(table_concepts['id'].values)
                for attribute in table_concepts.columns[1:]:
                    nx.set_node_attributes(G, dict(table_concepts[['id', attribute]].values), attribute)
                G.add_edges_from(edges.values)
                nx.write_gpickle(G, all_concept_tree_file_path)
            else: G = nx.read_gpickle(all_concept_tree_file_path)
        if not isinstance(subTree_concept_name, type(None)):
            Gs = []
            for area in subTree_concept_name:
                concept_names = nx.get_node_attributes(G, 'display_name') # 可以调用 self.get_node_id_given_attribute()
                node_concept_id = list(concept_names.keys())[list(concept_names.values()).index(area)]
                node_descendents = nx.descendants(G, node_concept_id)
                Gs.append(nx.subgraph(G, list(node_descendents.union({node_concept_id}))))
            G = Gs
        if len(sub_graph_attribute_values) > 0: 
            attribute, values = sub_graph_attribute_values
            where_conditions_ = ['{} IN ({})'.format(attribute, ', '.join("'{}'".format(value) for value in values))]
            concept_ids = DatabaseManager().query_table(table_name = 'concepts', columns=['id', 'level', 'display_name'], where_conditions=where_conditions_, batch_read=False).sort_values('level').id.values
            if same_area:
                all_nodes = set(nx.descendants(G, concept_ids[0])).intersection(set(nx.ancestors(G, concept_ids[-1])))
                for concept_id in concept_ids[1:-1]: all_nodes.update(set(nx.descendants(G, concept_id)).union(nx.ancestors(G, concept_id)))
            else:
                all_nodes = set()
                for concept_id in concept_ids: all_nodes.update(set(nx.descendants(G, concept_id)).union(nx.ancestors(G, concept_id)))
            all_nodes.update(concept_ids)
            G = nx.subgraph(G, list(all_nodes))
        if show: 
            from visualization import Visualizer
            # Visualizer.draw_network_old(G, layout = layout, fig_size=(24,16), draw_edge = True)
            Visualizer.draw_network(G, layout = layout, fig_size=fig_size, title = '_'.join([sub_str.replace(' ','_').replace('-','_') for sub_str in values]), margin = margin, x_scale_ratio = x_scale_ratio,  bbox_inches = bbox_inches, font_size = font_size, pad_inches = pad_inches, save = save)
        if return_concept_table:
            concepts_table = pd.read_csv(op.join(self.concept_tree_path, 'All_concepts_with_ancestors.csv'))
            return G, concepts_table
        else: return G
    
    @classmethod
    def level_0_ancestors(cls, table = None, connect_with = 'discipline'):
        if not isinstance(table,  pd.DataFrame):  
            table_concepts_path = op.join(cls.concept_tree_path, 'All_concepts_with_ancestors.csv')
            if not op.exists(table_concepts_path):
                concept_tree, table = cls.Tree(return_concept_table=True)
                concept_tree_names = nx.relabel_nodes(concept_tree, dict(table[['id','display_name']].values))
                concepts_0 = np.sort(table.loc[table.level==0].display_name.drop_duplicates().values)
                n_ancestors_path = op.join(cls.concept_tree_path, 'n_level_0_ancestors.npy')
                if not op.exists(n_ancestors_path):
                    concepts_non_0 = table.loc[table.level!=0].display_name.drop_duplicates().values
                    level_0_ancestor_indexes = []
                    for area in concepts_0:
                        level_0_ancestor_indexes.append([len(list(nx.all_simple_paths(concept_tree_names, area, concept))) for concept in concepts_non_0])
                    np.save(n_ancestors_path, np.array([concepts_non_0, level_0_ancestor_indexes], dtype = object))
                else: concepts_non_0, level_0_ancestor_indexes = np.load(n_ancestors_path,allow_pickle=True)
                level_0_ancestor_indexes = np.array(level_0_ancestor_indexes)
                n_ancestors_map = dict(np.vstack((concepts_non_0, np.sum(level_0_ancestor_indexes>0, axis=0))).T)
                table['n_ancestors'] = table.display_name.map(n_ancestors_map).fillna(0).astype('int')
                concepts_mutiple_ancestor = table.loc[table.n_ancestors>1].display_name.drop_duplicates().values # concepts with mutiple ancestors, dealing with concepts with multiple max number of paths
                getindex = lambda concept_set: np.concatenate([np.where(concepts_non_0==concept) for concept in concept_set]).flatten()
                n_max_ancectors = [np.sum(vec==vec.max()) for vec in level_0_ancestor_indexes.T[getindex(concepts_mutiple_ancestor)]]
                n_max_ancectors_map = np.concatenate([table.loc[table.n_ancestors<=1][['display_name','n_ancestors']].drop_duplicates().values, np.array([concepts_mutiple_ancestor,n_max_ancectors]).T])
                table['n_max_ancestors'] = table.display_name.map(dict(n_max_ancectors_map))
                
                # level_0_ancestor
                level_0_map_index = np.argmax(level_0_ancestor_indexes, axis=0).astype(int)
                level_0_ancestor_map, level_0_ancestor_map_2 = np.vstack((concepts_0, concepts_0)).T, np.vstack((concepts_non_0, concepts_0[level_0_map_index])).T
                level_0_ancestor_map = np.vstack((level_0_ancestor_map, level_0_ancestor_map_2))
                table['level_0_ancestor'] = table.display_name.map(dict(level_0_ancestor_map))

                # 仅仅因为和level 0 concept 有路径链接, 就对其进行分类, 似乎对高level的concept不太合理
                # level_0_ancestor_refined
                subset_1, subset_2, subset_3 = table.loc[table.n_max_ancestors<1], table.loc[table.n_max_ancestors==1], table.loc[table.n_max_ancestors>1]
                concepts_subset_1 =  subset_1.loc[subset_1.level>0].display_name.drop_duplicates().values
                map_level_0 = np.vstack((concepts_0, concepts_0)).T
                map_non_level_0 = np.vstack((concepts_subset_1, np.repeat('Undefined', concepts_subset_1.shape[0]))).T
                map_subset_1 = np.concatenate([map_level_0, map_non_level_0])
                concepts_subset_2 = subset_2.display_name.drop_duplicates().values
                subset_2_map_index = np.argmax(level_0_ancestor_indexes.T[getindex(concepts_subset_2)], axis=1).astype(int)
                map_subset_2= np.vstack((concepts_subset_2, concepts_0[subset_2_map_index])).T
                concepts_subset_3 = subset_3.display_name.drop_duplicates().values
                map_subset_3 = np.vstack((concepts_subset_3, np.repeat('Interdiscipline', concepts_subset_3.shape[0]))).T
                disciplinary_map = np.concatenate([map_subset_1, map_subset_2, map_subset_3])
                table['level_0_ancestor_refined'] = table.display_name.map(dict(disciplinary_map))

                # multiple_level_0_ancestors
                subset_2, subset_3 = table.loc[table.n_ancestors==1], table.loc[table.n_ancestors>1]
                concepts_subset_2 = subset_2.display_name.drop_duplicates().values
                subset_2_map_index = np.argmax(level_0_ancestor_indexes.T[getindex(concepts_subset_2)], axis=1).astype(int)
                map_subset_2= np.vstack((concepts_subset_2, concepts_0[subset_2_map_index])).T
                concepts_subset_3 = subset_3.display_name.drop_duplicates().values
                subset_3_map_index_df = pd.DataFrame(np.argwhere(level_0_ancestor_indexes.T[getindex(concepts_subset_3)]>0).astype(int),columns=['concept_index','ancestor_index'])
                subset_3_ancestors = subset_3_map_index_df.groupby('concept_index').apply(lambda df:','.join(concepts_0[df.ancestor_index.values]))
                subset_3_ancestors.index = concepts_subset_3
                mutiple_disciplinary_map = dict(subset_3_ancestors)
                mutiple_disciplinary_map.update(dict(np.concatenate([map_subset_1,map_subset_2])))
                table['multiple_level_0_ancestors'] = table.display_name.map(mutiple_disciplinary_map)

                index_no_level_0_ancestors = np.argwhere(np.sum(level_0_ancestor_indexes>0, axis=0)==0).flatten()
                concepts_no_level_0_ancestors = np.array(concepts_non_0)[index_no_level_0_ancestors]
                no_level_0_ancestor_concepts = pd.DataFrame(concepts_no_level_0_ancestors, columns=['no_level_0_ancestor_concept'])
                no_level_0_ancestor_concepts.to_csv(op.join(cls.concept_tree_path, 'no_level_0_ancestor_concepts.csv'), index=False)
                
                table.to_csv(table_concepts_path, index=False)
            else: table = pd.read_csv(table_concepts_path)
        if connect_with in table.loc[table.level==0].display_name.drop_duplicates().values:
            table = table.loc[table.multiple_level_0_ancestors.apply(lambda s: connect_with in s.split(','))]
        return table
    @classmethod
    def sample_concepts_with_ancestor_paths(cls):
        concept_tree, table = cls.Tree(return_concept_table=True)
        concept_tree_names = nx.relabel_nodes(concept_tree, dict(table[['id','display_name']].values))
        concepts_0 = np.sort(table.loc[table.level==0].display_name.drop_duplicates().values)
        n_ancestors_path = op.join(cls.concept_tree_path, 'n_level_0_ancestors.npy')
        concepts_non_0, level_0_ancestor_indexes = np.load(n_ancestors_path,allow_pickle=True)
        level_0_ancestor_indexes = np.array(level_0_ancestor_indexes)
        getindex = lambda concept_set: np.concatenate([np.where(concepts_non_0==concept) for concept in concept_set]).flatten()

        concepts_signle_ancestor = table.loc[(table.n_ancestors==1) & (table.level>0)]
        concepts_mutiple_ancestor_single_max = table.loc[(table.n_ancestors>1) & (table.n_max_ancestors==1)]
        concepts_mutiple_ancestor_mutiple_max = table.loc[table.n_max_ancestors>1]

        # concepts_signle_ancestor_index = getindex(concepts_signle_ancestor.display_name.drop_duplicates().values)
        # concepts_mutiple_ancestor_single_max_index = getindex(concepts_mutiple_ancestor_single_max.display_name.drop_duplicates().values)
        # concepts_mutiple_ancestor_mutiple_max_index = getindex(concepts_mutiple_ancestor_mutiple_max.display_name.drop_duplicates().values)

        def sample_concepts(table, num=1, concept_name=None):
            sampled_concept = table.sample(num).display_name.values if concept_name==None else concept_name
            n_paths = level_0_ancestor_indexes.T[getindex(sampled_concept)].flatten()
            indexes = np.argwhere(n_paths)
            disciplines_df = pd.DataFrame(np.hstack((concepts_0[indexes], n_paths[indexes])), columns=['ancestor','n_paths'], index=sampled_concept.tolist()*indexes.shape[0])
            return disciplines_df
        
        concept_df = sample_concepts(concepts_signle_ancestor)
        print(concept_df)
        print(5)
