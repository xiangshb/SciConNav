
# https://medium.com/swlh/the-ultimate-python-package-to-pre-process-data-for-machin-learning-c87bcc39fa66
import os.path as op
import pandas as pd
import numpy as np
import pymysql, tqdm
import networkx as nx
import time, os

this_file_path = op.abspath(__file__)
cmd_running = False
is_in_server = False

class sqlconnect():
    def __init__(self):
        self.db = pymysql.connect(user="", password= "", host = "", port = None, charset = 'utf8')

class Path(object):
    filedir = '../../../data/files/' if is_in_server else ('../files' if cmd_running else 'files') # op.join(os.getcwd(),'files')
    other_data_path = filedir if is_in_server else r'D:\data'
    embedding_path = op.join(filedir, 'Embeddings')
    concept_tree_path = op.join(filedir, 'Concept_Trees')
    model_path = op.join(embedding_path, 'model')
    train_data_path = op.join(embedding_path, 'train')
    inference_path = op.join(embedding_path, 'inference')
    external_data_dir = op.join(embedding_path, 'dists') if is_in_server else other_data_path

class Process(Path):
    def __init__(self):
        pass
    
    def map_column_to_num(table = None, column = 'attribute_name', start = 0):
        column_values = table[column].drop_duplicates()
        column_map = dict(pd.Series(range(0 + start, column_values.shape[0] + start), index = column_values.values))
        table[column + '_to_num'] = table[column].map(column_map)
        return column_map
    
    @classmethod
    def table_lexsort(cls, table): # specially for OpenAlexData().get_embedding_data()
        cls.map_column_to_num(table=table, column = 'author_id')
        cls.map_column_to_num(table=table, column = 'work_id')
        sort_indexes = []
        for _, table_k in table.groupby('author_id_to_num'):
            sort_indexes += table_k.index[np.lexsort(table_k[['work_id_to_num', 'publication_year']].values.T)].tolist()
        table = table.loc[sort_indexes]
        cls.map_column_to_num(table=table, column = 'work_id')
        return table
    
    @classmethod
    def massive_table_lexsort(cls, table):
        work_ids = table['work_id'].drop_duplicates()
        wrok_map = dict(pd.Series(range(0, work_ids.shape[0]), index = work_ids.values))
        table['work_id_to_num'] = table['work_id'].map(wrok_map)
        sort_indexes = [] 
        for _, table_k in table.groupby('author_id'):
            sort_indexes += table_k.index[np.lexsort(table_k[['work_id_to_num', 'publication_year']].values.T)].tolist()
        table = table.loc[sort_indexes]
        table = table.dropna()
        table.level = table.level.astype(int)
        table.index = range(table.shape[0]) 
        return table
    
    @classmethod
    def sample_concepts(cls, concept_names, levels = None, len_seq = 10, n_authors = 4):
        # author_ids = np.random.choice(concept_names.author_id.drop_duplicates(), n_authors, replace = False)
        author_ids = ['A158333346', 'A3179542944', 'A2419685366', 'A2443744265', 'A2109091881'] # [Mathematics Computer_science Physics Art Chemistry]
        # [1969 2260] [1920 2823] [1983 2742] [1800 2771] [1956 2758]
        for i in (levels if levels else range(0,3)):
            len_seq = 5 if i == 0 else 10
            concept_names_level_i = concept_names if len(levels)==1 else concept_names.loc[concept_names.level==i]
            sample_table = concept_names_level_i.loc[concept_names_level_i.author_id.isin(author_ids)]
            sample_concept_names = pd.DataFrame([], columns = author_ids)
            for author_id, table in sample_table.groupby('author_id'):
                m = int(table.shape[0]/2 - len_seq/2)
                sample_concept_names[author_id] = pd.concat([table.display_name[:len_seq], table.display_name[m:m+len_seq], table.display_name[-len_seq:]]).values
            path = op.join(cls.embedding_path, 'sample_concept_names_level_' + str(i) + '.csv')
            sample_concept_names.to_csv(path, index = False)
            print('Successfully saved\n',path)
    
    @classmethod
    def prep_concept_lists(cls, concept_names, level, multi_level):
        print('prepare lists for each author: level ',level, ' multi_level: ', 'True' if multi_level else 'False')
        if multi_level:
            level_adder = '_level_' + str(level) + '_to_5' if level else '_level_0_to_5'
        else: level_adder = '_level_' + str(level) if level else ''
        data_path = op.join(cls.train_data_path, 'concept_names_train' + level_adder + '.npy')
        concept_lists = []
        if not op.exists(data_path):
            for author, table in concept_names.groupby('author_id'):
                concept_lists.append(table.display_name.values.tolist())
            np.save(data_path, np.array(concept_lists, dtype=object))
        else: concept_lists = np.load(data_path, allow_pickle=True)
        return concept_lists
    
    @classmethod
    def concept_tree(cls, G = nx.DiGraph(), area = None, return_table = False):
        table_concepts = pd.read_csv(op.join(cls.concept_tree_path, 'All_concepts.csv'))
        if len(G) == 0: 
            edges = pd.read_csv(op.join(cls.concept_tree_path, 'All_concept_edges.csv'))
            G.add_nodes_from(table_concepts['id'].values)
            for attribute in table_concepts.columns[1:]: 
                nx.set_node_attributes(G, dict(table_concepts[['id', attribute]].values), attribute)
            G.add_edges_from(edges.values)
        if not isinstance(area, type(None)):
            concept_names = nx.get_node_attributes(G, 'display_name')
            if type(area) == str:
                area_concept_id = list(concept_names.keys())[list(concept_names.values()).index(area)]
                area_descendent_concepts = nx.descendants(G, area_concept_id)
                node_and_descendents = area_descendent_concepts.union({area_concept_id})
                G = nx.subgraph(G, list(node_and_descendents))
            if type(area) == list:
                G_areas = []
                for area_i in area:
                    area_concept_id = list(concept_names.keys())[list(concept_names.values()).index(area_i)]
                    area_descendent_concepts = nx.descendants(G, area_concept_id)
                    node_and_descendents = area_descendent_concepts.union({area_concept_id})
                    G_areas.append(nx.subgraph(G, list(node_and_descendents)))
                G = G_areas
        return (G, table_concepts) if return_table else G

    @classmethod
    def concept_pairs(cls, top_ratio = 0.2, return_areas = False):
        # 产生level 2的concept对儿, 
        concept_intersection_pairs = op.join(cls.concept_tree_path, 'concept_intersection_pairs.csv')
        table = pd.read_csv(op.join(cls.concept_tree_path, 'All_concepts.csv'))
        concepts_0 = table.loc[table.level==0].display_name.drop_duplicates().values.tolist()
        if not op.exists(concept_intersection_pairs):
            concept_tree, table = cls.concept_tree(return_table=True)
            concepts_0 = table.loc[table.level==0].display_name.drop_duplicates().values
            import warnings
            warnings.filterwarnings("ignore")
            area_concepts = pd.DataFrame(columns = ['area','sub_concept_names'])
            for area in concepts_0:
                concetp_tree = cls.concept_tree(G = concept_tree, area = area)
                concept_names = nx.get_node_attributes(concetp_tree, 'display_name')
                area_concepts = area_concepts.append({'area': area, 'sub_concept_names': set(concept_names.values())}, ignore_index=True)
            intersection_matrix = np.array([area_concepts['sub_concept_names'].apply(lambda x: len(work_refs_i.intersection(x))).tolist() for work_refs_i in area_concepts['sub_concept_names']]) 
            inter_sec_triu = np.triu(intersection_matrix, k=1)
            count_array = np.array(np.unique(inter_sec_triu, return_counts=True)).T[::-1]
            index = np.argwhere(np.cumsum(count_array[:,1]) / count_array[:,1].sum() > top_ratio)[0,0]
            indexes = np.array(np.where(inter_sec_triu > count_array[index,0])).T
            concept_pairs_df = pd.DataFrame(concepts_0[indexes])
            concept_pairs_df.to_csv(concept_intersection_pairs, index = False)
        else: concept_pairs_df = pd.read_csv(concept_intersection_pairs)
        return (concept_pairs_df, concepts_0) if return_areas else concept_pairs_df
    
    @classmethod
    def level_0_ancestors(cls):
        table_concepts_path = op.join(cls.concept_tree_path, 'All_concepts_with_ancestor.csv')
        if not op.exists(table_concepts_path):
            concept_tree, table = cls.concept_tree(return_table=True)
            concept_tree_names = nx.relabel_nodes(concept_tree, dict(table[['id','display_name']].values))
            concepts_0 = table.loc[table.level==0].display_name.drop_duplicates().values
            concepts_non_0 = table.loc[table.level!=0].display_name.drop_duplicates().values
            concepts_0.sort()
            level_0_ancestor_indes = []
            level_0_ancestor_map = np.vstack((concepts_0, concepts_0[np.arange(19)])).T
            for area in concepts_0:
                level_0_ancestor_indes.append([len(list(nx.all_simple_paths(concept_tree_names, area, concept))) for concept in concepts_non_0])
            level_0_map_index = np.argmax(level_0_ancestor_indes, axis=0).astype(int)
            level_0_ancestor_map_2 = np.vstack((concepts_non_0, concepts_0[level_0_map_index])).T
            level_0_ancestor_map = np.vstack((level_0_ancestor_map, level_0_ancestor_map_2))
            np.save(op.join(cls.concept_tree_path, 'n_level_0_ancestors.npy'),  np.array([concepts_non_0, level_0_ancestor_indes], dtype = object))
            table['level_0_ancestor'] = table.display_name.map(dict(level_0_ancestor_map))
            index_no_level_0_ancestors = np.argwhere(np.sum(np.array(level_0_ancestor_indes)>0, axis=0)==0).flatten()
            concepts_no_level_0_ancestors = np.array(concepts_non_0)[index_no_level_0_ancestors]
            no_level_0_ancestor_concepts = pd.DataFrame(concepts_no_level_0_ancestors, columns=['no_level_0_ancestor_concept'])
            no_level_0_ancestor_concepts.to_csv(op.join(OpenAlexData.concept_tree_path, 'no_level_0_ancestor_concepts.csv'), index=False)
            n_ancestors_map = dict(np.vstack((concepts_non_0, np.sum(np.array(level_0_ancestor_indes)>0, axis=0))).T)
            table['n_ancestors'] = table.display_name.map(n_ancestors_map).fillna(0).astype('int')
            table.to_csv(table_concepts_path, index=False)
        else: table = pd.read_csv(table_concepts_path)
        return table

class OpenAlexData(Path, sqlconnect):
    name = 'openalex2022'
    discipline_groups = ['theoretical', 'applied', 'biomedical', 'chemistry', 'geographical_environment', 'business_economy', 'philosophy_social_science', 'political_history']
    discipline_group = dict()
    discipline_group['theoretical'] = ['Mathematics', 'Physics']
    discipline_group['applied'] = ['Computer science', 'Engineering']
    discipline_group['biomedical'] = ['Biology', 'Medicine']
    discipline_group['chemistry'] = ['Chemistry', 'Materials science']
    discipline_group['geographical_environment'] = ['Geography', 'Geology', 'Environmental science']
    discipline_group['business_economy'] = [ 'Economics', 'Business']
    discipline_group['philosophy_social_science'] = ['Philosophy', 'Psychology', 'Sociology', 'Art']
    discipline_group['history_political'] = ['History', 'Political science']

    def __init__(self):
        super(sqlconnect).__init__()

    @staticmethod
    def access_table_cmd_line(sql_cmd = '', given_attribute = '', given_value = None, split_size = 5000, progress_bar = False, new_connection = False):
        query_tails, table = [], pd.DataFrame()
        if len(given_attribute) > 0 and len(given_value) > 0:
            link_word = ' and ' if 'where' in sql_cmd else ' where '
            if len(given_value) > split_size:
                given_value_ranges = np.array_split(given_value, np.ceil(len(given_value) / split_size))
                for given_value_i in given_value_ranges:
                    if len(given_value_i) <= 1: # it's impossible that len(given_value_i) == 0
                        query_tail = link_word + given_attribute + ' = ' + str(tuple(given_value_i)).replace(",", "").replace(")", "").replace("(", "")
                    else: query_tail = link_word + given_attribute + ' in ' + str(tuple(given_value_i))
                    query_tails.append(query_tail)
            elif len(given_value) > 1:
                query_tails.append(link_word + given_attribute + ' in ' + str(tuple(given_value)))
            else: query_tails.append(link_word + given_attribute + ' = ' + str(tuple(given_value)).replace(",", "").replace(")", "").replace("(", ""))
        else: query_tails = ['']
        quary_iterate = tqdm.tqdm(query_tails) if progress_bar else query_tails
        for query_tail in quary_iterate:
            table = pd.concat((table, pd.read_sql(sql_cmd + query_tail, con = sqlconnect().db)), ignore_index=True)
        return table
    
    def do_trim_enter(self, attributes = [], attributes_trim_enter = []):
        for target_attribute in attributes_trim_enter:
            # if '.' in target_attribute: table_name, attribute_name = target_attribute.split('.')
            if target_attribute in attributes:
                index = attributes.index(target_attribute)
                attributes.insert(index, "trim('\r' from " + target_attribute + ") as " + target_attribute)
                attributes.remove(target_attribute)
        return attributes

    @classmethod
    def read_table(cls, table_name = [], attributes = [], attributes_trim_enter = [], join_types = [], join_conditions = [], conditions = [], given_attribute = '', given_value = [], split_size = 5000, parallel_workers = -1, progress_bar = False, print_task = False, drop_duplicates = True):
        if len(table_name) == len(join_types) + 1: pass
        else: raise ValueError('Unable to join table')
        if len(attributes) <= 0:
            if len(table_name) == 0:
                raise ValueError('No table to read')
            if len(table_name) == 1:
                attributes_query = cls.attributes_dict[table_name[0]]
            else: raise ValueError('No attributes to access')
        else: attributes_query = attributes
        attributes_query = cls.do_trim_enter(cls, attributes = attributes_query, attributes_trim_enter = attributes_trim_enter)
        if isinstance(table_name, list) and len(table_name) > 0: # only for two tables 
            if print_task: print('Access table ' + ', '.join(table_name) + ' : ' + ', '.join(attributes_query))
            query_head = 'select ' + ','.join(attributes_query) + ' from ' + cls.name + '.' + table_name[0]
            for i in range(len(join_types)): query_head += ' ' + join_types[i] + ' ' + cls.name + '.' + table_name[i+1] + ' on ' + join_conditions[i]
            if len(conditions) > 0: 
                additional_conditions = ' where ' + conditions[0]
                for conditon in conditions[1:]:
                    additional_conditions += ' and ' + conditon
            else: additional_conditions = ''
        else: raise ValueError('Table names not a list or no name in list')
        sql_querys, query_tails, table = [], [], pd.DataFrame()
        if len(given_attribute) > 0 and len(given_value) > 0:
            link_word = ' and ' if 'where' in additional_conditions else ' where '
            if len(given_value) > split_size:
                given_value_ranges = np.array_split(given_value, np.ceil(len(given_value) / split_size))
                for given_value_i in given_value_ranges:
                    if len(given_value_i) <= 1: 
                        query_tail = link_word + given_attribute + ' = ' + str(tuple(given_value_i)).replace(",", "").replace(")", "").replace("(", "")
                    else: query_tail = link_word + given_attribute + ' in ' + str(tuple(given_value_i))
                    query_tails.append(query_tail)
            elif len(given_value) > 1:
                query_tails.append(link_word + given_attribute + ' in ' + str(tuple(given_value)))
            else: query_tails.append(link_word + given_attribute + ' = ' + str(tuple(given_value)).replace(",", "").replace(")", "").replace("(", ""))
        else: query_tails = ['']
        if parallel_workers > 0:
            for query_tail in query_tails: sql_querys.append( query_head + additional_conditions + query_tail )
            parallel_workers = parallel_workers if cpu_count() > parallel_workers else cpu_count()
            quary_iterate = tqdm.tqdm(sql_querys) if progress_bar else range(sql_querys)
            with cf.ThreadPoolExecutor(max_workers = parallel_workers) as executor: 
                for table_i in executor.map(cls.read_table_i, quary_iterate):
                    table = table.append(table_i)
        else:
            quary_iterate = tqdm.tqdm(query_tails) if progress_bar else query_tails
            table_dfs = []
            for query_tail in quary_iterate:
                sql_query = query_head + additional_conditions + query_tail
                table_dfs.append(pd.read_sql(sql_query, con = sqlconnect().db))
            table = pd.concat(table_dfs)
        return table.drop_duplicates() if drop_duplicates else table

    @classmethod
    def get_concept_lists(cls, level = None, multi_level = True):
        if not multi_level: 
            if level == None: raise ValueError('for single_level names, level value must be assigned')
            level_adder = '_level_' + str(level) if level else ''
        else: level_adder = '_level_' + str(level) + '_to_5' if level else '_level_0_to_5'
        concept_lists_path = op.join(cls.train_data_path, 'concept_names_train' + level_adder + '.npy')
        if not op.exists(concept_lists_path):
            concept_names_path = op.join(cls.embedding_path, 'all_author_id_concept_names' + level_adder + '.csv')
            if not op.exists(concept_names_path): # generate single level
                all_concept_names_path = op.join(cls.embedding_path, 'all_author_id_concept_names.csv')
                if not op.exists(all_concept_names_path):
                    table_sorted_path = op.join(cls.embedding_path, 'All_author_work_concept_ids_sorted.csv')
                    if not op.exists(table_sorted_path):
                        cmd_line = 'select author_id from openalex2022.author_yearlyfeature_field_geq10pubs where total_pubs >= 50'
                        author_ids = cls.access_table_cmd_line(cmd_line)
                        cmd_line = 'select works_authorships.author_id,works_authorships.work_id,works.publication_year,works_concepts.concept_id,concepts.display_name,concepts.level from openalex2022.works_authorships '
                        cmd_line += 'left join openalex2022.works on works_authorships.work_id = works.id left join openalex2022.works_concepts on works.id = works_concepts.work_id left join openalex2022.concepts on works_concepts.concept_id = concepts.id'
                        author_work_concept_ids = cls.access_table_cmd_line(sql_cmd = cmd_line, given_attribute= 'works_authorships.author_id', given_value=author_ids['author_id'].values, progress_bar = True)
                        author_work_concept_ids = Process.massive_table_lexsort(table = author_work_concept_ids) # or Process.table_lexsort()
                        author_work_concept_ids.to_csv(table_sorted_path, index = False)
                    else: author_work_concept_ids = pd.read_csv(table_sorted_path)
                    author_id_concept_names = author_work_concept_ids[['author_id','display_name','level']]
                    author_id_concept_names.to_csv(all_concept_names_path, index = False)
                else: author_id_concept_names = pd.read_csv(all_concept_names_path)
                for level_i in range(0,3):
                    level_adder_i = '_level_' + str(level_i) + '_to_5'
                    successive_path = op.join(cls.embedding_path, 'all_author_id_concept_names' + level_adder_i +'.csv')
                    temp = author_id_concept_names[author_id_concept_names.level.isin(range(level_i, 6))].drop(['level'], axis=1)
                    temp.to_csv(successive_path, index = False)
                    Process.prep_concept_lists(temp, level = level_i, multi_level = multi_level)
                    print('file ', successive_path, ' cashed')
                concept_names = author_id_concept_names[['display_name']]
                concept_names.to_csv(op.join(cls.embedding_path, 'all_concept_names.csv'), index = False)
                for i in range(1,6):
                    author_id_concept_names[author_id_concept_names.level==i].drop(['level'], axis=1).to_csv(op.join(cls.embedding_path, 'all_author_id_concept_names_level_' + str(i) +'.csv'), index = False)
                    concept_names[author_id_concept_names.level==i].to_csv(op.join(cls.embedding_path, 'all_concept_names_level_' + str(i) +'.csv'), index = False)
            else: 
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()))
                ts = time.time()
                concept_names = pd.read_csv(concept_names_path)
                te = time.time()
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()), te-ts)
                concept_lists = Process.prep_concept_lists(concept_names, level = level, multi_level = multi_level)
        else: 
            print(time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()))
            ts = time.time()
            concept_lists = np.load(concept_lists_path, allow_pickle=True)
            te = time.time()
            print(time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()), te-ts)
        return concept_lists

