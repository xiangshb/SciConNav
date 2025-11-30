import matplotlib.pyplot as plt
import seaborn as sns
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as skTSNE
from dataset import Process
from model import Embedding
from openTSNE import TSNE as opTSNE
from matplotlib.lines import Line2D
import os.path as op
from collections import OrderedDict

# import geoplotlib as geoplot
# from pandas.plotting import radviz

# import geoplotlib as geoplot
# https://blog.csdn.net/weixin_52011624/article/details/127273742
# https://mathematica.stackexchange.com/questions/252664/find-the-data-set-for-boundary-edge-of-listplot
# edge convec hull

LEVEL_0_CONCEPT_COLORS = OrderedDict([("Art", "#A5C93D"), ("Biology", "#8B006B"), ("Business", "#2000D7"), ("Chemistry", "#538CBA"), 
                                      ("Computer science", "#d7abd4"), ("Economics", "#B33B19"), ("Engineering", "#2d74bf"), 
                                      ("Environmental science", "#9e3d1b"), ("Geography", "#3b1b59"), ("Geology", "#C38A1F"), 
                                      ("History", "#11337d"), ("Materials science", "#1b5d2f"), ("Mathematics", "#51bc4c"), 
                                      ("Medicine", "#ffcb9a"), ("Philosophy", "#768281"), ("Physics", "#a0daaa"), 
                                      ("Political science", "#8c7d2b"), ("Psychology", "#98cc41"), ("Sociology", "#c52d94"), 
                                      ("Interdiscipline", 'red'), ("Undefined", "#00846F"),("Background","#dcecf5")])
discipline_abbreviations = OrderedDict([("Art", "Art"), ("Biology", "Bio."), ("Business", "Busi."), ("Chemistry", "Chem."), 
                                      ("Computer science", "C.S."), ("Economics", "Econ."), ("Engineering", "Eng."), 
                                      ("Environmental science", "Env.Sci."), ("Geography", "Geog."), ("Geology", "Geol."), 
                                      ("History", "Hist."), ("Materials science", "Mat.Sci."), ("Mathematics", "Math."), 
                                      ("Medicine", "Med."), ("Philosophy", "Philo."), ("Physics", "Phys."), 
                                      ("Political science", "Pol.Sci."), ("Psychology", "Psych."), ("Sociology", "Sociol."), 
                                      ("Interdiscipline", 'Inter.'), ("Undefined", "UD")])

MOUSE_10X_COLORS = {
    0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
    22: "#FF2F80",
    23: "#61615A",
    24: "#BA0900",
    25: "#6B7900",
    26: "#00C2A0",
    27: "#FFAA92",
    28: "#FF90C9",
    29: "#B903AA",
    30: "#D16100",
    31: "#DDEFFF",
    32: "#000035",
    33: "#7B4F4B",
    34: "#A1C299",
    35: "#300018",
    36: "#0AA6D8",
    37: "#013349",
    38: "#00846F",
}

class Visualizer():
    def color_map_emphasize(emphasize_discipline = None, background_color = "#dcecf5", emphasize_color = None):
        if emphasize_discipline == None: return LEVEL_0_CONCEPT_COLORS
        LEVEL_0_CONCEPT_COLORS_Alter = LEVEL_0_CONCEPT_COLORS.copy()
        if emphasize_color != None and isinstance(emphasize_discipline, str): # this means to keep the original color
            LEVEL_0_CONCEPT_COLORS_Alter[emphasize_discipline] = emphasize_color
        for discipline in LEVEL_0_CONCEPT_COLORS_Alter.keys():
            if discipline != emphasize_discipline:
                LEVEL_0_CONCEPT_COLORS_Alter[discipline] = background_color
        return LEVEL_0_CONCEPT_COLORS_Alter
    
    def get_node_labels_plot(self, G, positions):
        node_display_name = nx.get_node_attributes(G, 'display_name')
        node_level = nx.get_node_attributes(G, 'level')
        node_labels = {}
        def chunk_list(input_list, chunk_size):
            return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

        for id, display_name, level in zip(node_display_name.keys(), node_display_name.values(), node_level.values()):
            display_name_ = display_name.split(' ')
            if len(display_name_)>2: 
                name_ = []
                for list_ in chunk_list(display_name_, 2):
                    name_.append(' '.join(list_))
                node_labels[id] = '\n'.join(name_)
            else: node_labels[id] = display_name.replace(' ','\n') + ': {}'.format(level)
        return node_labels
    
    @classmethod
    def draw_network_old(self, network, layout = 'graphviz', fig_size=(24,16), node_labels = dict(), draw_edge = False, line_wide = 1,  edge_labels = dict(), pause = 0):
        plt.figure(figsize=fig_size)
        if layout == 'graphviz':
            positions = graphviz_layout(network, prog='dot')
        else: positions=nx.spring_layout(network)
        default_label = True if len(node_labels) == 0 else False
        nx.draw(network, with_labels = default_label, pos=positions, width = line_wide, node_color = 'red', edge_color = 'green')
        if not default_label: nx.draw_networkx_labels(network, pos = positions, labels = node_labels, width = line_wide, node_color = 'red', edge_color = 'green')
        if draw_edge: nx.draw_networkx_edge_labels(network,pos=positions,edge_labels=edge_labels)

    @classmethod
    def draw_network(cls, G, layout = 'graphviz', fig_size=(24,16), node_labels = dict(),  edge_labels = dict(), title = None, font_size = 18, margin = [-0.1, 1.1, -0.1, 1.1], x_scale_ratio = 0.8, bbox_inches = 'tight',pad_inches = 0.1, save = False):
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.figure(figsize=fig_size)
        plt.axis('off')
        positions = graphviz_layout(G, prog='dot') if layout == 'graphviz' else nx.spring_layout(G) # dot
        pos_x, pos_y = np.array(list(positions.values())).T
        pos_x_mid = (pos_x.max() + pos_x.min()) / 2
        pos_x = pos_x_mid + (pos_x - pos_x_mid) * x_scale_ratio
        for key_, value_ in zip(positions.keys(), [(x_,y_) for x_,y_ in zip(pos_x,pos_y)]): positions[key_] = value_
        nx.draw_networkx_nodes(G, pos=positions, node_color = 'lime', node_shape='.', node_size=400)
        node_labels = node_labels if len(node_labels)>0 else cls.get_node_labels_plot(cls, G, positions)
        nx.draw_networkx_labels(G, pos = positions, labels = node_labels, font_size=font_size)
        nx.draw_networkx_edges(G, pos = positions, connectionstyle="arc3, rad=0.2", edge_color = 'dodgerblue')
        if len(edge_labels)>0: nx.draw_networkx_edge_labels(G,pos=positions, edge_labels=edge_labels)
        left, right, bottom, top = margin
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        if save: 
            from dataset import Path
            plt.savefig(op.join(Path.concept_tree_path, title+'.pdf'),bbox_inches = bbox_inches, pad_inches = pad_inches if bbox_inches == 'tight' else 0.1)
        plt.show()

    @staticmethod
    def corrolation_map(data_df):
        sns.scatterplot(data=data_df[data_df.type=='closeness'], x="centrality", y="n_ancestors", color='red')
        sns.scatterplot(data=data_df[data_df.type=='betweenness'], x="centrality", y="n_ancestors", color='green')
        sns.kdeplot(data=data_df[data_df.type=='closeness'], x="centrality", hue="n_ancestors", bw_adjust=0.01, common_norm=False, palette=sns.color_palette("tab10"))
        sns.kdeplot(data=data_df[data_df.type=='betweenness'], x="centrality", hue="n_ancestors")
        sns.ecdfplot(data=data_df[data_df.type=='closeness'], x="centrality", hue="n_ancestors", palette=sns.color_palette("tab10"))
        sns.ecdfplot(data=data_df[data_df.type=='betweenness'], x="centrality", hue="n_ancestors", palette=sns.color_palette("tab10"))
        sns.kdeplot(data=data_df[data_df.type=='closeness'], x="centrality", hue="level_range", bw_adjust=0.1, common_norm=False)
        sns.kdeplot(data=data_df[data_df.type=='betweenness'], x="centrality", hue="ances_type", bw_adjust=0.1, common_norm=False)

        sns.scatterplot(data=data_df[(data_df.type=='closeness') & (data_df.centrality_size=='high')], x="centrality", y="level")
        sns.kdeplot(data=data_df[(data_df.type=='closeness') & (data_df.centrality_size=='high')], x="centrality", hue="level_range", bw_adjust=0.1, common_norm=False)
        pass

    @staticmethod
    def bar_plot(data_df, title, save=True):
        percentage = data_df.ratio.values
        _, ax = plt.subplots(figsize=(8,6))
        _palette={'Mathematics':LEVEL_0_CONCEPT_COLORS['Mathematics'],'All areas':LEVEL_0_CONCEPT_COLORS['Medicine']} # {'Mathematics': 'grey','All areas': 'orange'}
        ax=sns.barplot(data=data_df, errwidth=0.5, x='step_len', y='ratio', hue='area', palette= _palette)
        patches = ax.patches
        for i in range(len(patches)):
            x = patches[i].get_x() + patches[i].get_width()/2
            x += (-0.03 if i<6 else 0.03)
            y = patches[i].get_height()+0.003
            # if i == 0:y+=0.003
            # if i == 10: y+=0.005
            ax.annotate('{:.1f}'.format(100*percentage[i]), (x, y), ha='center', fontsize=16)
        # plt.title(title, loc="center")
        sns.despine()
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.xlabel('step size', fontsize = 18) # ,labelpad=10
        plt.ylabel('percentage (%)', fontsize = 20)
        plt.tight_layout()
        # if save: plt.savefig(op.join(Process.external_data_dir, 'step_size_ratio_'+title+'.pdf'), bbox_inches='tight')
        if save: plt.savefig(op.join(Process.external_data_dir, 'step_size_ratio_'+title+'.png'), dpi=500, bbox_inches='tight')
        plt.show()

    @staticmethod
    def chord_plot(matrix_df, save, level):
        # https://moshi4.github.io/pyCirclize/chord_diagram/
        from pycirclize import Circos
        if level == 0:
            circos = Circos.initialize_from_matrix(
                matrix_df,
                space=5,
                cmap = LEVEL_0_CONCEPT_COLORS,
                label_kws=dict(size=12),
                link_kws=dict(ec="black", lw=0.5, direction=0))
        elif level == 1:
            circos = Circos.initialize_from_matrix(
                matrix_df,
                space=5,
                label_kws=dict(size=10),
                link_kws=dict(ec="black", lw=0.5, direction=0))
        else: raise ValueError('undefiend level value')
        if save: circos.savefig('../concept_embedding/figures/average_path_weight.png')
        else: circos.plotfig()
    
    @staticmethod
    def similarity_pdf_ances_vs_nonances(data_df, title = 'similarity_distribution', curve = 'kde', bw_adjust=1, save = False):
        # https://seaborn.pydata.org/tutorial/distributions.html
        import seaborn as sns
        sns.displot(data_df, x='similarity', hue='type', kind= curve, bw_adjust=bw_adjust, common_norm=False, palette = {'inter-non-ancestor':'blue', 'inter-ancestor':'fuchsia', 'mono-non-ancestor':'lime', 'mono-ancestor':'red'})
        # plt.legend(loc='upper right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        if save: plt.savefig('./concept_embedding/figures/'+title+'.png', dpi=400, bbox_inches='tight')
        # plt.title(title, loc="center")
        plt.show()

    @staticmethod
    def multiple_pdf_plot(data_df, title = 'distance distribution', curve = 'kde',fig_size = (6,6), bw_adjust=1, save = False, file_name = 'None'):
        # https://seaborn.pydata.org/tutorial/distributions.html
        import seaborn as sns
        _, ax = plt.subplots(figsize=fig_size)
        _hue, _x = data_df.columns[np.argsort(['dist' in column for column in data_df.columns])]
        if curve == 'kde': 
            sns.kdeplot(ax=ax, data = data_df, x=_x, hue=_hue, bw_adjust=bw_adjust, common_norm=False, palette = LEVEL_0_CONCEPT_COLORS)
        elif curve == 'ecdf': sns.displot(ax=ax, data = data_df, x='distance', hue='area', element = 'step', kind= curve, common_norm=False, palette= LEVEL_0_CONCEPT_COLORS)
        else: raise ValueError('undefined cruve style')
        plt.title(title, loc="center")
        if save: plt.savefig(op.join(Process.external_data_dir, file_name+'.pdf'), bbox_inches='tight')
        plt.show()

    @staticmethod
    def mutiple_box_plot(data_df,title,file_name,fig_size = (6,6), save=False, xlabel = ''):
        import seaborn as sns
        labels = data_df.target_area.drop_duplicates().values[::-1]
        data_df_groub = data_df.groupby('target_area').apply(lambda df:df.distance.values)
        data, names = data_df_groub.values, data_df_groub.index.tolist()
        sort_index = [names.index(area) for area in labels]
        _, ax = plt.subplots(figsize=fig_size)
        plt.boxplot(x=data[sort_index], vert=False, labels = [discipline_abbreviations[label] for label in labels], showfliers=False, showmeans=True, showcaps=True)
        plt.title(title, loc="center", fontsize=14)
        plt.xlabel(xlabel, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=12)
        plt.xlim(0.4, 1.26)
        plt.tight_layout()
        if save: plt.savefig(op.join(Process.external_data_dir, file_name+'.png'), bbox_inches='tight')
        plt.show()

    @staticmethod
    def pdf_plot(data, title, bw_adjust = 1, save=False, colors=['dodgerblue', 'lime', 'magenta', 'red']):
        import seaborn as sns
        if isinstance(data, pd.DataFrame):
            fig, ax = plt.subplots(figsize=(4,4))
            sns.kdeplot(ax=ax, data=data, palette=dict(zip(data.columns.to_list(), colors)))
            plt.title(title, loc="center") # , fontsize=16
        else:
            sns.displot(data, kind="kde", fill=True)
            plt.title(title, loc="center")
        if save:plt.savefig(op.join(Process.external_data_dir, title.replace(' ','_')+'.pdf'), bbox_inches='tight')
        plt.show()

    @staticmethod
    def mapping_radar(df, area = None, n_areas = 19, k = 0, sigle_label = False, save = False):
        def get_params(df):
            df_2 = df.copy()
            from math import pi
            columns, indexes = list(df.columns), df.index
            if 'ancestor' in columns:columns.remove('ancestor')
            df_2[columns[0]+'_end'] = df_2[columns[0]]
            angles = [n / float(len(columns)) * 2 * pi for n in range(len(columns))]
            angles.append(angles[0])
            return df_2, columns, angles
        if isinstance(df, pd.DataFrame):
            df_2, columns, angles = get_params(df)
            for i, (area, table) in enumerate(df_2.groupby('ancestor')):
                values = table.drop('ancestor', axis=1).values
                if n_areas == 18:
                    if area == 'Environmental science':continue
                    k += 1
                    ax = plt.subplot(3,6,k, polar=True)
                else: ax = plt.subplot(4,5,i+1, polar=True)
                if sigle_label:
                    columns_sub = [''] * len(columns)
                    columns_sub[columns.index(area)] = columns.index(area)+1
                else: columns_sub = list(range(1,len(columns)+1))
                plt.xticks(angles[:-1], columns_sub, color='red', size=12)
                ax.set_rlabel_position(0) # Draw ylabels
                range_f = [round(num, 1) for num in np.hstack((np.arange(-1, 0,0.2), np.arange(0,1.1,0.2)))]
                plt.yticks(range_f, ['{:g}'.format(num) for num in range_f], color="grey", size=7)
                plt.ylim(-1,1)
                for i in range(len(values)):
                    ax.plot(angles, values[i], color='green', linewidth=1, linestyle='solid', alpha=0.4) # Plot data
                    ax.fill(angles, values[i], alpha=0) # MOUSE_10X_COLORS[i]
        elif isinstance(df, list) and area is not None:
            radar_values = []
            for df_i in df:
                df_sub, columns, angles = get_params(df_i)
                radar_values.append(df_sub.loc[df_sub.ancestor==area].drop('ancestor', axis=1).values)
            for i in range(len(radar_values)):
                values = radar_values[i]
                title = area+'_radar_mapping_level_' + str(i+1) + '_to_level_0'
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                if sigle_label:
                    columns_sub = [''] * len(columns)
                    columns_sub[columns.index(area)] = columns.index(area)+1
                else: columns_sub = list(range(1,len(columns)+1))
                column_colors = [LEVEL_0_CONCEPT_COLORS[discipline] for discipline in columns]
                plt.xticks(angles[:-1], columns_sub, color='red', size=12)
                for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), column_colors):
                    ticklabel.set_color(tickcolor)
                ax.set_rlabel_position(0) # Draw ylabels
                range_f = [round(num, 1) for num in np.arange(-0.8, 1.1, 0.3)] # np.hstack((np.arange(-1, 0,0.2), np.arange(0,1.1,0.2)))
                plt.yticks(range_f, ['{:g}'.format(num) for num in range_f], color="grey", size=7)
                plt.ylim(-1,1)
                if i == 0:
                    for k in range(len(values)):
                        ax.plot(angles, values[k], color=LEVEL_0_CONCEPT_COLORS[area], linewidth=1, linestyle='solid', alpha=0.4) # Plot data color='blueviolet'
                        ax.fill(angles, values[k], alpha=0) # MOUSE_10X_COLORS[i]
                    plt.title(area) 
                    if save: plt.savefig(op.join(Embedding.embedding_path, 'figures', 'level_1_to_level_0', title+'.png'), dpi=400, pad_inches=0.01, bbox_inches='tight')
                elif i == 1:
                    for k in range(len(values[:-3])):
                        ax.plot(angles, values[k], color=LEVEL_0_CONCEPT_COLORS[area], linewidth=1, linestyle='solid', alpha=0.1) # Plot data color='blueviolet'
                        ax.fill(angles, values[k], alpha=0) # MOUSE_10X_COLORS[i]
                    
                    area_inclination_mean = np.mean(values[:-3], axis=0) 
                    area_inclination_quantile_small = np.quantile(values[:-3], 0.05, axis=0) 
                    area_inclination_quantile_large = np.quantile(values[:-3], 0.95, axis=0) 
                    similarity_statistic = [area_inclination_quantile_small, area_inclination_mean, area_inclination_quantile_large]
                    statistic_color = ['black', 'blue', 'red'] # 'blueviolet'
                    
                    for statistic_, color_ in zip(similarity_statistic, statistic_color):
                        ax.plot(angles, statistic_, color=color_, linewidth=2, linestyle='solid', alpha=0.6) # Plot data
                        ax.fill(angles, statistic_, alpha=0) # MOUSE_10X_COLORS[i]
                    plt.title(area)
                    if save: plt.savefig(op.join(Embedding.embedding_path, 'figures', 'level_2_to_level_0', title+'.png'), dpi=400, pad_inches=0.05, bbox_inches='tight')
                # plt.show()
    
    @staticmethod
    def similarity_spectrum(df, attr_1, attr_2, title, file_name='test'):
        areas = df[attr_1].drop_duplicates().values
        val_map = dict([[areas[i], i] for i in range(len(areas))])
        df[attr_1 + '_map'] = df[attr_1].map(val_map)
        pal = sns.color_palette(palette='coolwarm', n_colors=len(areas))
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(df, hue = attr_1 + '_map', palette=pal,row=attr_1, aspect=10, height=0.6)
        g.map_dataframe(sns.kdeplot, x=attr_2, clip_on=True, fill=True, alpha=0.4)
        g.map_dataframe(sns.kdeplot, x=attr_2, clip_on=True, color='g')
        g.map(plt.axhline, y=0, lw=2, clip_on=False)
        for i, ax in enumerate(g.axes.flat):
            ax.text(-0.98, 0.5, areas[i], fontweight='bold', fontsize=15, color=ax.lines[-1].get_color()) # month_dict
        g.fig.subplots_adjust(hspace=-.5)
        g.set_titles("")
        g.despine(left=True)
        g.set(yticks=[], ylabel='', xlabel=attr_2)
        plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
        plt.savefig(op.join(Process.embedding_path, 'dists', file_name+'.pdf'), bbox_inches='tight')
        plt.show()

    @staticmethod
    def draw_analogy_graph(Gs, layout = 'graphviz', fig_size=(24,16), combine = True, line_width = 1, title = '', node_color_types = ['node_color','ma_node_color'], ancestor_types = ['ancestor','multiple_ancestor'], correct_edge_type = ['ce', 'mace'], edge_styles = ['edge_style', 'ma_edge_style'], save = False, discipline_abbrev = False):
        if isinstance(Gs, list) and len(Gs) > 1:
            if combine: plt.figure(figsize=fig_size) 
            for i in range(len(Gs)):
                G = Gs[i]
                if combine: plt.subplot(1,len(Gs),i+1)
                else:plt.figure(figsize=fig_size)
                # plt.rcParams["figure.figsize"] = fig_size
                plt.rcParams["figure.autolayout"] = True
                if layout == 'graphviz': positions = graphviz_layout(G, prog='dot') # prog=['neato'|'dot'|'twopi'|'circo'|'fdp'|'nop']
                elif layout == 'spring': positions=nx.spring_layout(G)
                elif layout == 'planar': positions=nx.planar_layout(G)
                elif layout == 'spectral': positions=nx.spectral_layout(G)
                elif layout == 'shell': positions=nx.shell_layout(G)
                elif layout == 'random': positions=nx.random_layout(G)
                elif layout == 'kamada_kawai': positions=nx.kamada_kawai_layout(G)
                elif layout == 'circular': positions=nx.circular_layout(G)
                elif layout == 'bipartite': positions=nx.bipartite_layout(G,nodes=G.nodes)
                elif layout == 'shell_planar': positions = nx.shell_planar_layout(G)
                elif layout == 'mds': positions = nx.mds_layout(G)
                elif layout == 'fruchterman_reingold': positions = nx.fruchterman_reingold_layout(G)
                elif layout == 'spiral': positions = nx.spiral_layout(G)

                else: raise ValueError('Undefined layout')
                edge_colors, node_colors = np.array([G.edges[u, v]['edge_color'] for u,v in G.edges]), nx.get_node_attributes(G, node_color_types[i]).values()
                ancestor_map = nx.get_node_attributes(G, ancestor_types[i])
                level_map = nx.get_node_attributes(G, 'level')

                if i == 0:
                    abbrev_list = list(map(lambda x: discipline_abbreviations[x], list(ancestor_map.values())))
                elif i == 1: 
                    abbrev_list = [','.join(list(map(lambda x: discipline_abbreviations[x], ances.split(',')))) for ances in ancestor_map.values()]
                else: raise ValueError('Undefined ancestor type')
                values_ = abbrev_list if discipline_abbrev else list(ancestor_map.values())
                keys, values = list(ancestor_map.keys()), [item[0] + '\n(' + item[1] + ')' for item in np.array([list(ancestor_map.keys()), values_]).T] # '\n'.join(item)
                nx.draw_networkx_nodes(G, pos=positions, node_color = node_colors, node_shape='.', node_size=300)
                nx.draw_networkx_labels(G, pos = positions, labels = dict(np.array([keys, values]).T.tolist()))
                nx.draw_networkx_edges(G, pos = positions, connectionstyle="arc3, rad=0.2", edge_color = edge_colors, style = list(nx.get_edge_attributes(G, edge_styles[i]).values()))
                nx.draw_networkx_edge_labels(G, pos=positions, edge_labels = nx.get_edge_attributes(G, correct_edge_type[i]))
                lines = [Line2D([],[], linewidth=line_width, color=color) for color in ['dodgerblue', 'lime']]
                plt.legend(lines, ['Positive', 'Negative'], loc = 'best')
                plt.title(title[i])
                plt.axis('off')
        else: 
            G = Gs
            plt.figure(figsize=fig_size)
            plt.rcParams["figure.autolayout"] = True
            if layout == 'graphviz':
                positions = graphviz_layout(G, prog='dot') # prog=['neato'|'dot'|'twopi'|'circo'|'fdp'|'nop']
            elif layout == 'spring': positions=nx.spring_layout(G)
            else: raise ValueError('Undefined layout')
            edge_colors, node_colors = np.array([G.edges[u, v]['edge_color'] for u,v in G.edges]), nx.get_node_attributes(G, 'node_color').values()
            ancestor_map = nx.get_node_attributes(G, 'ancestor')
            level_map = nx.get_node_attributes(G, 'level')
            keys, values = list(ancestor_map.keys()), [item[0] + '(' + str(level_map[item[0]]) + ')'+ '\n(' + item[1] + ')' for item in np.array([list(ancestor_map.keys()), list(ancestor_map.values())]).T] # '\n'.join(item)
            # nx.draw_networkx(G, pos=positions, with_labels = False, node_color = node_colors, connectionstyle="arc3, rad=0.2", edge_color = edge_colors, width = line_width)
            nx.draw_networkx_nodes(G, pos=positions, node_color = node_colors, node_shape='.', node_size=300)
            nx.draw_networkx_labels(G, pos = positions, labels = dict(np.array([keys, values]).T.tolist()))
            nx.draw_networkx_edges(G, pos = positions, connectionstyle="arc3, rad=0.2", edge_color = edge_colors, style = list(nx.get_edge_attributes(G, 'edge_style').values()))
            nx.draw_networkx_edge_labels(G, pos=positions, edge_labels = nx.get_edge_attributes(G, 'correct_edge'))
            lines = [Line2D([],[], linewidth=line_width, color=color) for color in ['dodgerblue', 'lime']]
            plt.legend(lines, ['Positive', 'Negative'], loc = 'best')
            plt.title(title)
            plt.axis('off')
        if save: plt.savefig(op.join(Process.embedding_path, 'inference', 'analogy_inference.pdf'), bbox_inches='tight')
        plt.show()

    @staticmethod
    def tsnescatterplot(model, word, list_names, dim, figsize=(8, 8)):
        arrays = np.empty((0, dim), dtype='f')
        word_labels = [word]
        color_list  = ['red']
        arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0) # adds the vector of the query word
        close_words = model.wv.most_similar([word]) # gets list of most similar words
        for wrd_score in close_words: # adds the vector for each of the closest words to the array
            wrd_vector = model.wv.__getitem__([wrd_score[0]])
            word_labels.append(wrd_score[0])
            color_list.append('blue')
            arrays = np.append(arrays, wrd_vector, axis=0)
        for wrd in list_names: # adds the vector for each of the words from list_names to the array
            wrd_vector = model.wv.__getitem__([wrd])
            word_labels.append(wrd)
            color_list.append('green')
            arrays = np.append(arrays, wrd_vector, axis=0)
        reduc = PCA(n_components=10).fit_transform(arrays) # Reduces the dimensionality from 300 to 50 dimensions with PCA
        Y = skTSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
        df = pd.DataFrame({'x': [x for x in Y[:, 0]], 'y': [y for y in Y[:, 1]], 'words': word_labels, 'color': color_list})
        fig = plt.figure(figsize=figsize)
        p1 = sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="o", scatter_kws={'s': 40, 'facecolors': df['color']})
        for line in range(0, df.shape[0]): # Adds annotations one by one with a loop
            p1.text(df["x"][line], df['y'][line], '  ' + df["words"][line].title(),
                    horizontalalignment='left', verticalalignment='bottom', size='medium',
                    color=df['color'][line], weight='normal').set_size(15)
        plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
        plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
        plt.title('t-SNE visualization for {}'.format(word.title()))
        plt.show()
    
    @classmethod
    def count_intersections(self, table_1, table_2, level_i, less_than=False, return_concepts = False, additional_intersection = set()):
        if less_than: table_1_level_i, table_2_level_i = table_1.loc[table_1.level<=level_i], table_2.loc[table_2.level<=level_i]
        else: table_1_level_i, table_2_level_i = table_1.loc[table_1.level==level_i], table_2.loc[table_2.level==level_i]
        concepts_1_l1, concepts_2_l1 = set(table_1_level_i.display_name), set(table_2_level_i.display_name)
        if len(additional_intersection)>0: concepts_1_l1, concepts_2_l1 = concepts_1_l1.intersection(additional_intersection), concepts_2_l1.intersection(additional_intersection)
        intersection = concepts_1_l1.intersection(concepts_2_l1)
        if return_concepts: 
            concept_names_1, concept_names_2, intersection = list(concepts_1_l1-intersection), list(concepts_2_l1-intersection), list(intersection)
            concept_names_1.sort()
            intersection.sort()
            concept_names_2.sort()
            return [concept_names_1, concept_names_2, intersection]
        else: return [len(concepts_1_l1), len(concepts_2_l1), len(intersection)]
    
    @classmethod
    def sub_intersection_df(cls, level_i, less_than=False, thresholds=[20, 2, 'or'], top = -1):
        area_intersection_df_path = op.join(Process.concept_tree_path, 'number_of_area_intersection_df.csv')
        if not op.exists(area_intersection_df_path): 
            from concept import Concept
            table = Concept.level_0_ancestors()
            concepts_0 = table.loc[table.level==0].display_name.drop_duplicates().values
            import itertools
            area_intersection = []
            for area_1, area_2 in itertools.combinations(concepts_0, 2):
                table_1, table_2 = Concept.level_0_ancestors(table, area_1), Concept.level_0_ancestors(table, area_2)
                data_1_2 = [area_1, area_2]
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
        condition_1, condition_2 = (area_intersection_df[columns[0]]>=thresholds[0])&(area_intersection_df[columns[1]]>=thresholds[0]), area_intersection_df[columns[2]]>=thresholds[1]
        condition = (condition_1 & condition_2) if thresholds[2] == 'and' else ((condition_1 | condition_2) if thresholds[2] == 'or' else (condition_1 if thresholds[2] == 'first' else condition_2))
        sub_df = area_intersection_df.loc[condition].sort_values(columns[2], ascending = False).reset_index()
        zero_intersection_index = np.argwhere(sub_df[columns[2]].values==0).flatten()
        top_n = (min(zero_intersection_index[0],top) if top>0 else zero_intersection_index[0]) if len(zero_intersection_index)>0 else (min(sub_df.shape[0], top) if top>0 else sub_df.shape[0])
        return sub_df[['concept_1', 'concept_2']+columns][:top_n]
    
    def get_area_color(self, author_viz_df):
        area_cnt = author_viz_df.area.value_counts()
        result = dict(zip(area_cnt[:10].index, sns.color_palette().as_hex()))
        result.update(dict(zip(area_cnt[10:19].index, sns.color_palette("Set2").as_hex()[:-1])))
        return result

    def plot_author_specter(self, author_viz_df, save=False):
        # author_viz_df, columns = ['PID', 'name', 'area', 'umap_x', 'umap_y', 'color']
        fig = plt.figure(figsize=(11,11))
        area_color = self.get_area_color(author_viz_df)
        sample = author_viz_df.sample(frac=.2)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('off')
        ax.scatter(sample.umap_x, sample.umap_y, color=sample.color, s=.5, alpha=.6)
        top = 8
        for i, k in enumerate(author_viz_df.area.value_counts()[:17].index):
            ax.scatter(19, top-i, s=50, color=area_color[k])
            ax.text(19.5, top-i, k, va='center')
        ax.set(ylim=[sample.umap_y.min(), sample.umap_y.max()])
        if save: plt.savefig('figures/author_specter_umap.jpg', bbox_inches='tight', dpi=240)
    
    def get_selected_concept_embedding(model, table_subset, label_column, params, coordinate_filter = []):
        combine_pca, subset, metric, tsne_lib, dim, path_s, path_e, path_node_ids, no_inter= params
        table_subset_selected = table_subset if len(coordinate_filter) == 0 else table_subset.loc[coordinate_filter]
        all_names, area_labels = table_subset_selected[['display_name', label_column]].values.T.tolist()
        vecs = np.concatenate([model.wv.__getitem__([word]) for word in all_names])
        if combine_pca: vecs = PCA(n_components=int(min(vecs.shape)/2)).fit_transform(vecs)
        file_name_0 = '_subset_' + str(subset) + '_' + metric+ '_' + tsne_lib + '_dim_' + str(dim) + ('_'+'_'.join(np.sort([path_s, path_e]).tolist()).replace(' ', '_') if len(path_node_ids)>0 else '')
        Y_np_file_name = 'Y_np' + file_name_0 + ('' if len(coordinate_filter) == 0 else '_path_region')
        embedding_file_name = 'embedding_map' + file_name_0 + ('_no_inter' if no_inter else '')
        Y_np_path = op.join(Process.embedding_path, 'dists', Y_np_file_name + '.npy')
        Y_csv_path = op.splitext(Y_np_path)[0] + '.csv'
        if not op.exists(Y_np_path) or (not op.exists(Y_csv_path)): # or (not op.exists(Y_csv_path) and subset == 2):
            tsne = opTSNE(perplexity=30, metric=metric, n_jobs=8, random_state=42, verbose=True)
            Y = tsne.fit(vecs)
            Y_df = pd.DataFrame(Y, columns=['x','y'])
            Y_df['label'] = area_labels
            Y_df.to_csv(Y_csv_path, index=False)
            np.save(Y_np_path, Y)
        else: 
            Y = np.load(Y_np_path)
            Y_df = pd.read_csv(Y_csv_path)
        return area_labels,Y,Y_df,embedding_file_name

    @classmethod
    def concept_embedding_plot(cls, n_areas = -1, subset = 2, model_pars = [24, 0, True, 5], level = 1, figsize = (10, 8), combine_pca = False, paths = [], path_distance = [], draw_route_region = False, 
                               colors = ['green', 'red', 'blue'], metric = 'cosine', tsne_lib = 'sk', point_size= 2, label_refined = True, less_than = False, centrality = [], save = False, no_inter=False, emphasize_discipline= None):
        dim, level_vec, multi_level, top_n = model_pars
        model = Embedding.train_model(vec_dim = dim, level = level_vec, multi_level = multi_level)
        model_words = model.wv.index_to_key
        label_column = 'level_0_ancestor_refined' if label_refined else 'level_0_ancestor'
        if len(centrality) > 0: subset = 2
        if n_areas == -1:
            all_concepts = pd.read_csv(op.join(Process.concept_tree_path, 'All_concepts_with_ancestors.csv'))
            table_subset_2 = all_concepts.loc[(all_concepts.works_count>=3433) & (all_concepts.display_name.isin(model_words))][['display_name', 'level_0_ancestor', 'level_0_ancestor_refined', 'multiple_level_0_ancestors']].drop_duplicates(ignore_index = True)
            path_node_names = [table_subset_2.loc[path_id].display_name.values.tolist() for path_id in paths]
            path_ancestors = []
            for concepts in path_node_names:
                path_ancestors += all_concepts.loc[all_concepts.display_name.isin(concepts)].level_0_ancestor_refined.drop_duplicates().values.tolist()
            if subset == 0: 
                table_subset = all_concepts.loc[all_concepts.display_name.isin(model_words)][['display_name', 'level_0_ancestor', 'level_0_ancestor_refined', 'multiple_level_0_ancestors']].drop_duplicates(ignore_index = True)
                if no_inter: 
                    areas = np.sort(all_concepts.level_0_ancestor.drop_duplicates().values)
                    all_concepts_sub = all_concepts.loc[((all_concepts.n_max_ancestors==1) | all_concepts.display_name.isin(areas))]
                    table_subset_no_inter = all_concepts_sub.loc[all_concepts_sub.display_name.isin(model_words)][['display_name', 'level_0_ancestor', 'level_0_ancestor_refined', 'multiple_level_0_ancestors']].drop_duplicates(ignore_index = True)
                    indexs_no_inter = table_subset.loc[table_subset.display_name.isin(table_subset_no_inter.display_name)].index.tolist()
                    table_subset = table_subset_no_inter
                path_node_ids = [[table_subset.loc[table_subset.display_name==name].index[0] for name in path_node_name] for path_node_name in path_node_names]
            elif subset == 1: 
                pass
            elif subset == 2: 
                table_subset = table_subset_2
            else: 
                table = Process.level_0_ancestors().drop(['id'], axis=1).drop_duplicates()
                table_level = table.loc[table.level==level].drop(['level'], axis=1).drop_duplicates() # & table.display_name.isin(model_words)
                table_subset = table_level.loc[table_level.display_name.isin(model_words)][['display_name','level_0_ancestor_refined']].drop_duplicates(ignore_index = True)
            if len(paths)>0:
                if 'Interdiscipline' in path_ancestors: 
                    path_ancestors.remove('Interdiscipline')
                    condition_intercipline = ((table_subset.level_0_ancestor_refined=='Interdiscipline')&(table_subset.multiple_level_0_ancestors.map(lambda s:sum([ances in s for ances in path_ancestors])>1)))
                    table_subset = table_subset.loc[table_subset.level_0_ancestor_refined.isin(path_ancestors) | condition_intercipline | table_subset.display_name.isin(sum(path_node_names, []))]
                else: table_subset = table_subset.loc[table_subset.level_0_ancestor_refined.isin(path_ancestors)]
                table_subset.index = range(table_subset.shape[0]) 
                path_node_ids = [[table_subset.loc[table_subset.display_name==name].index[0] for name in path_node_name] for path_node_name in path_node_names]
                path_s, path_e = path_node_names[0][0], path_node_names[0][-1]
            all_names, area_labels = table_subset[['display_name', label_column]].values.T.tolist()
            vecs = np.concatenate([model.wv.__getitem__([word]) for word in all_names])
            if combine_pca: vecs = PCA(n_components=int(min(vecs.shape)/2)).fit_transform(vecs)
            file_name_0 = '_subset_' + str(subset) + '_' + metric+ '_' + tsne_lib + '_dim_' + str(dim) + ('_'+'_'.join(np.sort([path_s, path_e]).tolist()).replace(' ', '_') if len(path_node_ids)>0 else '')
            Y_np_file_name = 'Y_np' + file_name_0
            embedding_file_name = 'embedding_map' + file_name_0 + ('_no_inter' if no_inter else '')
            Y_np_path = op.join(Process.embedding_path, 'dists', Y_np_file_name + '.npy')
            Y_csv_path = op.splitext(Y_np_path)[0] + '.csv'
            if not op.exists(Y_np_path) or (not op.exists(Y_csv_path) and subset == 2):
                tsne = opTSNE(perplexity=30, metric=metric, n_jobs=8, random_state=42, verbose=True)
                Y = tsne.fit(vecs)
                Y_df = pd.DataFrame(Y, columns=['x','y'])
                Y_df['label'] = area_labels
                Y_df.to_csv(Y_csv_path, index=False)
                np.save(Y_np_path, Y)
            else: 
                Y = np.load(Y_np_path)
                Y_df = pd.read_csv(Y_csv_path)
            if no_inter: Y = Y[indexs_no_inter]
            if draw_route_region:
                coordinate_min = Y[path_node_ids].min(axis=0)
                coordinate_max = Y[path_node_ids].max(axis=0)
                padding = (coordinate_max-coordinate_min)/20
                coordinate_min -= padding
                coordinate_max += padding
                coordinate_filter = ((coordinate_min[0] < Y_df.x) & (Y_df.x < coordinate_max[0])) & ((coordinate_min[1] < Y_df.y) & (Y_df.y < coordinate_max[1]))
                route_region_id = coordinate_filter[coordinate_filter==True].index.tolist()
                Y = Y[route_region_id]
                area_labels = Y_df.loc[route_region_id].label.values.tolist()
                path_node_ids_ = []
                id_map = dict(np.array([route_region_id,list(range(len(route_region_id)))]).T)
                for path_node_id in path_node_ids:
                    path_node_ids_.append([id_map[id_] for id_ in path_node_id])
                path_node_ids = path_node_ids_

            concept_color_map = cls.color_map_emphasize(emphasize_discipline = emphasize_discipline, background_color = "#dcecf5", emphasize_color = None)
            if len(centrality) > 0:
                cls.plot_concepts_embedding(Y, area_labels, paths_with_name = [path_node_ids, path_node_names, path_distance], draw_centers = True, draw_legend = False, figsize = figsize, centrality = centrality, save=save, point_size = point_size)
            else: cls.plot_concepts_embedding(Y, area_labels, paths_with_name = [path_node_ids, path_node_names, path_distance], colors = concept_color_map, draw_centers = True, figsize = figsize, save=save, file_name = embedding_file_name, emphasize = emphasize_discipline != None)
        elif n_areas == 2:
            area_pairs = cls.sub_intersection_df(level_i=level, less_than = less_than, thresholds=[20, 2, 'or'])
            area_1, area_2 = area_pairs[['concept_1','concept_2']].values[0]
            from concept import Concept
            table = Concept.level_0_ancestors()
            for area_1, area_2 in area_pairs[['concept_1','concept_2']].values:
                table_1, table_2 = Concept.level_0_ancestors(table, area_1), Concept.level_0_ancestors(table, area_2) # 计算D(c1) D(c2)
                _, _, intersection = cls.count_intersections(table_1, table_2, level, less_than=False, return_concepts = True, additional_intersection = model_words) # 默认less_than=False
                concept_names_1 = table.loc[table.display_name.isin(set(table_1.display_name).intersection(model_words)) & (table.level_0_ancestor_refined==area_1) & ((table.level<=level) if less_than else (table.level==level))].display_name.drop_duplicates().values.tolist() # 严格属于area_1 + 可分属于area_1 + 默认table.level<=level
                concept_names_2 = table.loc[table.display_name.isin(set(table_2.display_name).intersection(model_words)) & (table.level_0_ancestor_refined==area_2) & ((table.level<=level) if less_than else (table.level==level))].display_name.drop_duplicates().values.tolist() # 严格属于area_2 + 可分属于area_2 + 默认table.level<=level
                concept_names_1_reduced, concept_names_2_reduced = list(set(concept_names_1)-set(intersection)), list(set(concept_names_2)-set(intersection))
                intersection_rest = list(set(intersection)-set(concept_names_1).union(concept_names_2)) #J_rest 所有area_1和area_2的交集 - 上述两部分 = 不属于两者的交集

                schame_a = [sum([concept_names_1, intersection_rest, concept_names_2],[]), sum([[area_1]*len(concept_names_1), ['Interdiscipline']*len(intersection_rest), [area_2]*len(concept_names_2)],[])]
                df_two_schames = pd.DataFrame(schame_a, index=['concept','ancestor_schame_a']).T
                schame_b_map = dict(np.array([sum([concept_names_1_reduced, intersection, concept_names_2_reduced],[]), sum([[area_1]*len(concept_names_1_reduced), ['Interdiscipline']*len(intersection), [area_2]*len(concept_names_2_reduced)],[])]).T)
                df_two_schames['ancestor_schame_b'] = df_two_schames.concept.map(schame_b_map)
                df_two_schames['color_a'] = df_two_schames.ancestor_schame_a.map({area_1:colors[0], 'Interdiscipline':colors[1], area_2:colors[2]})
                df_two_schames['color_b'] = df_two_schames.ancestor_schame_b.map({area_1:colors[0], 'Interdiscipline':colors[1], area_2:colors[2]})
                df_two_schames['color_a_original'] = df_two_schames.ancestor_schame_a.map(LEVEL_0_CONCEPT_COLORS)
                df_two_schames['color_b_original'] = df_two_schames.ancestor_schame_b.map(LEVEL_0_CONCEPT_COLORS)
                
                original_classification_count = df_two_schames.ancestor_schame_a.value_counts()
                if 'Interdiscipline' not in original_classification_count.index: original_classification_count.loc['Interdiscipline'] = 0
                df_J1_J2_count = (original_classification_count-df_two_schames.ancestor_schame_b.value_counts())
                df_J1_J2_count_contributor = df_J1_J2_count.drop('Interdiscipline')
                df_J1_J2_contribution_ratio = (df_J1_J2_count_contributor/df_J1_J2_count_contributor.sum()).round(3)
                df_J1_J2_contribution_ratio.loc['Interdisciplinary'] = -df_J1_J2_count.loc['Interdiscipline']
                original_classification_count.rename(index={'Interdiscipline': 'Interdisciplinary'}, inplace=True)
                value_dicts = [dict(original_classification_count), dict(df_J1_J2_contribution_ratio)]
                
                original_colors = [LEVEL_0_CONCEPT_COLORS[area_1], LEVEL_0_CONCEPT_COLORS['Interdiscipline'], LEVEL_0_CONCEPT_COLORS[area_2]]
                color_lists = df_two_schames[['color_a_original','color_b_original']].values.T.tolist()
                all_names = df_two_schames.concept.values.tolist()
                vecs = np.concatenate([model.wv.__getitem__([word]) for word in all_names])
                if combine_pca: vecs = PCA(n_components=int(min(vecs.shape)/2)).fit_transform(vecs)
                area_names = '_'.join([area_1.replace(' ','_'),area_2.replace(' ','_')]) + ('_lt_level_{}'.format(level) if less_than else '')
                Y_np_file_name = 'Y_np_area_' + area_names + '.npy'
                Y_np_path = op.join(Process.embedding_path, 'dists', Y_np_file_name)
                if not op.exists(Y_np_path):
                    tsne = skTSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate='auto', n_iter=1000, metric=metric, verbose=1, random_state=501, angle=0.5, n_jobs=4)
                    Y = tsne.fit_transform(vecs)
                    np.save(Y_np_path, Y)
                else: Y = np.load(Y_np_path)
                cls.plot_concept_embedding(Y, (figsize[0]*2,figsize[1]), [area_1, area_2], all_names, color_lists, draw_label = False, colors = original_colors, value_dicts = value_dicts, file_name = area_names) # colors = colors
        else: raise ValueError('undefined areas type')

    @classmethod
    def concept_path_embedding_plot(cls, n_areas = -1, subset = 2, model_pars = [24, 0, True, 5], level = 1, figsize = (10, 8), combine_pca = False, paths = [], path_distance = [], draw_route_region = False, 
                               colors = ['green', 'red', 'blue'], metric = 'euclidean', tsne_lib = 'sk', point_size= 2, label_refined = True, less_than = False, centrality = [], save = False, no_inter=False, emphasize_discipline= None):
        dim, level_vec, multi_level, top_n = model_pars
        model = Embedding.train_model(vec_dim = dim, level = level_vec, multi_level = multi_level)
        model_words = model.wv.index_to_key
        label_column = 'level_0_ancestor_refined' if label_refined else 'level_0_ancestor'
        if len(centrality) > 0: subset = 2 # for only subset 2 have shortest paths within each node
        if n_areas == -1:
            all_concepts = pd.read_csv(op.join(Process.concept_tree_path, 'All_concepts_with_ancestors.csv'))
            table_subset_2 = all_concepts.loc[(all_concepts.works_count>=3433) & (all_concepts.display_name.isin(model_words))][['display_name', 'level_0_ancestor', 'level_0_ancestor_refined', 'multiple_level_0_ancestors']].drop_duplicates(ignore_index = True)
            path_node_names = [table_subset_2.loc[path_id].display_name.values.tolist() for path_id in paths]
            path_ancestors = []
            for concepts in path_node_names:
                path_ancestors += all_concepts.loc[all_concepts.display_name.isin(concepts)].level_0_ancestor_refined.drop_duplicates().values.tolist()
            if subset == 0: 
                table_subset = all_concepts.loc[all_concepts.display_name.isin(model_words)][['display_name', 'level_0_ancestor', 'level_0_ancestor_refined', 'multiple_level_0_ancestors']].drop_duplicates(ignore_index = True)
                if no_inter: 
                    areas = np.sort(all_concepts.level_0_ancestor.drop_duplicates().values)
                    all_concepts_sub = all_concepts.loc[((all_concepts.n_max_ancestors==1) | all_concepts.display_name.isin(areas))]
                    table_subset_no_inter = all_concepts_sub.loc[all_concepts_sub.display_name.isin(model_words)][['display_name', 'level_0_ancestor', 'level_0_ancestor_refined', 'multiple_level_0_ancestors']].drop_duplicates(ignore_index = True)
                    indexs_no_inter = table_subset.loc[table_subset.display_name.isin(table_subset_no_inter.display_name)].index.tolist()
                    table_subset = table_subset_no_inter
                path_node_ids = [[table_subset.loc[table_subset.display_name==name].index[0] for name in path_node_name] for path_node_name in path_node_names]
            elif subset == 1: 
                pass
            elif subset == 2: 
                table_subset = table_subset_2
            else: 
                table = Process.level_0_ancestors().drop(['id'], axis=1).drop_duplicates()
                table_level = table.loc[table.level==level].drop(['level'], axis=1).drop_duplicates() # & table.display_name.isin(model_words)
                table_subset = table_level.loc[table_level.display_name.isin(model_words)][['display_name','level_0_ancestor_refined']].drop_duplicates(ignore_index = True)
            if len(paths)>0:
                if 'Interdiscipline' in path_ancestors: 
                    path_ancestors.remove('Interdiscipline')
                    condition_intercipline = ((table_subset.level_0_ancestor_refined=='Interdiscipline')&(table_subset.multiple_level_0_ancestors.map(lambda s:sum([ances in s for ances in path_ancestors])>1)))
                    table_subset = table_subset.loc[table_subset.level_0_ancestor_refined.isin(path_ancestors) | condition_intercipline | table_subset.display_name.isin(sum(path_node_names, []))]
                else: table_subset = table_subset.loc[table_subset.level_0_ancestor_refined.isin(path_ancestors)]
                table_subset.index = range(table_subset.shape[0]) 
                path_node_ids = [[table_subset.loc[table_subset.display_name==name].index[0] for name in path_node_name] for path_node_name in path_node_names]
                path_s, path_e = path_node_names[0][0], path_node_names[0][-1]
            params = [combine_pca, subset, metric, tsne_lib, dim, path_s, path_e, path_node_ids, no_inter]
            area_labels,Y,Y_df,embedding_file_name = cls.get_selected_concept_embedding(model, table_subset, label_column, params)
            if no_inter: Y = Y[indexs_no_inter]
            if draw_route_region:
                coordinate_min = Y[path_node_ids].min(axis=0)
                coordinate_max = Y[path_node_ids].max(axis=0)
                padding = (coordinate_max-coordinate_min)/20
                coordinate_min -= padding
                coordinate_max += padding
                coordinate_filter = ((coordinate_min[0] < Y_df.x) & (Y_df.x < coordinate_max[0])) & ((coordinate_min[1] < Y_df.y) & (Y_df.y < coordinate_max[1]))

                route_region_id = coordinate_filter[coordinate_filter==True].index.tolist()
                Y = Y[coordinate_filter]
                area_labels = np.array(area_labels)[route_region_id].tolist()
                path_node_ids_ = []
                id_map = dict(np.array([route_region_id,list(range(len(route_region_id)))]).T)
                for path_node_id in path_node_ids:
                    path_node_ids_.append([id_map[id_] for id_ in path_node_id])
                path_node_ids = path_node_ids_

            concept_color_map = cls.color_map_emphasize(emphasize_discipline = emphasize_discipline, background_color = "#dcecf5", emphasize_color = None)
            if len(centrality) > 0:
                cls.plot_concepts_path_embedding(Y, area_labels, paths_with_name = [path_node_ids, path_node_names, path_distance], draw_centers = True, draw_legend = False, figsize = figsize, centrality = centrality, save=save, point_size = point_size)
            else: cls.plot_concepts_path_embedding(Y, area_labels, paths_with_name = [path_node_ids, path_node_names, path_distance], colors = concept_color_map, draw_centers = True, figsize = figsize, save=save, file_name = embedding_file_name, emphasize = emphasize_discipline != None)
        elif n_areas == 2: # ['Mathematics', 'Computer science']
            area_pairs = cls.sub_intersection_df(level_i=level, less_than = less_than, thresholds=[20, 2, 'or'])
            area_1, area_2 = area_pairs[['concept_1','concept_2']].values[0]
            from concept import Concept
            table = Concept.level_0_ancestors()
            for area_1, area_2 in area_pairs[['concept_1','concept_2']].values:
                table_1, table_2 = Concept.level_0_ancestors(table, area_1), Concept.level_0_ancestors(table, area_2)
                _, _, intersection = cls.count_intersections(table_1, table_2, level, less_than=False, return_concepts = True, additional_intersection = model_words)
                concept_names_1 = table.loc[table.display_name.isin(model_words) & (table.level_0_ancestor_refined==area_1) & (table.level==level)].display_name.drop_duplicates().values.tolist() # 严格属于area_1 + 可分属于area_1
                concept_names_2 = table.loc[table.display_name.isin(model_words) & (table.level_0_ancestor_refined==area_2) & (table.level==level)].display_name.drop_duplicates().values.tolist() # 严格属于area_2 + 可分属于area_2
                concept_names_1_copy, concept_names_2_copy, intersection_copy = concept_names_1.copy(), concept_names_2.copy(), intersection.copy()

                intersection = list(set(intersection)-set(concept_names_1).union(concept_names_2))  # 所有area_1和area_2的交集 - 上述两部分 = 不属于两者的交集
                concept_two_df = pd.DataFrame([[name, colors[0]] for name in concept_names_1] + [[name, colors[1]] for name in intersection] + [[name, colors[2]] for name in concept_names_2], columns=['concept','color_1'])

                concept_names_1_copy, concept_names_2_copy = list(set(concept_names_1_copy)-set(intersection_copy)), list(set(concept_names_2_copy)-set(intersection_copy))
                concept_two_df['color_2'] = concept_two_df.concept.map(dict([[name,  colors[0]] for name in concept_names_1_copy] + [[name,  colors[1]] for name in intersection_copy] + [[name,  colors[2]] for name in concept_names_2_copy]))
                
                color_lists = concept_two_df[['color_1','color_2']].values.T.tolist()
                all_names = concept_two_df.concept.values.tolist()
                vecs = np.concatenate([model.wv.__getitem__([word]) for word in all_names])
                if combine_pca: vecs = PCA(n_components=int(min(vecs.shape)/2)).fit_transform(vecs)
                Y = opTSNE(perplexity=30, metric='cosine', n_iter=500, n_jobs = 8, random_state=42, verbose=False).fit(vecs) # opTSNE 
                cls.plot_concepts_path_embedding(Y, (figsize[0]*2,figsize[1]), [area_1, area_2], all_names, color_lists, draw_label = False, colors = colors) 
        else: raise ValueError('undefined areas type')

    @staticmethod
    def plot_concept_embedding(Y, figsize, areas, all_names, color_lists, multiline = False, draw_label = False, colors = [], value_dicts = [], file_name = ''):
        if multiline: all_names = [name.replace('-','_').replace(' ','\n') for name in all_names]
        plt.figure(figsize=figsize)
        if len(color_lists) == 2 and len(value_dicts) == 2 and len(all_names)>2:
            figtype = ['discipline', 'intersection']
            for i in range(len(color_lists)):
                dict_i = value_dicts[i]
                plt.subplot(1,len(color_lists),i+1)
                df = pd.DataFrame({'x': [x for x in Y[:, 0]], 'y': [y for y in Y[:, 1]], 'words': all_names, 'color': color_lists[i]})
                if i==0: labels = ['{}({})'.format(areas[0],dict_i[areas[0]]), '{}({})'.format('Interdisciplinary', dict_i['Interdisciplinary']), '{}({})'.format(areas[1],dict_i[areas[1]])]
                elif i==1: labels = ["{}({:.1%})".format(areas[0],dict_i[areas[0]]), "{}(+{})".format('Interdisciplinary', int(dict_i['Interdisciplinary'])), "{}({:.1%})".format(areas[1],dict_i[areas[1]])]
                for k in range(len(colors)):
                    df_k = df.loc[df.color==colors[k]]
                    sns.regplot(data=df_k, x="x", y="y", marker="o", fit_reg=False, scatter_kws={'s': 30, 'facecolors': df_k['color'], 'edgecolors': 'none'}, label = labels[k])
                plt.xlim(Y[:, 0].min()-1, Y[:, 0].max()+1)
                plt.ylim(Y[:, 1].min()-1, Y[:, 1].max()+1)
                plt.axis('off')
                plt.legend(fontsize='large') # loc='upper right'
        else:
            df = pd.DataFrame({'x': [x for x in Y[:, 0]], 'y': [y for y in Y[:, 1]], 'words': all_names, 'color': color_lists})
            p1 = sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="o", scatter_kws={'s': 40, 'facecolors': df['color']})
            if draw_label:
                for line in range(0, df.shape[0]): # Adds annotations one by one with a loop
                    p1.text(df["x"][line], df['y'][line], df["words"][line].title(), horizontalalignment='center', verticalalignment='top', size='medium', color=df['color'][line], weight='normal').set_size(15)
            plt.xlim(Y[:, 0].min()-1, Y[:, 0].max()+1)
            plt.ylim(Y[:, 1].min()-1, Y[:, 1].max()+1)
            plt.title('t-SNE visualization for {} and {}'.format(*areas))
        
        save_file_path = op.join(Process.embedding_path, '2_area_map', 'level_lt_3', file_name+'_map.png')
        plt.savefig(save_file_path, dpi=400, bbox_inches='tight')
        # plt.show()

    def get_continuous_cmap(hex_list, float_list=None):
        # https://github.com/KerryHalupka/custom_colormap/blob/master/generate_colormap.py
        import matplotlib.colors as mcolors
        def hex_to_rgb(value):
            '''
            Converts hex to rgb colours
            value: string of 6 characters representing a hex colour.
            Returns: list length 3 of RGB values'''
            value = value.strip("#") # removes hash symbol if present
            lv = len(value)
            return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        def rgb_to_dec(value):
            '''
            Converts rgb to decimal colours (i.e. divides each value by 256)
            value: list (length 3) of RGB values
            Returns: list (length 3) of decimal values'''
            return [v/256 for v in value]
        rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
        if float_list: pass
        else: float_list = list(np.linspace(0,1,len(rgb_list)))
        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp
    
    @classmethod
    def plot_concepts_embedding_simple(cls, x, y, paths_with_name = [], ax=None, title=None, draw_legend=True, draw_centers=False, draw_cluster_labels=False, 
        colors=None, legend_kwargs=None, label_order=None, figsize=(8, 8), save = False, file_name = None, centrality = [], **kwargs):
        pass
    @classmethod
    def plot_concepts_embedding(cls, x, y, paths_with_name = [], ax=None, title=None, draw_legend=True, draw_centers=False, draw_cluster_labels=False, 
        colors=None, legend_kwargs=None, label_order=None, figsize=(8, 8), save = False, file_name = None, centrality = [], point_size=100, emphasize = False, **kwargs):
        
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        # https://www.color-hex.com/
        import matplotlib
        if len(centrality) > 0:
            centrality_min, centrality_max = np.exp(centrality+1).min(), np.exp(centrality+1).max()
            centrality = (np.exp(centrality+1) - centrality_min) / (centrality_max - centrality_min)
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        # if title is not None: ax.set_title(title)
        plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}
        # Create main plot
        if label_order is not None:
            assert all(np.isin(np.unique(y), label_order))
            classes = [l for l in label_order if l in np.unique(y)]
        else: classes = np.unique(y)
        if colors is None:
            default_colors = matplotlib.rcParams["axes.prop_cycle"]
            colors = {k: v["color"] for k, v in zip(classes, default_colors())}
        
        point_colors = list(map(colors.get, y))
        if len(centrality) > 0:
            sc = ax.scatter(x[:, 0], x[:, 1], c=centrality, cmap='rainbow', rasterized=True, **plot_params)
            fig.colorbar(sc, ax=ax, location='left') 
        else: ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)
        
        import matplotlib.patches as patches
        path_arrow_colors = ['green','blue','black','red','gray']
        if len(paths_with_name) == 2:
            paths, path_node_names = paths_with_name
            print(paths, path_node_names)
            for path, names, arrow_color in zip(paths, path_node_names, path_arrow_colors[:len(paths)]):
                path_x_y = x[path] 
                for i in range(len(path)-1):
                    plt.gca().add_patch(patches.FancyArrowPatch(path_x_y[i], path_x_y[i+1], connectionstyle="arc3,rad=.5",color=arrow_color))
                    plt.text(path_x_y[i,0], path_x_y[i,1], str(i)+'-'+names[i] + '('+y[i]+')', color = colors[y[i]])
                plt.text(path_x_y[i+1,0], path_x_y[i+1,1], str(i+1)+'-'+names[i+1] + '('+y[i+1]+')', color = colors[y[i+1]])

        if draw_legend:
            legend_handles = [matplotlib.lines.Line2D([], [], marker="s", color="w", markerfacecolor=colors[yi], ms=10, alpha=1, linewidth=0, label=yi, markeredgecolor="k", ) for yi in classes]
            legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(0.96, 0.5), frameon=True, )
            if legend_kwargs is not None: legend_kwargs_.update(legend_kwargs)
            ax.legend(handles=legend_handles, **legend_kwargs_)

        if draw_centers:
            centers = []
            for yi in classes:
                mask = yi == np.array(y)
                centers.append(np.median(x[mask, :2], axis=0))
            centers = np.array(centers)
            colors_center = LEVEL_0_CONCEPT_COLORS.copy() if emphasize else colors
            center_colors = list(map(colors_center.get, classes))
            ax.scatter(centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k")

            if draw_cluster_labels:
                for idx, label in enumerate(classes):
                    ax.text(centers[idx, 0], centers[idx, 1] + 2.2, label, fontsize=kwargs.get("fontsize", 6), horizontalalignment="center")

        ax.set(xlim=[x[:, 0].min(), 1.1*x[:, 0].max()])
        ax.set(ylim=[x[:, 1].min(), x[:, 1].max()])
        ax.axis("off")
        if save: 
            save_file_path = op.join(Process.embedding_path, file_name+'.png')
            plt.savefig(save_file_path, dpi=400, bbox_inches='tight')
            print(save_file_path,'cached')
        plt.show()

    @classmethod
    def plot_concepts_path_embedding(cls, x, y, paths_with_name = [], ax=None, title=None, draw_legend=True, draw_centers=False, draw_cluster_labels=False, 
        colors=None, legend_kwargs=None, label_order=None, figsize=(8, 8), save = False, file_name = None, centrality = [], point_size=100, emphasize = False, **kwargs):
        
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        # https://www.color-hex.com/

        import matplotlib
        if len(centrality) > 0:
            centrality_type = 'betweenness' if max(centrality)>1000 else 'closeness'
            file_name = 'concept_embedding_' + centrality_type + '_centrality'
            from sklearn.preprocessing import quantile_transform
            centrality = np.log(centrality)
            centrality_min, centrality_max = centrality.min(), centrality.max()
            centrality = (centrality-centrality_min) / (centrality_max - centrality_min)
            centrality = quantile_transform(centrality.reshape(-1, 1), output_distribution='uniform').flatten()
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        plot_params = {"alpha": kwargs.get("alpha", 0.08), "s": kwargs.get("s", point_size)}
        if label_order is not None:
            assert all(np.isin(np.unique(y), label_order))
            classes = [l for l in label_order if l in np.unique(y)]
        else: classes = np.unique(y)
        if colors is None:
            default_colors = matplotlib.rcParams["axes.prop_cycle"]
            colors = {k: v["color"] for k, v in zip(classes, default_colors())}

        hex_list = ['#a52a2a', '#f6546a', '#daa520', '#f0f8ff', '#c6e2ff', '#391285']
        _cmap = cls.get_continuous_cmap(hex_list) 
        point_colors = list(map(colors.get, y))
        if len(centrality) > 0:
            sc = ax.scatter(x=x[:, 0], y=x[:, 1], c=centrality, cmap='turbo', rasterized=True, **plot_params)#'coolwarm'
            fig.colorbar(sc, ax=ax, location='left') # cax = fig.add_axes([0.1, 0.1, 0.03, 0.8])
        else: ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params) # c= LEVEL_0_CONCEPT_COLORS['Background']
        
        import matplotlib.patches as patches
        path_arrow_colors = ['red','green','black','blue','gray']
        if (len(paths_with_name) == 3) & (len(sum(paths_with_name, []))>0):
            height = x[:,1].max()-x[:,1].min()
            width = x[:,0].max()-x[:,0].min()
            paths, path_node_names, distances = paths_with_name
            print(paths, path_node_names)
            path_number_size = 22
            for path, names, arrow_color in zip(paths, path_node_names, path_arrow_colors[:len(paths)]):
                file_name = 'path_'+'_'.join(np.sort(np.array([names[0], names[-1]]))).replace(' ', '_')
                path_x_y = x[path] 
                ances_path = np.array(y)[path]
                for i in range(len(path)-1):
                    if i == 0: text_x_shift,text_y_shift = -1.5,-1.2
                    elif i>2:text_x_shift,text_y_shift = -0.7,2.2
                    else:text_x_shift,text_y_shift = -1, 2
                    plt.arrow(path_x_y[i,0], path_x_y[i,1], path_x_y[i+1,0]-path_x_y[i,0], path_x_y[i+1,1]-path_x_y[i,1], color = arrow_color)
                    plt.text(path_x_y[i,0]+text_x_shift, path_x_y[i,1]+text_y_shift, str(i+1), fontsize=path_number_size, color = 'black', fontstyle='italic')
                plt.text(path_x_y[i+1, 0]+text_x_shift, path_x_y[i+1, 1]+text_y_shift, str(i+2), fontsize=path_number_size, color = 'black', fontstyle='italic')
                path_colors = list(map(colors.get, ances_path))
                ax.scatter(path_x_y[:, 0], path_x_y[:, 1], c=path_colors, s=250, alpha=1, edgecolor="k")

            text_size = 18
            for i in range(len(names)):
                plt.text(x[:,0].max()+width/35, x[:,1].max() - height/2.8 - height/3*(i/len(names)), str(i+1)+'-'+names[i], fontsize=text_size, color = colors[ances_path[i]], fontstyle='italic')
            
            distance_type = ['distance', 'global: ','cosine: ']
            distance_type_colors = ['black', 'red','blue']

            if distances[0] == distances[1]:
                for i in range(1,3):
                    plt.text(x[:,0].max()+width/25, x[:,1].min()+height/25 + height/3*(i/len(names)), distance_type[i]+str(round(distances[i],6)), fontsize=text_size, color = distance_type_colors[i], fontstyle='italic')
                plt.text(x[:,0].max()+width/25, x[:,1].min()+height/25 + height/3*((i+1)/len(names)), distance_type[0], fontsize=text_size, color = distance_type_colors[0], fontstyle='italic')
            else: raise ValueError('distance not valid')
        
        if draw_legend:
            legend_handles = [matplotlib.lines.Line2D([], [], marker="s", color="w", markerfacecolor=colors[yi], ms=12, alpha=1, linewidth=0, label=str(list(LEVEL_0_CONCEPT_COLORS.keys()).index(yi)+1) + ' ' + yi, markeredgecolor="k", ) for index, yi in enumerate(classes)]
            legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(0.95, 0.8), frameon=False, ) # frameon默认未True
            if legend_kwargs is not None: legend_kwargs_.update(legend_kwargs)
            ax.legend(handles=legend_handles, **legend_kwargs_)

        if draw_centers:
            centers = []
            for yi in classes:
                mask = yi == np.array(y)
                centers.append(np.median(x[mask, :2], axis=0))
            centers = np.array(centers)
            colors_center = LEVEL_0_CONCEPT_COLORS.copy() if emphasize else colors
            center_colors = list(map(colors_center.get, classes))
            if len(sum(paths_with_name, [])) == 0:
                ax.scatter(centers[:, 0], centers[:, 1], c=center_colors, s=24, alpha=1, edgecolor="k") # original s=48

            if draw_cluster_labels:
                for idx, label in enumerate(classes):
                    ax.text(centers[idx, 0], centers[idx, 1] + 2.2, label, fontsize=kwargs.get("fontsize", 6), horizontalalignment="center")

        ax.set(xlim=[x[:, 0].min(), 1.1*x[:, 0].max()])
        ax.set(ylim=[x[:, 1].min(), x[:, 1].max()])
        ax.axis("off")
        if save: 
            plt.savefig(op.join(Process.external_data_dir, file_name + ('_emphasize' if emphasize else '') +'.png'), dpi=400, bbox_inches='tight')
        plt.show()

# best
# upper right
# upper left
# lower left
# lower right
# right
# center left
# center right
# lower center
# upper center
# center

