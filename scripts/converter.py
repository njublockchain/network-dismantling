import networkx as nx
import graph_tool.all as gt
import lzma

def gt2nx(gt_graph: gt.Graph) -> nx.MultiDiGraph | nx.MultiGraph:
    """
    Convert a graph-tool graph to a networkx graph.
    """
    is_directed = gt_graph.is_directed()
    is_multi = True
    nx_graph = nx.MultiDiGraph() if is_directed else nx.MultiGraph()

    for edge in gt_graph.edges():
        index = int(gt_graph.edge_index[edge])
        nx_graph.add_edge(int(edge.source()), int(edge.target()), key=index)
        # assign edge attributes
        nx_graph.edges[int(edge.source()), int(edge.target()), index].update(gt_graph.ep)

    # assign node attributes
    for v in gt_graph.vertices():
        v = int(v)
        nx_graph.nodes[v].update(gt_graph.vp)
    
    return nx_graph

def nx2gt(nx_graph: nx.Graph) -> gt.Graph:
    """
    Convert a networkx graph to a graph-tool graph.
    """
    gt_graph = gt.Graph(directed=False)
    for u, v in nx_graph.edges():
        gt_graph.add_edge(u, v)

    # assign node attributes
    for v in nx_graph.nodes():
        gt_graph.vp.update(nx_graph.nodes[v])
    
    return gt_graph

if __name__ == '__main__':
    import argparse
    import pickle
    import os
    parser = argparse.ArgumentParser(description='Converter')
    parser.add_argument('--input', type=str, help='Input file')
    args = parser.parse_args()

    def convert_file(input_file: str):
        if input_file.endswith('.gt'):
            output_file = input_file.replace('.gt', '.nx.pkl.xz')
            gt_graph = gt.load_graph(input_file)
            nx_graph = gt2nx(gt_graph)
            pickle.dump(nx_graph, lzma.open(output_file, 'wb'))
        elif input_file.endswith('.gt.xz'):
            output_file = input_file.replace('.gt.xz', '.nx.pkl.xz')
            gt_graph = gt.load_graph(input_file)
            nx_graph = gt2nx(gt_graph)
            pickle.dump(nx_graph, lzma.open(output_file, 'wb'))
        elif input_file.endswith('.nx.pkl'):
            output_file = input_file.replace('.nx.pkl', '.gt.xz')
            nx_graph = pickle.load(open(input_file, 'rb'))
            gt_graph = nx2gt(nx_graph)
            gt_graph.save(output_file)
        elif input_file.endswith("nx.pkl.xz"):
            output_file = input_file.replace('.nx.pkl.xz', '.gt.xz')
            nx_graph = pickle.load(lzma.open(input_file, 'rb'))
            gt_graph = nx2gt(nx_graph)
            gt_graph.save(output_file)

    if os.path.isdir(args.input):
        for file in os.listdir(args.input):
            input_file = os.path.join(args.input, file)
            convert_file(input_file)
    elif os.path.isfile(args.input):
        convert_file(args.input)
    else:
        raise ValueError('Invalid input file or directory')
