import torch
import networkx as nx
import pandas as pd

def to_csr(graph):
    csr = nx.to_scipy_sparse_matrix(graph,format='csr')    
    row_ptr = torch.Tensor(csr.indptr).to(int).contiguous()
    col_idx = torch.Tensor(csr.indices).to(int).contiguous()
    return row_ptr, col_idx

def nodes_tensor(graph):
    nodes = list(graph.nodes())
    nodes_index = []
    for node in nodes:
        nodes_index.append(nodes.index(node))

    nodes_t = torch.LongTensor(nodes_index).contiguous()
    return nodes_t


def to_edge_list_indexed(graph):
    edges = list(graph.edges())
    nodes = sorted(list(graph.nodes()))
    node_index_mapping = {}
    is_directed = nx.is_directed(graph)

    edge_list_len = len(edges)

    edge_list_indexed = torch.zeros((edge_list_len,2)).to(int).contiguous()

    for index, edge in enumerate(edges):
        head = edge[0]
        tail = edge[1]

        head_index = -1
        if head in node_index_mapping:
            head_index = node_index_mapping[head]
        else:
            head_index = nodes.index(head)
            node_index_mapping[head] = head_index

        tail_index = -1
        if tail in node_index_mapping:
            tail_index = node_index_mapping[tail]
        else:
            tail_index = nodes.index(tail)
            node_index_mapping[tail] = tail_index

        edge_list_indexed[index][0] = head_index
        edge_list_indexed[index][1] = tail_index

    if is_directed == False:
        edge_list_reversed = torch.fliplr(edge_list_indexed)
        edge_list_indexed = torch.cat((edge_list_indexed,edge_list_reversed),dim=0)

    return edge_list_indexed, node_index_mapping

def build_node_edge_index(edge_list_indexed, nodes_tensor):

    edge_list_indexed_pd = pd.DataFrame(data=edge_list_indexed.numpy(),columns=["head","tail"])
    edge_list_indexed_np = edge_list_indexed_pd.sort_values(by="head",ascending=True).to_numpy()
    edge_list_indexed = torch.from_numpy(edge_list_indexed_np).contiguous()

    nodes_unique = torch.unique(nodes_tensor)
    nodes_sorted,_ = torch.sort(nodes_unique)

    num_nodes = len(nodes_sorted)
    num_edges = len(edge_list_indexed)
    node_edge_index = torch.full((num_nodes,2),-1).to(int).contiguous()


    current_node = edge_list_indexed[0][0]
    for edge_index in range(num_edges):
        edge = edge_list_indexed[edge_index]
        head = edge[0]

        if head != current_node:
            node_edge_index[current_node][1] = edge_index - 1
            node_edge_index[head] = edge_index
            current_node = head
        else:
            if edge_index == 0:
                node_edge_index[head][0] = 0
            else:
                node_edge_index[head][1] = edge_index
    
    return node_edge_index, edge_list_indexed

def build_relation_tail_index(triples_indexed_tensor,all_entities_tensor):

    triples_indexed_pd = pd.DataFrame(data=triples_indexed_tensor.numpy(),columns=["head","relation","tail"])
    triples_indexed_pd = triples_indexed_pd.sort_values(by="head",ascending=True)    
    triples_indexed_tensor = torch.from_numpy(triples_indexed_pd.to_numpy()).to(int).contiguous()
    nodes_sorted,_ = torch.sort(all_entities_tensor)

    num_nodes = len(nodes_sorted)
    num_edges = len(triples_indexed_tensor)
    node_edge_index = torch.full((num_nodes,2),-1).to(int).contiguous()

    current_node = triples_indexed_tensor[0][0]
    for edge_index in range(num_edges):
        edge = triples_indexed_tensor[edge_index]
        head = edge[0]

        if head != current_node:
            node_edge_index[current_node][1] = edge_index - 1
            node_edge_index[head] = edge_index
            current_node = head
        else:
            if edge_index == 0:
                node_edge_index[head][0] = 0
            else:
                node_edge_index[head][1] = edge_index
    
    return node_edge_index, triples_indexed_tensor
            

            


