from graph import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import torch
import copy


import sys
 
# setting path
sys.path.append('..')
sys.path.append('.')

from kslt import data

normalize = sys.argv[2]

if len(sys.argv) >= 4:
    hand_only = True
    body_part = sys.argv[3]
else:
    hand_only = False
    body_part = ""

if normalize == 'original':
    info_type = '/data/sl_datasets/wlasl/wlasl100_lmdb_train_original'
elif normalize == 'normalize':
    info_type = '/data/sl_datasets/wlasl/wlasl100_lmdb_train_normalize_neo_54_hand_0_0.5'
elif normalize == 'old_normalize':
    info_type = '/data/sl_datasets/wlasl/wlasl100_lmdb_train_normalize_0.5'
elif normalize == 'new_neo_normalize':
    info_type = '/data/sl_datasets/wlasl/wlasl100_lmdb_test_normalize_new_neo_54_hand_0_0.5'
elif normalize == 'rotate_13':
    info_type = '/data/sl_datasets/wlasl/wlasl100_lmdb_train_rotate_normalize_neo_hand_0_0.5__40'
elif normalize == 'noise_161':
    info_type = '/data/sl_datasets/wlasl/wlasl100_lmdb_train_gaussian_54_normalize_neo_hand_0_0.5__0.01_0.06_0.01'
elif normalize == 'noise_116':
    info_type = '/data/sl_datasets/wlasl/wlasl100_lmdb_train_gaussian_54_normalize_neo_hand_0_0.5__0.01_0.01_0.06'
elif normalize == 'noise_211':
    info_type = '/data/sl_datasets/wlasl/wlasl100_lmdb_train_gaussian0.20.1_0.1'
elif normalize == 'flip':
    info_type = '/data/sl_datasets/wlasl/wlasl100_lmdb_train_flip_54_normalize_neo_hand_0_0.5__1_1_1'

r = data.WlaSLDataset(
        db_path = info_type, seq_len = -1, min_seq_len = -1, normalize = False, padding = False, sub_sample = False, normalize_hand = False
)



index = int(sys.argv[1])

tgt = r[index]['keypoint']

if tgt.shape[2] == 54:
    temp = torch.zeros(67, 2)
    temp[broadcast_54_67, :] = tgt
    tgt = temp

fig, ax = plt.subplots()

G = nx.Graph()


for i in range(25):
    G.add_node(i)
        
for (i, j) in POSE_MAP:
    G.add_edge(i, j)

for (i, j) in HAND_MAP:
    G.add_edge(i+25, j+25)

for (i, j) in HAND_MAP:
    G.add_edge(i+46, j+46)

start = 0
end = 67

if hand_only:
    if sys.argv[3] == "Remove_Body":
        G.remove_nodes_from(list(range(8, 15)))
        G.remove_nodes_from(list(range(19, 25)))
        node_list = list(range(0, 8)) + list(range(15, 19)) + list(range(25, 67))
    if sys.argv[3] == "Left":
        G.remove_nodes_from(list(range(46)))
        node_list = list(range(46, 67))
        start = 46
        end = 67
    elif sys.argv[3] == "Right":
        G.remove_nodes_from(list(range(46, 67)))
        G.remove_nodes_from(list(range(25)))
        start = 25
        end = 46
    elif sys.argv[3] == "Body":
        G.remove_nodes_from(list(range(8, 15)))
        G.remove_nodes_from(list(range(19, 25)))
        G.remove_nodes_from(list(range(25, 67)))
        start = 0
        end = 12
def draw_graph(i, G, ax):
    ax.clear()
    ax.invert_yaxis()
    # Add nodes to the graph
    ax.set_title(normalize+body_part+"_"+r[index]['gloss'] +  "__" + str(i))
    positions = {}
    for j in list(G.nodes):
        positions[j] = tuple(tgt[i, j, :])
    nx.draw(G, positions, ax, node_color=range(start, end), with_labels=True, node_size=9, font_size=7)

if False:
    ani = animation.FuncAnimation(fig, draw_graph, frames=tgt.shape[0], fargs=(G, ax))

    gif_path = "/home/ajkim/kslt/slr/gifs/"+normalize+body_part+"_"+str(index)+".gif"
    ani.save(gif_path, writer='pillow', fps=8)
else:
    draw_graph(40, G, ax)
    plt.savefig("/home/ajkim/kslt/slr/gifs/"+normalize+body_part+"_"+str(index)+".png")