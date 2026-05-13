import json
import networkx as nx
from pyvis.network import Network

# 1. 读取你的数据
with open("MC1_graph.json", "r") as f:
    data = json.load(f)

# 2. 转成 networkx 图
G = nx.node_link_graph(data)

# 3. （关键）抽样，不然太大画不出来
# 随机取一小部分节点
sample_nodes = list(G.nodes())[:200]   # 你可以改成 300 / 500
H = G.subgraph(sample_nodes)

# 4. 创建可视化网络
net = Network(height="800px", width="100%", directed=True)

# 5. 添加节点（按类型上色）
for node, attr in H.nodes(data=True):
    node_type = attr.get("Node Type", "Unknown")

    color_map = {
        "Person": "orange",
        "Song": "blue",
        "Album": "purple",
        "MusicalGroup": "green",
        "RecordLabel": "red"
    }

    net.add_node(
        node,
        label=attr.get("name", attr.get("stage_name", str(node))),
        color=color_map.get(node_type, "gray"),
        title=str(attr)  # 鼠标悬停可以看到信息
    )

# 6. 添加边（标注关系类型）
for u, v, attr in H.edges(data=True):
    edge_type = attr.get("type", "")

    net.add_edge(
        u, v,
        label=edge_type,
        title=edge_type
    )

# 7. 生成 HTML
net.show("music_graph.html", notebook=False)