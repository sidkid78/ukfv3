class KnowledgeGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = []  # list of (source, relation, target)

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, source, relation, target):
        self.edges.append((source, relation, target))
        self.nodes.update([source, target])

    def get_nodes(self):
        return list(self.nodes)

    def get_edges(self):
        return self.edges
