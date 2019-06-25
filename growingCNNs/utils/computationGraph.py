"""
    Helper class to wrap a list of edges together with the index of the input
    node and the index of the output node.
"""
class ComputationGraph:

    def __init__(self, edges, inputIndex, outputIndex):

        self.edges = edges
        self.inputIndex = inputIndex
        self.outputIndex = outputIndex

        self.nodes = []
        for start, end in self.edges:
            for node in [start, end]:
                if node not in self.nodes:
                    self.nodes.append(node)
        self.numNodes = len(self.nodes)

    def clone(self):

        cloneGraph = ComputationGraph(
                list(self.edges),
                self.inputIndex,
                self.outputIndex
        )

        return cloneGraph
