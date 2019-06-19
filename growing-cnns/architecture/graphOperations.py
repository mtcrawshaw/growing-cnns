# This is in case no parent packages are imported, such as in the test cases
try:
    from .computationGraph import ComputationGraph
except:
    from computationGraph import ComputationGraph


"""
    Builds a computation graph which is a sequence of nodes,
    each node has input degree one and output degree one.
"""
def getInitialCompGraph(numNodes):

    edges = [(i, i + 1) for i in range(numNodes - 1)]
    inputIndex = 0
    outputIndex = numNodes - 1
    return ComputationGraph(edges, inputIndex, outputIndex)


"""
    From a given computation graph, builds a returns a "grown" version,
    where new nodes are inserted after each node other than the input node
    and the output node.
"""
def growCompGraph(compGraph):

    # Find nodes in current computation graph
    nodes = []
    for start, end in compGraph.edges:
        for node in start, end:
            if node not in nodes:
                nodes.append(node)

    # Find next available node
    nextAvailableNode = 0
    while nextAvailableNode in nodes:
        nextAvailableNode += 1

    # Find nodes to expand
    nodesToExpand = list(compGraph.nodes)
    nodesToKeep = [compGraph.inputIndex, compGraph.outputIndex]
    for node in nodesToKeep:
        if node in nodesToExpand:
            nodesToExpand.remove(node)

    # Expand nodes
    newEdges = list(compGraph.edges)
    for node in nodesToExpand:
        
        # Expand node
        newNode = nextAvailableNode
        nextAvailableNode += 1

        tempEdges = []
        for start, end in newEdges:

            if node == start:
                tempEdges.append((start, newNode))
                tempEdges.append((newNode, end))
            else:
                tempEdges.append((start, end))

        newEdges = list(tempEdges)

    newCompGraph = ComputationGraph(
            newEdges,
            compGraph.inputIndex,
            compGraph.outputIndex
    )
    return newCompGraph

