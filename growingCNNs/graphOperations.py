# This is in case no parent packages are imported, such as in the test cases
try:
    from .utils.computationGraph import ComputationGraph
except:
    from utils.computationGraph import ComputationGraph


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
    For a given computation graph, builds and returns a grown version
    using the either linear, skip, skipSlim, or branching growth mode.
"""
def growCompGraph(compGraph, growthHistory, growthMode, numConvToAdd,
        itemsToExpand):

    assert growthMode in ['expandEdge'] # ['expandEdge', 'expandNode']
    assert itemsToExpand in ['all', 'oldest', 'youngest']

    growthFn = eval(growthMode)
    return growthFn(compGraph, growthHistory, numConvToAdd, itemsToExpand)

"""
    Helper function for growCompGraph which implements the 'expandEdge'
    growth mode.
"""
def expandEdge(compGraph, growthHistory, numConvToAdd, itemsToExpand):

    assert numConvToAdd >= 1

    # Find next available node
    nextAvailableNode = 0
    while nextAvailableNode in compGraph.nodes:
        nextAvailableNode += 1

    # Find edges to expand
    if itemsToExpand == 'all':
        edgesToExpand = list(compGraph.edges)
    elif itemsToExpand == 'oldest':
        edgesToExpand = [(start, end) for (start, end) in compGraph.edges if
                growthHistory[start] == 0 and
                growthHistory[end] == 0]
    elif itemsToExpand == 'youngest':
        currentStep = max(growthHistory.values())
        edgesToExpand = [(start, end) for (start, end) in compGraph.edges if
                growthHistory[start] == currentStep or
                growthHistory[end] == currentStep]

    # Expand edges
    newEdges = []
    for start, end in compGraph.edges:
        newEdges.append((start, end))

        if (start, end) in edgesToExpand:

            prevNode = start
            for i in range(numConvToAdd):
                currentNode = nextAvailableNode
                newEdges.append((prevNode, currentNode))
                prevNode = currentNode
                nextAvailableNode += 1
            newEdges.append((prevNode, end))

    newCompGraph = ComputationGraph(
            newEdges,
            compGraph.inputIndex,
            compGraph.outputIndex
    )
    return newCompGraph

