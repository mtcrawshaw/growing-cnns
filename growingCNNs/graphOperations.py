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
def growCompGraph(compGraph, growthHistory, mode='linear'):

    assert mode in ['linear', 'skip', 'skipSlim', 'branching']

    # Find next available node
    nextAvailableNode = 0
    while nextAvailableNode in compGraph.nodes:
        nextAvailableNode += 1

    # Find edges to expand
    if 'slim' in mode.lower():
        currentStep = max(growthHistory.values())
        edgesToExpand = [(start, end) for (start, end) in compGraph.edges if
                growthHistory[start] == currentStep or
                growthHistory[end] == currentStep]
    else:
        edgesToExpand = list(compGraph.edges)

    # Expand nodes
    newEdges = []
    for start, end in compGraph.edges:

        if mode == 'linear':

            # Linear
            if (start, end) in edgesToExpand:
                newNode = nextAvailableNode
                newEdges.append((start, newNode))
                newEdges.append((newNode, end))
                nextAvailableNode += 1
            else:
                tempEdges.append((start, end))

        elif 'skip' in mode:

            # Skip/Skip slim
            newEdges.append((start, end))
            if (start, end) in edgesToExpand:
                newNode = nextAvailableNode
                newEdges.append((start, newNode))
                newEdges.append((newNode, end))
                nextAvailableNode += 1

        elif mode == 'branching':

            # Branching
            newEdges.append((start, end))
            if (start, end) in edgesToExpand:
                newNode1 = nextAvailableNode
                nextAvailableNode += 1
                newNode2 = nextAvailableNode

                newEdges.append((start, newNode1))
                newEdges.append((newNode1, newNode2))
                newEdges.append((newNode2, end))

                nextAvailableNode += 1


    newCompGraph = ComputationGraph(
            newEdges,
            compGraph.inputIndex,
            compGraph.outputIndex
    )
    return newCompGraph

