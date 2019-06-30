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
    For a given computation graph, builds and returns a grown version.

    Returns:
        newCompGraph: a grown version of the input computation graph
        nodesToCopy: a list of 3-tuples of integers representing the
            nodes whose weights should be copied to preserve the
            network's function in a growth step. [(x, y, 'conv'),
            (x, z, 'bn')] means that the convolutional weights from
            node x should be copied to node y and the batch norm
            parameters from node x should be copied to node z.
"""
def growCompGraph(compGraph, growthHistory, growthMode, numConvToAdd,
        itemsToExpand):

    assert growthMode in ['expandEdge', 'expandNode']
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
    nodesToCopy = []
    for start, end in compGraph.edges:
        newEdges.append((start, end))

        if (start, end) in edgesToExpand:

            prevNode = start
            for i in range(numConvToAdd):
                currentNode = nextAvailableNode
                newEdges.append((prevNode, currentNode))
                nodesToCopy.append((start, currentNode, 'bn'))

                prevNode = currentNode
                nextAvailableNode += 1
            newEdges.append((prevNode, end))

    newCompGraph = ComputationGraph(
            newEdges,
            compGraph.inputIndex,
            compGraph.outputIndex
    )

    # Empty list shows that no weights need to be copied to new nodes
    return newCompGraph, nodesToCopy

"""
    Helper function for growCompGraph which implements the 'expandNode'
    growth mode.
"""
def expandNode(compGraph, growthHistory, numConvToAdd, itemsToExpand):

    assert numConvToAdd >= 1

    # Find next available node
    nextAvailableNode = 0
    while nextAvailableNode in compGraph.nodes:
        nextAvailableNode += 1

    # Find nodes to expand
    if itemsToExpand == 'all':
        nodesToExpand = list(compGraph.nodes)
    elif itemsToExpand == 'oldest':
        nodesToExpand = [node for node in compGraph.nodes if
                growthHistory[node] == 0]
    elif itemsToExpand == 'youngest':
        currentStep = max(growthHistory.values())
        nodesToExpand = [node for node in compGraph.nodes if
                growthHistory[node] == currentStep]
    nodesToKeep = [compGraph.inputIndex, compGraph.outputIndex]
    for node in nodesToKeep:
        if node in nodesToExpand:
            nodesToExpand.remove(node)

    # Expand nodes
    newEdges = []
    nodesToCopy = []
    for node in compGraph.nodes:
        inputNodes = [inputNode for inputNode in compGraph.nodes if
                (inputNode, node) in compGraph.edges]
        outputNodes = [outputNode for outputNode in compGraph.nodes if
                (node, outputNode) in compGraph.edges]

        # Add currently existing edges to graph
        originalEdges = [(inputNode, node) for inputNode in inputNodes] + [
                (node, outputNode) for outputNode in outputNodes]
        for edge in originalEdges:
            if edge not in newEdges:
                newEdges.append(edge)

        if node in nodesToExpand:

            # Connect input nodes to first new node
            currentNode = nextAvailableNode
            for inputNode in inputNodes:
                newEdges.append((inputNode, currentNode))
            prevNode = currentNode
            nextAvailableNode += 1

            # Ensure that weights from expanded node are copied
            # to the first new node in the expansion, to preserve
            # the function computed by the network
            nodesToCopy.append((node, currentNode, 'conv'))

            # Connect adjacent new nodes
            for i in range(numConvToAdd - 1):
                currentNode = nextAvailableNode
                newEdges.append((prevNode, currentNode))
                nodesToCopy.append((node, currentNode, 'bn'))
                prevNode = currentNode
                nextAvailableNode += 1

            # Connect last new node to output nodes
            for outputNode in outputNodes:
                newEdges.append((currentNode, outputNode))

    newCompGraph = ComputationGraph(
            newEdges,
            compGraph.inputIndex,
            compGraph.outputIndex
    )
    return newCompGraph, nodesToCopy

