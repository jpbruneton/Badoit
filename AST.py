import config
from State import State

# =============================== CLASS: AST NODE ================================ #

class Node:
    # ---------------------------------------------------------------------------- #
    # Constructs a node from a new symbol
    # we go upward : always one parent and at most two childrens (if arity max = 2)

    def __init__(self, symbol, arity, children, parent, label):
        self.symbol = symbol
        self.arity = arity
        self.parent = parent
        self.children = children
        self.absolutelabel = label


# =============================== CLASS: AST ================================ #
# A class representing an AST
class AST:
    # ---------------------------------------------------------------------------- #
    # Constructs a tree from initial scalar #here root is actually a leaf
    def __init__(self, startingsymbol):

        self.onebottomnode = self.createNode(startingsymbol, 0, None, None, 1)
        self.topnode = None

    # ---------------------------------------------------------------------------- #
    # Builds a node from the state and adds it to the tree
    def createNode(self, newsymbol, arity, children, parent, label):
        node = Node(newsymbol, arity, children, parent, label)
        return node

# --------------------------------------------------------------------------- #
# recursive way of getting the rpn from the ast, starting from topnode:
    def from_ast_to_rpn(self, node, rpn = None):
        if rpn == None:
            rpn = []

        for i in range(0, node.arity):
            self.from_ast_to_rpn(node.children[i], rpn)

        rpn.append(node.symbol)
        return rpn

# --------------------------------------------------------------------------- #
# recursive way of getting the polish from the ast, starting from topnode:
    def from_ast_to_prefix(self, node, pn = None):
        if pn == None:
            pn = []
        node.absolutelabel = len(pn)
        pn.append(node.symbol)

        for i in range(0, node.arity):
            self.from_ast_to_rpn(node.children[i], pn)

        return pn

# --------------------------------------------------------------------------- #
# recursive way of getting the polish from the ast, starting from topnode:
    def from_ast_get_node(self, node, n, result = None):
        if result == None:
            result = []

        if node.absolutelabel == n:
            result.append(node)

        for i in range(0, node.arity):
            self.from_ast_get_node(node.children[i], n, result)

        return result
