import ast
import networkx as nx
from itertools import islice


class C_CFG():
    def __init__(self):
        # 结束点
        self.finlineno = []
        # 起始点
        self.firstlineno = 1
        self.loopflag = 0
        self.clean_code = ''

        self.func_name = dict()
        self.G = nx.DiGraph()
        self.DG = nx.DiGraph()
        self.circle = []
        # 分支点
        self.dece_node = []

    def k_shortest_paths(self, G, source, target, k, weight=None):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

    def get_allpath(self):

        # # three most coverage paths
        path_real = 10
        all_paths = []
        self.finlineno = list(set(self.finlineno))
        self.finlineno.sort(reverse=False)            # sort the exit node from the small to big
        coverage = -1
        path1 = []
        node_uncover = set(self.G.nodes())
        length_path = 10000
        for fno in self.finlineno:
            if nx.has_path(self.G, self.firstlineno, fno):
                paths = self.k_shortest_paths(self.G, self.firstlineno, fno, 50)
                for path in paths:
                    if len(set(path) & node_uncover) > coverage:
                        path1 = path
                        coverage = len(set(path1) & node_uncover)
        for i in range(0, len(path1) - 1):
            n1 = path1[i]
            n2 = path1[i + 1]
            if len(self.G.adj[n1]) > 1:
                self.G[n1][n2]['weight'] = 100
        all_paths.append(path1)
        all_nodes = set(self.G.nodes())
        node_uncover = all_nodes - set(path1)
        if path1 == []:
            path_real -= 1
        
        if 0 in node_uncover:
            node_uncover.remove(0)

        coverage = -1
        path2 = []
        for fno in self.finlineno:
            if nx.has_path(self.G, self.firstlineno, fno):
                paths = self.k_shortest_paths(self.G, self.firstlineno, fno, 50)
                for path_p in all_paths:
                    if path_p in paths:
                        paths.remove(path_p)
                for path in paths:
                    if len(set(path) & node_uncover) > coverage:
                        path2 = path
                        coverage = len(set(path2) & node_uncover)
                    elif len(set(path) & node_uncover) == coverage and len(set(path)) > len(set(path2)):
                        path2 = path
        for i in range(0, len(path2)-1):
            n1 = path2[i]
            n2 = path2[i+1]
            if len(self.G.adj[n1]) > 1:
                self.G[n1][n2]['weight'] = 100

        node_uncover = node_uncover-set(path2)
        all_paths.append(path2)
        if path2 == []:
            path_real -= 1


        coverage = -1
        path3 = []
        for fno in self.finlineno:
            if nx.has_path(self.G, self.firstlineno, fno):
                paths = self.k_shortest_paths(self.G, self.firstlineno, fno, 50)
                for path_p in all_paths:
                    if path_p in paths:
                        paths.remove(path_p)
                for path in paths:
                    if len(set(path) & node_uncover) > coverage:
                        path3 = path
                        coverage = len(set(path3) & node_uncover)
                    elif len(set(path) & node_uncover) == coverage and len(set(path)) > len(set(path3)):
                        path3 = path
        for i in range(0, len(path3)-1):
            n1 = path3[i]
            n2 = path3[i+1]
            if len(self.G.adj[n1]) > 1:
                self.G[n1][n2]['weight'] = 100

        node_uncover = node_uncover-set(path3)
        all_paths.append(path3)
        if path3 == []:
            path_real -= 1


        coverage = -1
        path4 = []
        for fno in self.finlineno:
            if nx.has_path(self.G, self.firstlineno, fno):
                paths = self.k_shortest_paths(self.G, self.firstlineno, fno, 50)
                for path_p in all_paths:
                    if path_p in paths:
                        paths.remove(path_p)
                for path in paths:
                    if len(set(path) & node_uncover) > coverage:
                        path4 = path
                        coverage = len(set(path4) & node_uncover)
                    elif len(set(path) & node_uncover) == coverage and len(set(path)) > len(set(path4)):
                        path4 = path
        for i in range(0, len(path4)-1):
            n1 = path4[i]
            n2 = path4[i+1]
            if len(self.G.adj[n1]) > 1:
                self.G[n1][n2]['weight'] = 100

        node_uncover = node_uncover-set(path4)
        all_paths.append(path4)
        if path4 == []:
            path_real -= 1


        coverage = -1
        path5 = []
        for fno in self.finlineno:
            if nx.has_path(self.G, self.firstlineno, fno):
                paths = self.k_shortest_paths(self.G, self.firstlineno, fno, 50)
                for path_p in all_paths:
                    if path_p in paths:
                        paths.remove(path_p)
                for path in paths:
                    if len(set(path) & node_uncover) > coverage:
                        path5 = path
                        coverage = len(set(path5) & node_uncover)
                    elif len(set(path) & node_uncover) == coverage and len(set(path)) > len(set(path5)):
                        path5 = path
        for i in range(0, len(path5)-1):
            n1 = path5[i]
            n2 = path5[i+1]
            if len(self.G.adj[n1]) > 1:
                self.G[n1][n2]['weight'] = 100

        node_uncover = node_uncover-set(path5)
        all_paths.append(path5)
        if path5 == []:
            path_real -= 1


        coverage = -1
        path6 = []
        for fno in self.finlineno:
            if nx.has_path(self.G, self.firstlineno, fno):
                paths = self.k_shortest_paths(self.G, self.firstlineno, fno, 50)
                for path_p in all_paths:
                    if path_p in paths:
                        paths.remove(path_p)
                for path in paths:
                    if len(set(path) & node_uncover) > coverage:
                        path6 = path
                        coverage = len(set(path6) & node_uncover)
                    elif len(set(path) & node_uncover) == coverage and len(set(path)) > len(set(path6)):
                        path6 = path
        for i in range(0, len(path6)-1):
            n1 = path6[i]
            n2 = path6[i+1]
            if len(self.G.adj[n1]) > 1:
                self.G[n1][n2]['weight'] = 100

        node_uncover = node_uncover-set(path6)
        all_paths.append(path6)
        if path6 == []:
            path_real -= 1


        coverage = -1
        path7 = []
        for fno in self.finlineno:
            if nx.has_path(self.G, self.firstlineno, fno):
                paths = self.k_shortest_paths(self.G, self.firstlineno, fno, 50)
                for path_p in all_paths:
                    if path_p in paths:
                        paths.remove(path_p)
                for path in paths:
                    if len(set(path) & node_uncover) > coverage:
                        path7 = path
                        coverage = len(set(path7) & node_uncover)
                    elif len(set(path) & node_uncover) == coverage and len(set(path)) > len(set(path7)):
                        path7 = path
        for i in range(0, len(path7)-1):
            n1 = path7[i]
            n2 = path7[i+1]
            if len(self.G.adj[n1]) > 1:
                self.G[n1][n2]['weight'] = 100

        node_uncover = node_uncover-set(path7)
        all_paths.append(path7)
        if path7 == []:
            path_real -= 1


        coverage = -1
        path8 = []
        for fno in self.finlineno:
            if nx.has_path(self.G, self.firstlineno, fno):
                paths = self.k_shortest_paths(self.G, self.firstlineno, fno, 50)
                for path_p in all_paths:
                    if path_p in paths:
                        paths.remove(path_p)
                for path in paths:
                    if len(set(path) & node_uncover) > coverage:
                        path8 = path
                        coverage = len(set(path8) & node_uncover)
                    elif len(set(path) & node_uncover) == coverage and len(set(path)) > len(set(path8)):
                        path8 = path
        for i in range(0, len(path8)-1):
            n1 = path8[i]
            n2 = path8[i+1]
            if len(self.G.adj[n1]) > 1:
                self.G[n1][n2]['weight'] = 100

        node_uncover = node_uncover-set(path8)
        all_paths.append(path8)
        if path8 == []:
            path_real -= 1


        coverage = -1
        path9 = []
        for fno in self.finlineno:
            if nx.has_path(self.G, self.firstlineno, fno):
                paths = self.k_shortest_paths(self.G, self.firstlineno, fno, 50)
                for path_p in all_paths:
                    if path_p in paths:
                        paths.remove(path_p)
                for path in paths:
                    if len(set(path) & node_uncover) > coverage:
                        path9 = path
                        coverage = len(set(path9) & node_uncover)
                    elif len(set(path) & node_uncover) == coverage and len(set(path)) > len(set(path9)):
                        path9 = path
        for i in range(0, len(path9)-1):
            n1 = path9[i]
            n2 = path9[i+1]
            if len(self.G.adj[n1]) > 1:
                self.G[n1][n2]['weight'] = 100

        node_uncover = node_uncover-set(path9)
        all_paths.append(path9)
        if path9 == []:
            path_real -= 1


        
        coverage = -1
        path10 = []
        for fno in self.finlineno:
            if nx.has_path(self.G, self.firstlineno, fno):
                paths = self.k_shortest_paths(self.G, self.firstlineno, fno, 50)
                for path_p in all_paths:
                    if path_p in paths:
                        paths.remove(path_p)
                for path in paths:
                    if len(set(path) & node_uncover) > coverage:
                        path10 = path
                        coverage = len(set(path10) & node_uncover)
                    elif len(set(path) & node_uncover) == coverage and len(set(path)) > len(set(path10)):
                        path10 = path
        
        node_uncover = node_uncover - set(path10)
        all_paths.append(path10)
        if path10 == []:
            path_real -= 1
        # # --------------------------------------------------------------

        num_path = 0
        num_path = len(all_paths)
        # 节点覆盖率
        ratio = 1-(len(node_uncover)/len(all_nodes))
        #return num_path, all_paths
        return num_path, all_paths, len(self.dece_node), ratio, path_real

    def run(self, root):
        # self.visit(root)
        self.clean_code = root
        self.finlineno.append(root.end_point[0]+1)
        self.ast_visit(root)

    def parse_ast_file(self, ast_code):
        self.run(ast_code)
        return ast_code

    def parse_ast(self, source_ast):
        self.run(source_ast)
        return source_ast

    def get_source(self, fn):
        ''' Return the entire contents of the file whose name is given.
            Almost most entirely copied from stc. '''
        try:
            f = open(fn, 'r')
            s = f.read()
            f.close()
            return s
        except IOError:
            return ''

    def ast_visit(self, node):
        method = getattr(self, "visit_" + node.type)
        return method(node)

    def visit_program(self, node):
        # self.finlineno.append(node.end_point[0] + 1)
        self.finlineno.append(node.children[-1].end_point[0] + 1)

        for index, z in enumerate(node.children):
            for i in range(z.start_point[0] + 1, z.end_point[0] + 2):
                self.G.add_node(i)
            if self.firstlineno > z.start_point[0] + 1:
                self.firstlineno = z.start_point[0] + 1
            if z.type == "method_declaration":
                if index == len(node.children) - 1:
                    self.finlineno.append(z.end_point[0] + 1)
                self.ast_visit(z)
            else:
                self.visit_piece(z)

            # if z.type == "compound_statement":
            #     if index == len(node.children) - 1:
            #         self.finlineno.append(z.end_point[0] + 1)
            #     self.ast_visit(z)
            # if z.type == "local_variable_declaration":
            #     self.ast_visit(z)
            # else:
            #     self.visit_piece(z)

    def visit_translation_unit(self, node):
        self.finlineno.append(node.children[-1].end_point[0] + 1)
        # for i in range(node.start_point[0] + 1, node.end_point[0] + 2):
        #     self.G.add_edge(i, i + 1, weight=1)
        for index, z in enumerate(node.children):
            for i in range(z.start_point[0] + 1, z.end_point[0] + 2):
                self.G.add_node(i)
            if self.firstlineno > z.start_point[0] + 1:
                self.firstlineno = z.start_point[0] + 1
            if z.type == "function_definition":
                if index == len(node.children) - 1:
                    self.finlineno.append(z.end_point[0] + 1)
                self.ast_visit(z)
            else:
                self.visit_piece(z)

    def visit_function_definition(self, node):
        for z in node.children:
            if z.type == "compound_statement":
                self.ast_visit(z)
            else:
                self.visit_piece(z)

    def visit_method_declaration(self, node):
        for z in node.children:
            if z.type == "block":
                self.ast_visit(z)
            else:
                self.visit_piece(z)

    def visit_block(self, node):
        # self.G.add_edge(node.start_point[0]+1, node.end_point[0]+1)
        for index, z in enumerate(node.children):
            if z.type == "for_statement":
                self.ast_visit(z)
            elif z.type == "enhanced_for_statement":
                self.ast_visit(z)
            elif z.type == "while_statement":
                self.ast_visit(z)
            elif z.type == "do_statement":
                self.ast_visit(z)
            elif z.type == "try_with_resources_statement":
                self.ast_visit(z)
            elif z.type == "assert_statement":
                self.ast_visit(z)
            elif z.type == "switch_statement":
                self.ast_visit(z)
            elif z.type == "case_statement":
                self.G.add_edge(node.start_point[0]+1, z.start_point[0]+1, weight=1)
                self.ast_visit(z)
            elif z.type == "switch_block":
                self.ast_visit(z)
            elif z.type == "switch_block_statement_group":
                self.ast_visit(z)
            elif z.type == "labeled_statement":
                self.ast_visit(z)
            elif z.type == "continue_statement":
                self.ast_visit(z)
            elif z.type == "try_statement":
                self.ast_visit(z)
            elif z.type == "throw_statement":
                self.ast_visit(z)
            elif z.type == "if_statement":
                self.ast_visit(z)
            elif z.type == "synchronized_statement":
                self.ast_visit(z)
            elif z.type == "expression_statement":
                self.ast_visit(z)
            elif z.type == "local_variable_declaration":
                self.ast_visit(z)
            elif z.type == "return_statement":
                self.ast_visit(z)
            elif z.type == "block":
                self.ast_visit(z)
            elif z.type == "parenthesized_expression":
                self.ast_visit(z)
            elif z.type == "ERROR":
                self.ast_visit(z)
            elif z.type == "break_statement":
                self.ast_visit(z)
            elif z.type == "class_declaration":
                self.ast_visit(z)
            elif z.type == "declaration":
                self.ast_visit(z)
            elif z.type == "function_declarator":
                self.ast_visit(z)
        #     elif z.type == "}":
        #         self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
        #     elif z.type == "{":
        #         self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
        #     elif z.type == ";":
        #         self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
        #     else:
        #         self.visit_piece(z)
        # if len(node.children) > 0 and node.children[0].type == "{":
        #     self.G.add_edge(node.children[0].start_point[0] + 1, node.children[0].start_point[0] + 2, weight=1)
        # if len(node.children) > 0 and node.children[-1].type == "}":
        #     self.G.add_edge(node.children[-1].start_point[0], node.children[-1].start_point[0] + 1, weight=1)



            elif z.type == "}":
                if index > 0 and (node.children[index-1].type == "throw_statement"
                                  or node.children[index-1].type == "return_statement"):
                    pass
                else:
                    self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
            elif z.type == "{":
                self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
                self.G.add_edge(z.start_point[0] + 1, z.start_point[0] + 2, weight=1)
            elif z.type == ";":
                self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
            else:
                self.visit_piece(z)



    # def visit_compound_statement(self, node):
    #     # self.G.add_edge(node.start_point[0]+1, node.end_point[0]+1)
    #     for index, z in enumerate(node.children):
    #         if z.type == "for_statement":
    #             self.ast_visit(z)
    #         elif z.type == "enhanced_for_statement":
    #             self.ast_visit(z)
    #         elif z.type == "while_statement":
    #             self.ast_visit(z)
    #         elif z.type == "do_statement":
    #             self.ast_visit(z)
    #         elif z.type == "try_with_resources_statement":
    #             self.ast_visit(z)
    #         elif z.type == "assert_statement":
    #             self.ast_visit(z)
    #         elif z.type == "switch_statement":
    #             self.ast_visit(z)
    #         elif z.type == "case_statement":
    #             self.G.add_edge(node.start_point[0]+1, z.start_point[0]+1, weight=1)
    #             self.ast_visit(z)
    #         elif z.type == "switch_block":
    #             self.ast_visit(z)
    #         elif z.type == "switch_block_statement_group":
    #             self.ast_visit(z)
    #         elif z.type == "labeled_statement":
    #             self.ast_visit(z)
    #         elif z.type == "continue_statement":
    #             self.ast_visit(z)
    #         elif z.type == "try_statement":
    #             self.ast_visit(z)
    #         elif z.type == "throw_statement":
    #             self.ast_visit(z)
    #         elif z.type == "if_statement":
    #             self.ast_visit(z)
    #         elif z.type == "synchronized_statement":
    #             self.ast_visit(z)
    #         elif z.type == "expression_statement":
    #             self.ast_visit(z)
    #         elif z.type == "local_variable_declaration":
    #             self.ast_visit(z)
    #         elif z.type == "return_statement":
    #             self.ast_visit(z)
    #         elif z.type == "compound_statement":
    #             self.ast_visit(z)
    #         elif z.type == "parenthesized_expression":
    #             self.ast_visit(z)
    #         elif z.type == "ERROR":
    #             self.ast_visit(z)
    #         elif z.type == "break_statement":
    #             self.ast_visit(z)
    #         elif z.type == "class_declaration":
    #             self.ast_visit(z)
    #         elif z.type == "declaration":
    #             self.ast_visit(z)
    #         elif z.type == "function_declarator":
    #             self.ast_visit(z)
    #         elif z.type == "}":
    #             self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
    #         elif z.type == "{":
    #             self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
    #         elif z.type == ";":
    #             self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
    #         else:
    #             self.visit_piece(z)
    #     if len(node.children) > 0 and node.children[0].type == "{":
    #         self.G.add_edge(node.children[0].start_point[0] + 1, node.children[0].start_point[0] + 2, weight=1)
    #     if len(node.children) > 0 and node.children[-1].type == "}":
    #         self.G.add_edge(node.children[-1].start_point[0], node.children[-1].start_point[0] + 1, weight=1)

    def visit_piece(self, node):
        # self.G.add_edge(node.start_point[0]+1, node.end_point[0]+1)
        if node.type == "for_statement":
            self.ast_visit(node)
        elif node.type == "enhanced_for_statement":
            self.ast_visit(node)
        elif node.type == "while_statement":
            self.ast_visit(node)
        elif node.type == "do_statement":
            self.ast_visit(node)
        elif node.type == "try_with_resources_statement":
            self.ast_visit(node)
        elif node.type == "assert_statement":
            self.ast_visit(node)
        elif node.type == "switch_statement":
            self.ast_visit(node)
        elif node.type == "switch_block":
            self.ast_visit(node)
        elif node.type == "switch_block_statement_group":
            self.ast_visit(node)
        elif node.type == "labeled_statement":
            self.ast_visit(node)
        elif node.type == "continue_statement":
            self.ast_visit(node)
        elif node.type == "try_statement":
            self.ast_visit(node)
        elif node.type == "throw_statement":
            self.ast_visit(node)
        elif node.type == "if_statement":
            self.ast_visit(node)
        elif node.type == "synchronized_statement":
            self.ast_visit(node)
        elif node.type == "expression_statement":
            self.ast_visit(node)
        elif node.type == "local_variable_declaration":
            self.ast_visit(node)
        elif node.type == "parenthesized_expression":
            self.ast_visit(node)
        elif node.type == "return_statement":
            self.ast_visit(node)
        elif node.type == "ERROR":
            self.ast_visit(node)
        elif node.type == "break_statement":
            self.ast_visit(node)
        elif node.type == "class_declaration":
            self.ast_visit(node)
        elif node.type == "declaration":
            self.ast_visit(node)
        elif node.type == "function_declarator":
            self.ast_visit(node)
        elif node.type == "compound_statement":
            self.ast_visit(node)
        elif node.type == "block":
            self.ast_visit(node)
        elif node.type == "goto_statement":
            self.ast_visit(node)
        elif node.type == "preproc_if":
            self.ast_visit(node)
        elif node.type == "preproc_params":
            self.ast_visit(node)
        elif node.type == "pointer_declarator":
            self.ast_visit(node)
        elif node.type == "preproc_ifdef":
            self.ast_visit(node)
        elif node.type == "preproc_elif":
            self.ast_visit(node)
        elif node.type == "preproc_function_def":
            self.ast_visit(node)
        elif node.type == "preproc_call":
            self.ast_visit(node)
        elif node.type == "preproc_else":
            self.ast_visit(node)
        elif node.type == "preproc_def":
            self.ast_visit(node)
        elif node.type == "proproc_include":
            self.ast_visit(node)
        elif node.type == "preproc_defined":
            self.ast_visit(node)
        elif node.type == "function_definition":
            self.ast_visit(node)
        elif node.type == "method_declaration":
            self.ast_visit(node)
        elif node.type == "}":
            self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        elif node.type == "{":
            self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        elif node.type == ";":
            self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        elif node.type == "\n":
            self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        else:
            pass

    def visit_function_declarator(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] + 1 > node.start_point[0] + 1:
            for i in range(node.start_point[0] + 1, node.end_point[0] + 1):
                self.G.add_edge(i, i + 1, weight=1)

    def visit_class_declaration(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] + 1 > node.start_point[0] + 1:
            for i in range(node.start_point[0] + 1, node.end_point[0] + 1):
                self.G.add_edge(i, i + 1, weight=1)

    def visit_pointer_declarator(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] + 1 > node.start_point[0] + 1:
            for i in range(node.start_point[0] + 1, node.end_point[0] + 1):
                self.G.add_edge(i, i + 1, weight=1)

    def visit_for_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.next_sibling is not None:
            self.G.add_edge(node.start_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.circle.append((node.start_point[0] + 1, node.end_point[0] + 1))
        self.dece_node.append(node.start_point[0] + 1)
        # add the statement of 'For condiation'

        for z in node.children:
            if z.type == "block":
                self.ast_visit(z)
            elif z.type == "if_statement":
                for j in z.children:
                    if j.type == "block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            elif z.type == "try_statement":
                for j in z.children:
                    if j.type == "block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            elif z.type == "switch_statement":
                for j in z.children:
                    if j.type == "switch_block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
                    elif j.type == "block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            else:
                self.visit_piece(z)
                self.G.add_edge(z.end_point[0]+1, node.start_point[0]+1, weight=1)
        self.loopflag = node.end_point[0] + 1

        # for z in node.children:
        #     if z.type == "compound_statement":
        #         self.ast_visit(z)
        #     elif z.type == "if_statement":
        #         for j in z.children:
        #             if j.type == "compound_statement":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #     elif z.type == "try_statement":
        #         for j in z.children:
        #             if j.type == "compound_statement":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #     elif z.type == "switch_statement":
        #         for j in z.children:
        #             if j.type == "switch_block":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #             elif j.type == "compound_statement":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #     else:
        #         self.visit_piece(z)
        #         self.G.add_edge(z.end_point[0]+1, node.start_point[0]+1, weight=1)
        # self.loopflag = node.end_point[0] + 1

    def visit_enhanced_for_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.next_sibling is not None:
            self.G.add_edge(node.start_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.circle.append((node.start_point[0] + 1, node.end_point[0] + 1))
        self.dece_node.append(node.start_point[0] + 1)

        for z in node.children:
            if z.type == "block":
                self.ast_visit(z)
            elif z.type == "if_statement":
                for j in z.children:
                    if j.type == "block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            elif z.type == "try_statement":
                for j in z.children:
                    if j.type == "block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            elif z.type == "switch_statement":
                for j in z.children:
                    if j.type == "switch_block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
                    elif j.type == "block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            else:
                self.visit_piece(z)
                self.G.add_edge(z.end_point[0]+1, node.start_point[0]+1, weight=1)
        self.loopflag = node.end_point[0] + 1

        # for z in node.children:
        #     if z.type == "compound_statement":
        #         self.ast_visit(z)
        #     elif z.type == "if_statement":
        #         for j in z.children:
        #             if j.type == "compound_statement":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #     elif z.type == "try_statement":
        #         for j in z.children:
        #             if j.type == "compound_statement":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #     elif z.type == "switch_statement":
        #         for j in z.children:
        #             if j.type == "switch_block":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #             elif j.type == "compound_statement":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #     else:
        #         self.visit_piece(z)
        #         self.G.add_edge(z.end_point[0]+1, node.start_point[0]+1, weight=1)
        # self.loopflag = node.end_point[0] + 1

    def visit_do_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.next_sibling is not None:  # named_child_count
            self.G.add_edge(node.start_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        if node.end_point[0] + 1 != self.finlineno[0]:
            self.G.add_edge(node.children[-1].end_point[0] + 1, node.end_point[0] + 2, weight=1)

        for z in node.children:
            if z.type == "block":
                self.ast_visit(z)
            else:
                self.visit_piece(z)

        # for z in node.children:
        #     if z.type == "compound_statement":
        #         self.ast_visit(z)
        #     else:
        #         self.visit_piece(z)

    def visit_try_with_resources_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        body_node = {}

        for z in node.children:
            if z.type == "block":
                self.ast_visit(z)
                body_node['bs'] = z.start_point[0] + 1
                body_node['be'] = z.end_point[0] + 1
            elif z.type == "finally_clause":
                self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
                self.dece_node.append(z.start_point[0] + 1)
                body_node['fs'] = z.start_point[0] + 1
                body_node['fe'] = z.end_point[0] + 1
                self.ast_visit(z)
            elif z.type == "catch_clause":
                self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
                self.dece_node.append(z.start_point[0] + 1)
                body_node['cs'] = z.start_point[0] + 1
                body_node['ce'] = z.end_point[0] + 1
                self.ast_visit(z)
            else:
                self.visit_piece(z)

        # for z in node.children:
        #     if z.type == "compound_statement":
        #         self.ast_visit(z)
        #         body_node['bs'] = z.start_point[0] + 1
        #         body_node['be'] = z.end_point[0] + 1
        #     elif z.type == "finally_clause":
        #         self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
        #         self.dece_node.append(z.start_point[0] + 1)
        #         body_node['fs'] = z.start_point[0] + 1
        #         body_node['fe'] = z.end_point[0] + 1
        #         self.ast_visit(z)
        #     elif z.type == "catch_clause":
        #         self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
        #         self.dece_node.append(z.start_point[0] + 1)
        #         body_node['cs'] = z.start_point[0] + 1
        #         body_node['ce'] = z.end_point[0] + 1
        #         self.ast_visit(z)
        #     else:
        #         self.visit_piece(z)

        if 'be' in body_node and 'cs' in body_node:
            self.G.add_edge(body_node['be'], body_node['cs'], weight=1)
            # for i in range(body_node['bs'], body_node['be']+1):
            #    self.G.add_edge(i, body_node['cs'], weight=1)
        if 'ce' in body_node and 'fs' in body_node:
            self.G.add_edge(body_node['ce'], body_node['fs'], weight=1)
            # for i in range(body_node['cs'], body_node['ce']+1):
            #    self.G.add_edge(i, body_node['fs'], weight=1)

    def visit_assert_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] + 1 > node.start_point[0] + 1:
            for i in range(node.start_point[0] + 1, node.end_point[0] + 1):
                self.G.add_edge(i, i + 1, weight=1)
        if node.end_point[0] + 1 not in self.finlineno:
            self.finlineno.append(node.end_point[0] + 1)

    def visit_switch_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        self.circle.append((node.start_point[0] + 1, node.end_point[0] + 1))
        if node.next_sibling is not None:  # named_child_count
            self.G.add_edge(node.start_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        if node.end_point[0] + 1 != self.finlineno[0]:
            self.G.add_edge(node.children[-1].end_point[0] + 1, node.end_point[0] + 2, weight=1)
        for z in node.children:
            if z.type == "switch_block":
                self.dece_node.append(z.start_point[0] + 1)
                self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
                for i in range(len(z.children)):
                    if node.start_point[0] != z.children[i].start_point[0]:
                        self.G.add_edge(node.start_point[0] + 1, z.children[i].start_point[0] + 1, weight=1)
                self.ast_visit(z)

            elif z.type == "block":
                self.ast_visit(z)

            # elif z.type == "compound_statement":
            #     self.ast_visit(z)

            else:
                self.visit_piece(z)

    def visit_case_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            self.visit_piece(z)

    def visit_switch_block(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)

        for z in node.children:
            if z.type == "switch_block_statement_group":
                self.ast_visit(z)

    def visit_switch_block_statement_group(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            self.ast_vist(z)

    def visit_while_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.next_sibling is not None:
            self.G.add_edge(node.start_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.circle.append((node.start_point[0] + 1, node.end_point[0] + 1))
        # add the statement of 'While condiation'

        for z in node.children:
            if z.type == "block":
                self.ast_visit(z)
            elif z.type == "if_statement":
                for j in z.children:
                    if j.type == "block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            elif z.type == "try_statement":
                for j in z.children:
                    if j.type == "block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            elif z.type == "switch_statement":
                for j in z.children:
                    if j.type == "switch_block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
                    elif j.type == "block":
                        self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
            elif z.type == 'else':
                self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
            else:
                self.visit_piece(z)
        self.loopflag = node.end_point[0] + 1

        # for z in node.children:
        #     if z.type == "compound_statement":
        #         self.ast_visit(z)
        #     elif z.type == "if_statement":
        #         for j in z.children:
        #             if j.type == "compound_statement":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #     elif z.type == "try_statement":
        #         for j in z.children:
        #             if j.type == "compound_statement":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #     elif z.type == "switch_statement":
        #         for j in z.children:
        #             if j.type == "switch_block":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #             elif j.type == "compound_statement":
        #                 self.G.add_edge(j.end_point[0]+1, node.start_point[0]+1, weight=1)
        #     elif z.type == 'else':
        #         self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
        #     else:
        #         self.visit_piece(z)
        # self.loopflag = node.end_point[0] + 1

    def visit_labeled_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            self.visit_piece(z)

    def visit_goto_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_continue_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if len(self.circle) > 0:
            init_no, end_no = self.circle[-1]
            self.G.add_edge(node.start_point[0] + 1, end_no, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)

    def visit_try_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        body_node = {}
        for z in node.children:

            if z.type == "block":

            # if z.type == "compound_statement":

                self.ast_visit(z)
                body_node['bs'] = z.start_point[0] + 1
                body_node['be'] = z.end_point[0] + 1
            elif z.type == "finally_clause":
                # self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
                self.dece_node.append(z.start_point[0] + 1)
                body_node['fs'] = z.start_point[0] + 1
                body_node['fe'] = z.end_point[0] + 1
                self.ast_visit(z)
            elif z.type == "catch_clause":
                # self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)
                self.dece_node.append(z.start_point[0] + 1)
                body_node['cs'] = z.start_point[0] + 1
                body_node['ce'] = z.end_point[0] + 1
                self.ast_visit(z)
            else:
                self.visit_piece(z)
        if 'be' in body_node and 'cs' in body_node:
            self.G.add_edge(body_node['be'], body_node['cs'], weight=1)
            # for i in range(body_node['bs'], body_node['be']+1):
            #    self.G.add_edge(i, body_node['cs'], weight=1)
        if 'ce' in body_node and 'fs' in body_node:
            self.G.add_edge(body_node['ce'], body_node['fs'], weight=1)
            # for i in range(body_node['cs'], body_node['ce']+1):
            #    self.G.add_edge(i, body_node['fs'], weight=1)
        if 'be' in body_node and 'ce' not in body_node and 'fs' in body_node:
            self.G.add_edge(body_node['be'], body_node['fs'], weight=1)

    def visit_catch_clause(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        for z in node.children:

            if z.type == "block":
                self.ast_visit(z)

            # if z.type == "compound_statement":
            #     self.ast_visit(z)

            else:
                self.visit_piece(z)

    def visit_finally_clause(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        for z in node.children:

            if z.type == "block":
                self.ast_visit(z)

            # if z.type == "compound_statement":
            #     self.ast_visit(z)

            else:
                self.visit_piece(z)

    def visit_throw_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        for z in node.children:
            if z.type == "object_creation_expression":
                self.ast_visit(z)
        if node.end_point[0] + 1 not in self.finlineno:
            self.finlineno.append(node.end_point[0] + 1)

    def visit_object_creation_expression(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_argument_list(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_if_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.next_sibling is not None:  # named_child_count
            self.G.add_edge(node.start_point[0]+1, node.next_sibling.start_point[0]+1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        self.dece_node.append(node.start_point[0] + 1)
        if node.end_point[0] + 1 != self.finlineno[0]:
            self.G.add_edge(node.children[-1].end_point[0] + 1, node.end_point[0] + 2, weight=1)
        for z in node.children:
            if z.type == "else":
                self.dece_node.append(z.start_point[0] + 1)
                self.G.add_edge(node.start_point[0] + 1, z.start_point[0] + 1, weight=1)

            elif z.type == "block":

            # elif z.type == "compound_statement":

                if node.next_sibling is not None:
                    self.G.add_edge(z.end_point[0] + 1, node.next_sibling.start_point[0] + 1, weight=1)
                self.ast_visit(z)
            else:
                self.visit_piece(z)

    def visit_preproc_if(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            if z.type == "#if":
                self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
            elif z.type == "preproc_else":
                self.G.add_edge(node.start_point[0]+1, z.start_point[0]+1, weight=1)
                self.visit_piece(z)
            elif z.type == "#endif":
                self.G.add_edge(node.start_point[0]+1, z.start_point[0]+1, weight=1)
                self.G.add_edge(z.start_point[0], z.start_point[0]+1, weight=1)
            elif z.type == "preproc_elif":
                self.G.add_edge(node.start_point[0]+1, z.start_point[0]+1, weight=1)
                self.ast_visit(z)
            else:
                self.visit_piece(z)

    def visit_preproc_elif(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            self.visit_piece(z)

    def visit_preproc_else(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            self.visit_piece(z)

    def visit_preproc_ifdef(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            if z.type == "#ifdef":
                self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
            elif z.type == "#endif":
                self.G.add_edge(node.start_point[0]+1, z.start_point[0]+1, weight=1)
                self.G.add_edge(z.start_point[0], z.start_point[0]+1, weight=1)
            else:
                self.visit_piece(z)

    def visit_preproc_params(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for z in node.children:
            if z.type == "#define":
                self.G.add_edge(z.start_point[0], z.start_point[0] + 1, weight=1)
            elif z.type == "preproc_arg":
                self.G.add_edge(z.start_point[0], z.start_point[0]+1, weight=1)
                if node.start_point[0] != node.end_point[0]:
                    for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                        self.G.add_edge(j, j + 1, weight=1)
            else:
                self.visit_piece(z)

    def visit_preproc_function_def(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_preproc_call(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_preproc_def(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_preproc_defined(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_preproc_include(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_break_statement(self, node):
        if node.start_point[0] != 0:
            self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.start_point[0] + 2, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for i in range(node.start_point[0] + 1, node.end_point[0] + 1):
                self.G.add_edge(i, i + 1, weight=1)
        if len(self.circle) > 0:
            init_no, end_no = self.circle[-1]
            self.G.add_edge(node.start_point[0] + 1, end_no, weight=1)
        else:
            self.G.add_edge(node.start_point[0] + 1, node.end_point[0] + 1, weight=1)
        if node.end_point[0] + 1 == self.finlineno[-1]:
            self.finlineno.append(node.end_point[0] + 1)

    def visit_ternary_expression(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.start_point[0] != node.end_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_synchronized_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        for index, z in enumerate(node.children):
            if z.type == "block":
                self.ast_visit(z)
            # if z.type == "compound_statement":
            #     self.ast_visit(z)

    def visit_expression_statement(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_local_variable_declaration(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_declaration(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for j in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_return_statement(self, node):
        if node.end_point[0] == self.finlineno[0]:
            self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        else:
            self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
            self.G.add_edge(node.start_point[0], node.end_point[0] + 1, weight=1)
        if node.end_point[0] > node.start_point[0]:
            for i in range(node.start_point[0] + 1, node.end_point[0] + 2):
                self.G.add_edge(i, i + 1, weight=1)
        if node.end_point[0] + 1 not in self.finlineno:
            self.finlineno.append(node.end_point[0] + 1)

    def visit_ERROR(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.start_point[0] != node.end_point[0]:
            for j in range(node.start_point[0], node.end_point[0] + 1):
                self.G.add_edge(j, j + 1, weight=1)
        for z in node.children:
            self.visit_piece(z)

    def visit_parenthesized_expression(self, node):
        self.G.add_edge(node.start_point[0], node.start_point[0] + 1, weight=1)
        if node.start_point[0] != node.end_point[0]:
            for j in range(node.start_point[0], node.end_point[0] + 1):
                self.G.add_edge(j, j + 1, weight=1)

    def visit_generic_type(self, node):
        pass

    def visit_identifier(self, node):
        pass

    def visit_if(self, node):
        pass

    def visit_for(self, node):
        pass

    def visit_binary_expression(self, node):
        pass