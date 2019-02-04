import ast

from . import utils


class AttrEntry:
    def __init__(self, name, node):
        self.name = name
        self.node = node

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class AttrOrderer:
    LEFT_FIELD_MAP = {
        ast.Attribute: 'value',
        ast.Call: 'func',
        ast.Subscript: 'value',
    }

    def __init__(self, node):
        self.attribute_chain = self.get_attribute_chain(node)

    def __getitem__(self, i):
        return self.attribute_chain[i]

    def __len__(self):
        return len(self.attribute_chain)

    def __iter__(self):
        return iter(self.attribute_chain)

    @classmethod
    def get_left_field(cls, node):
        return AttrOrderer.LEFT_FIELD_MAP.get(type(node))

    @classmethod
    def reorder_node(cls, node, left_field):
        left_node = getattr(node, left_field)

        if type(left_node) is not ast.Attribute:
            left_node = cls.reorder(left_node)

        if type(left_node) is ast.Attribute:
            attr = left_node.attr
            left_node.attr = node
            setattr(node, left_field, attr)

            return cls.reorder(left_node)

        setattr(node, left_field, left_node)
        return node

    @classmethod
    def reorder(cls, node):
        left_field = cls.get_left_field(node)
        if left_field:
            return cls.reorder_node(node, left_field)
        return node

    @classmethod
    def get_node_name(cls, node):
        left_field = cls.get_left_field(node)
        if left_field:
            return cls.get_node_name(getattr(node, left_field))
        return utils.py2okl(node)

    @classmethod
    def get_node_attribute_chain(cls, node):
        if type(node) is ast.Attribute:
            return [node.value, *cls.get_node_attribute_chain(node.attr)]

        left_field = cls.get_left_field(node)
        if not left_field:
            return [node]

        left_chain = cls.get_node_attribute_chain(getattr(node, left_field))
        if len(left_chain) == 1:
            return [node]

        setattr(node, left_field, left_chain[-1])
        return [*left_chain[:-1], node]

    @classmethod
    def get_attribute_chain(cls, node):
        node = cls.reorder(node)
        attr_chain_nodes = cls.get_node_attribute_chain(node)
        return [
            AttrEntry(cls.get_node_name(chain_node), chain_node)
            for chain_node in attr_chain_nodes
        ]
