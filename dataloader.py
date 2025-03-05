import numpy as np
import torch
from tqdm import tqdm
from torch.nn.init import xavier_normal_

class Data:
    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        """
        Data loading and preparing

        Parameters
        ----------
        train_data - data for training model
        valid_data - data for model validation
        test_data - data for testing model
        data - train, valid and test data together
        entities - list of entities in knowledge graph
        train_relations - relations in train data
        valid_relations - relations in validation data
        test_relations - relations in test data
        relations - all relations in knowledge graph
        """
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False):
        """
        Loads dataset from the directory
        
        Parameters
        ----------
        data_dir - path to directory with dataset
        data_type in [train, valid, test] - train or validation or test dataset
        reverse - Add reverse edges in knowledge graph 
        """
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        """
        Returns list of relations in data
        """
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        """
        Returns list of entities in data
        """
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
