from dataloader import Data


class Experiment:
    """
    Experiments setup and functions

    Attributes
    ----------
    learning_rate - learning rate
    ent_vec_dim - Euclidean entity embeddings dimension
    rel_vec_dim - Euclidean relation embeddings dimension
    hyp_emb_dim - Hyperbolic embedding dimension
    ro1 - hyperbolic entity embedding init parameter
    ro2 - hyperbolic relation embedding init parameter
    qr - apply QR decomposition for hyperbolic relation embeddings [True or False]
    c - Curvature of Loretz hyperbolic model
    num_iterations - number of epoches
    batch_size - batch size
    decay_rate - decay rate
    cuda - GPU [True or false]
    """
    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 hyp_emb_dim = 50, ro1 = 0.0001, ro2 = 0.0001, qr = False, c = 1.3,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., sample_type = "uno"):
        """
        Parameters
        ----------
        learning_rate - learning rate
        ent_vec_dim - Euclidean entity embeddings dimension
        rel_vec_dim - Euclidean relation embeddings dimension
        hyp_emb_dim - Hyperbolic embedding dimension
        ro1 - hyperbolic entity embedding init parameter
        ro2 - hyperbolic relation embedding init parameter
        qr - apply QR decomposition for hyperbolic relation embeddings [True or False]
        c - Curvature of Loretz hyperbolic model
        num_iterations - number of epoches
        batch_size - batch size
        decay_rate - decay rate
        cuda - GPU [True or false]
        input_dropout - Input layer dropout
        hidden_dropout1 - Dropout after the first hidden layer
        hidden_dropout2 - Dropout after the second hidden layer
        label_smoothing - Amount of label smoothing
        sample_type - type of data sampling [uno or bi]
        """
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.hyp_emb_dim = hyp_emb_dim
        self.c = c
        self.ro1 = ro1
        self.ro2 = ro2
        self.qr = qr
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}

    def get_data_idxs(self, data):
        """
        Forms vocab Triple - Index
        Key: Triple Entity-Relation-Entity from data
        Value: index of triple in data
        """
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        """
        Forms vocab Pair - Entity
        Key: Pair of Entity (e) and Relation (r) from data
        Value: list of Entities connected with e with relation r
        """
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, idx_vocab, idx):
        """
        Forms batch

        Parameters
        ----------
        er_vocab - vocab Pair - Entity
        idx_vocab - vocab Triple - Index
        idx - index of first batch element in data
        """
        batch = idx_vocab[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets


    def evaluate(self, model, data, flag = 0):
        """
        Evaluate metrics
        print metrics: HR@10, HR@3, HR@1, MR and MRR

        Parameters
        ----------
        model - model
        data - data for evaluation (Valid or Test)
        flag - evaluate on Test [0, 1]
        """
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        

        for i in range(0, len(test_data_idxs), self.batch_size):

            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))
