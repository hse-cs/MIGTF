from Models import LPITF, PITF
from dataloder import Data
from Experiment import Experiment

class ExperimentL(Experiment):
    def train_and_eval(self):
        """
        Train procedure for LPITF model
        """
        print("Training the LPITF/PITF model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = LPITF(d, self.emb_dim, self.ro1, self.ro2, self.model_type, **self.kwargs)

        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            print("iteration: ", it)
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])
                e2_idx = torch.tensor(data_batch[:,0])
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2_idx = e2_idx.cuda()

                loss = self.sample_parser(e1_idx, r_idx, e2_idx, targets, model)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print(it)
            print(time.time()-start_train)
            print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                print("Validation:")
                self.evaluate(model, d.valid_data)
                if not it%2:
                    print("Test:")
                    start_test = time.time()
                    self.evaluate(model, d.test_data)
                    print(time.time()-start_test)
