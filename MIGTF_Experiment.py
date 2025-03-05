from dataloader import Data
from Models import MIGTF, TuckER
from Experiment import Experiment

class ExperimentM(Experiment):
    def train_and_eval(self):
        """
        Train procedure for MIGTF model
        Uses pretrained TuckER model with frozen weights
        """
        print("Training the MIGTF model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model_base = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        model_state = torch.load('TuckER_weights.pth')
        model_base.load_state_dict(model_state)

        if self.cuda:
            model_base.cuda()

        E_pr = model_base.E
        R_pr = model_base.R
        W_pr = model_base.W
        model = MIGTF(d, self.ent_vec_dim, self.rel_vec_dim, self.hyp_emb_dim, self.c, E_pr, R_pr, W_pr, **self.kwargs)

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
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))
                loss = model.loss(predictions, targets) + 0.0001 * nnn1
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
                self.evaluate(model, d.valid_data, 0)
                if not it%2:
                    print("Test:")
                    start_test = time.time()
                    self.evaluate(model, d.test_data, 1)
                    print(time.time()-start_test)
