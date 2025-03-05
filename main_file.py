from Models import TuckER, LPITF, PITF, MIGTF
from dataloader import Data
from MIGTF_Experiment import ExperimentM
from Hyperbolic_Experiment import ExperimentL
from TuckER_Experiment import ExperimentT


class Argument: 
    def __init__(self):
        """
        Hyperparameters for Experiment with TuckER 
        and Mixed Geometry models

        Attributes
        ----------
        num_iterations - Number of eposhes
        batch_size - Batch_size
        lr - Learning rate
        dr - Decay rate
        edim - Euclidean Entity embedding dimensionality
        rdim - Euclidean Relation embedding dimensionality
        d - Hyperbolic embedding dimensionality
        ro1 - Hyperbolic entity embedding init parameter
        ro2 - Hyperbolic relation embedding init parameter
        qr - Apply QR decomposition to hyperbolic relation embeddings
        c - Curvature of Lorentz Hyperbolic model 
        cuda - GPU [True, False]
        input_dropout - Input layer dropout
        hidden_dropout1 - Dropout after the first hidden layer
        hidden_dropout2 - Dropout after the second hidden layer
        label_smoothing - Amount of label smoothing
        sampling_type - sampling strategy

        """
        self.num_iterations = 500
        self.batch_size = 128
        self.lr = 0.003
        self.dr = 1.0
        self.edim = 200
        self.rdim = 30
        self.d = 50
        self.ro1 = 0.2
        self.ro2 = 0.2
        self.qr = False
        self.c = 1.3
        self.cuda = True
        self.input_dropout = 0.2
        self.hidden_dropout1 = 0.2
        self.hidden_dropout2 = 0.3
        self.label_smoothing = 0.1
        self.sample_type = "bi"
        
        
args = Argument()  # Arguments
dataset = "WN18RR"  # Dataset
data_dir = "data/%s/" % dataset # Dtaset directory
torch.backends.cudnn.deterministic = True
seed = 20 # Random seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
d = Data(data_dir=data_dir, reverse=True)

experiment = ExperimentH(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr,
                                    decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, 
                                    hyp_emb_dim=args.d, ro1=args.ro1, ro2=args.ro2, qr=args.qr, c=args.c, cuda=args.cuda,
                                    input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1,
                                    hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing,
                                sample_type = args.sample_type)

experiment.train_and_eval()