""" Arguments for MAC Network """

from argparse import ArgumentParser
parser = ArgumentParser()


parser.add_argument('--config',
	required=True, type=str,
	help='Name of config set.'
)
parser.add_argument('--dataset_root',
	required=True, type=str,
	help='CLEVR dataset directory.'
)
parser.add_argument('--name',
	required=True, type=str,
	help='Name of run.'
)

parser.add_argument('--seed',
	default=None, type=int,
	help='Random seed for reproducibility.'
)
parser.add_argument('--batch_size',
	default=None, type=int,
	help='Batch size'
)
parser.add_argument('--n_epoch',
	default=None, type=int,
	help='Number of training epochs.'
)
parser.add_argument('--dim',
	default=None, type=int,
	help='Model dimension.'
)
parser.add_argument('--lr',
	default=None, type=float,
	help='Learning rate.'
)
parser.add_argument('--self_attention',
	action='store_true',
	help=''
)
parser.add_argument('--memory_gate',
	action='store_true',
	help=''
)
parser.add_argument('--embed_hidden',
	default=None, type=int,
	help=''
)
parser.add_argument('--max_step',
	default=None, type=int,
	help=''
)
parser.add_argument('--classes',
	default=None, type=int,
	help=''
)
parser.add_argument('--dropout',
	default=None, type=float,
	help=''
)
parser.add_argument('--wandb',
  default=None, type=str,
  help='Wandb entity for logging.'
)
parser.add_argument('--activation',
  default=None, type=str,
  help='Softmax or sparsemax attention.'
)

