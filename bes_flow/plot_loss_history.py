# bes_flow/plot_loss_history.py
# Plot training loss history from a json file

import json
import argparse
from bes_flow.config import cfg
from bes_flow.train import plot_loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot training loss history'
    )
    
    parser.add_argument('--history_path', type=str,
                        help='Path to json history file')
    args = parser.parse_args()
    
    with open(args.history_path, 'r') as file:
        loss_history = json.load(file)
        print(f'\nLoaded {args.history_path}')

    plot_loss_history(loss_history, cfg)
