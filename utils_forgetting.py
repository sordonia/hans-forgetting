import numpy as np
import fire
import pickle
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_forgetting(accuracies):
    forgetting_max = accuracies.shape[1]
    num_examples = accuracies.shape[0]
    forgetting = np.zeros(num_examples)

    for example in range(num_examples):
        never_learnt = (accuracies[example].sum() == 0)
        if never_learnt:
            forgetting[example] = forgetting_max
        else:
            num_forgetting_events = 0
            last_acc = 0
            num_present = 0
            for current_acc in accuracies[example]:
                if current_acc == -1:
                    break
                if current_acc == 0 and last_acc == 1:
                    num_forgetting_events += 1
                num_present += 1
                last_acc = current_acc
            assert num_present == accuracies.shape[1]
            forgetting[example] = num_forgetting_events

    most_forgotten_id = np.argsort(forgetting)[::-1]
    most_forgotten_count = np.take(forgetting, most_forgotten_id)
    logger.info("%d forgettables", (most_forgotten_count > 0).sum())
    logger.info("%d never learnt", (most_forgotten_count == forgetting_max).sum())
    return np.where(forgetting > 0)[0], forgetting, forgetting_max
