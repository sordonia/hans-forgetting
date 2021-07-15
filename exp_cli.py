import os
import numpy
import subprocess
import glob
import logging
from pprint import pprint
import inspect
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MNLI_DATA_PATH = os.getenv("MNLI_PATH", "~/workspace/data/multinli_1.0")
FEVER_DATA_PATH = os.getenv("FEVER_PATH", "~/mygit/jiant/data/FEVER/")
PAWSQQP_DATA_PATH = os.getenv("PAWS_QQP_PATH", "~/mygit/jiant/data/soroush_data/extra/datasets/glue/paws/paws-qqp")


class settings(type):
    def __new__(self, name, bases, classdict):
        classdict['fields'] = dict(
            (str(key), str(value)) for key, value in classdict.items()
            if key not in ('__module__', '__qualname__'))
        return type.__new__(self, name, bases, classdict)


class bert_defaults(metaclass=settings):
    per_gpu_eval_batch_size = 180
    per_gpu_train_batch_size = 32
    num_train_epochs = 10
    decay_learning_rate = 'True'
    do_lower_case = 'True'
    learning_rate = 5e-5
    model_name_or_path = 'bert-base-uncased'
    model_type = 'bert'


class bert_large_defaults(metaclass=settings):
    per_gpu_eval_batch_size = 180
    per_gpu_train_batch_size = 16
    num_train_epochs = 10
    decay_learning_rate = 'True'
    do_lower_case = 'True'
    learning_rate = 5e-5
    model_name_or_path = 'bert-large-uncased'
    model_type = 'bert'


class xlnet_defaults(metaclass=settings):
    model_type = 'xlnet',
    num_train_epochs = 10
    model_name_or_path = "xlnet-base-cased"
    learning_rate = 3e-5
    per_gpu_train_batch_size = 16,
    do_lower_case = "False"


class xlnet_large_defaults(metaclass=settings):
    model_type = 'xlnet',
    num_train_epochs = 10
    model_name_or_path = "xlnet-large-cased"
    learning_rate = 1e-5
    per_gpu_train_batch_size = 16
    do_lower_case = "False"


class lstmatt_defaults(metaclass=settings):
    model_type = 'baseline'
    model_name_or_path = 'lstm-att'
    learning_rate = 0.0005
    num_train_epochs = 5
    per_gpu_train_batch_size = 256
    do_lower_case = 'False'
    config_name = './config/lstmatt_small_config.json'


class bilstm_defaults(metaclass=settings):
    model_type = 'baseline'
    model_name_or_path = 'bilstm'
    learning_rate = 0.0005
    num_train_epochs = 5
    per_gpu_train_batch_size = 256
    do_lower_case = 'False'
    config_name = './config/lstmatt_small_config.json'


class bow_defaults(metaclass=settings):
    model_type = 'baseline'
    model_name_or_path = 'bow'
    learning_rate = 0.001
    num_train_epochs = 5
    per_gpu_train_batch_size = 256
    do_lower_case = 'False'
    config_name = './config/lstmatt_small_config.json'


class mnli_defaults(metaclass=settings):
    data_dir = f'{MNLI_DATA_PATH}'
    fp16 = ''
    task_name = 'mnli'
    do_train = ''
    overwrite_output_dir = ''
    per_gpu_eval_batch_size = 128
    num_train_epochs = 4


class pawsqqp_defaults(metaclass=settings):
    data_dir = f'{PAWSQQP_DATA_PATH}'
    fp16 = ''
    task_name = 'qqp'
    do_train = ''
    overwrite_output_dir = ''
    eval_tasks = 'qqp-wang qqp-wang-test paws-qqp paws-wiki paws-qqp-all-val'
    learning_rate = '5e-5'
    num_train_epochs = 3
    weight_decay = 0.0
    per_gpu_train_batch_size = 32
    per_gpu_eval_batch_size = 400


class fever_defaults(metaclass=settings):
    data_dir = f'{FEVER_DATA_PATH}'
    fp16 = ''
    task_name = 'fever'
    do_train = ''
    overwrite_output_dir = ''
    eval_tasks = 'fever fever-symmetric-r1 fever-symmetric-r2'
    learning_rate = '2e-5'
    num_train_epochs = 2
    max_seq_length = 128
    weight_decay = 0.0
    per_gpu_train_batch_size = 32
    per_gpu_eval_batch_size = 200
    warmup_proportion = 0.


class fever_test_defaults(metaclass=settings):
    data_dir = f'{FEVER_DATA_PATH}'
    fp16 = ''
    task_name = 'fever'
    overwrite_output_dir = ''
    eval_tasks = 'fever fever-symmetric-r1 fever-symmetric-r2'
    per_gpu_eval_batch_size = 400


def execute(entry_point, kwargs):
    pprint(kwargs)
    args = ' '.join(f'--{str(k)} {str(v)}' for k, v in kwargs.items())
    print(f"python {entry_point} {args}")
    os.system(f"python {entry_point} {args}")


class Main():
    def extract_subset_from_glove(self, glove_path, dictionary, output_dir):
        """Extracts a subset of vectors from the full glove dictionary
           and stores them in output_dir/embeddings.pkl
        """
        from models_weak import extract_subset_from_glove
        extract_subset_from_glove(glove_path, dictionary, output_dir)

    def extract_hard_examples(
            self,
            example_stats_path,
            labels_file=None,
            train_path=None,
            task='mnli'
        ):
        """Given a model examples stats, filter all examples if unlearnt after epoch_num,
        and store an example file in the specified directory.
        """

        import pickle
        import numpy as np
        from pathlib import Path
        import pandas as pd 

        output_path = example_stats_path + '/hard_examples.pkl'
        examples_stats = pickle.load(open(example_stats_path + '/example_stats.pkl', 'rb'))
        n_epochs = examples_stats['accuracy'].shape[1]
        print("Loaded example stats,", examples_stats.keys())

        if labels_file:
            labels = open(labels_file, 'r').readlines()
            labels_dict = dict()
            for line in labels:
                id, label = line.strip().split()
                labels_dict[int(id)] = int(label)
        if train_path:
            if task == 'mnli':
                df = pd.read_csv(
                        train_path,
                        sep='\t',
                        error_bad_lines=False,
                        skiprows=0,
                        quoting=3,
                        keep_default_na=False,
                        encoding="utf-8",)
            elif task == 'fever':
                import json
                with open(train_path, 'r') as f:
                    data = [json.loads(s.strip()) for s in f.readlines()]
                df = pd.DataFrame(data)
            labels_dict = dict()
            for id, label in enumerate(df.gold_label):
                labels_dict[int(id)] = label

        def balance_by_class(hard_ids):
            by_label = dict()
            for id in hard_ids:
                label = labels_dict[id]
                arr = by_label.get(label, [])
                arr.append(id)
                by_label[label] = arr
            min_num = np.min([len(arr) for arr in by_label.values()])
            balanced_ids = []
            for arr in by_label.values():
                balanced_ids.extend(arr[:min_num])
            return np.array(balanced_ids)

        def select_unlearnt_after_n_epochs(n_epoch):
            accuracy = examples_stats['accuracy'][:, n_epoch:]
            accuracy_min = np.min(accuracy, 1)
            hard_indices = np.where(accuracy_min == 0)[0]
            return hard_indices

        def select_by_loss():
            end_loss = examples_stats['loss'][:, -1]
            indices_by_loss = np.argsort(end_loss)[::-1]
            return indices_by_loss

        def select_forgettables():
            from utils_forgetting import compute_forgetting
            f, c, m = compute_forgetting(examples_stats['accuracy'])
            never_learnt = np.where(c == m)[0]
            forgettables = f
            return forgettables, never_learnt

        results = {}
        for n_epoch in range(n_epochs):
            results[f'not_learnt_after_epc_{n_epoch}'] = select_unlearnt_after_n_epochs(n_epoch)
            results[f'not_learnt_after_epc_{n_epoch}_b'] = balance_by_class(results[f'not_learnt_after_epc_{n_epoch}'])

        f, u = select_forgettables()
        ordered_by_loss = select_by_loss()
        num_examples = len(ordered_by_loss)
        for perc in [1, 5, 10, 25, 50, 75]:
            results[f'top_{perc}%_loss'] = ordered_by_loss[:int((float(perc) * num_examples) / 100)]
            results[f'top_{perc}%_loss_b'] = balance_by_class(results[f'top_{perc}%_loss'])

        results['forgettables'] = f
        results['forgettables_b'] = balance_by_class(f)
        results['never_learnt'] = u
        results['never_learnt_b'] = balance_by_class(u)

        for key, ids in results.items():
            print(key, '=', ids[:5], ',', len(ids))

        with open(output_path, "wb") as f:
           pickle.dump(results, f)

    ###########
    #   MNLI  #
    ###########
    def train_mnli_bow(self, output_dir, config_name="./config/lstmatt_small_config.json", seed=0):
        """You want to probably run first:
           python exp_cli.py extract_subset_from_glove glove.txt config/dictionary.txt config/
        """
        args = bow_defaults.fields
        args.update(mnli_defaults.fields)
        args.update(num_train_epochs=3)
        args.update(dict(config_name=config_name, output_dir=output_dir, seed=seed))
        execute("exp_glue.py", args)
    
    def train_mnli_lstmatt(self, output_dir, config_name="./config/lstmatt_small_config.json", seed=0):
        """You want to probably run first:
           python exp_cli.py extract_subset_from_glove glove.txt config/dictionary.txt config/
        """
        args = lstmatt_defaults.fields
        args.update(mnli_defaults.fields)
        args.update(dict(config_name=config_name, output_dir=output_dir, seed=seed))
        execute("exp_glue.py", args)

    def train_mnli_bert_base(self, output_dir, seed=0):
        args = bert_defaults.fields
        args.update(mnli_defaults.fields)
        args.update(dict(output_dir=output_dir, seed=seed))
        execute("exp_glue.py", args)

    def train_mnli_xlnet_base(self, output_dir, seed=0):
        args = xlnet_defaults.fields
        args.update(mnli_defaults.fields)
        args.update(dict(output_dir=output_dir, seed=seed))
        execute("exp_glue.py", args)

    def train_mnli_xlnet_large(self, output_dir, seed=0):
        args = xlnet_large_defaults.fields
        args.update(mnli_defaults.fields)
        args.update(dict(output_dir=output_dir, seed=seed))
        execute("exp_glue.py", args)
    
    ###########
    #  FEVER  #
    ###########
    def train_fever_bow(self, output_dir, config_name="./config/lstmatt_small_config.json", seed=0):
        """You want to probably run first:
           python exp_cli.py extract_subset_from_glove glove.txt config/dictionary.txt config/
        """
        args = bow_defaults.fields
        args.update(fever_defaults.fields)
        args.update(num_train_epochs=5)
        args.update(dict(config_name=config_name, output_dir=output_dir, seed=seed))
        execute("exp_glue.py", args)
        
    def train_fever_bilstm(self, output_dir, config_name="./config/lstmatt_small_config.json", seed=0):
        """You want to probably run first:
           python exp_cli.py extract_subset_from_glove glove.txt config/dictionary.txt config/
        """
        args = bilstm_defaults.fields
        args.update(fever_defaults.fields)
        args.update(num_train_epochs=5)
        args.update(dict(config_name=config_name, output_dir=output_dir, seed=seed))
        execute("exp_glue.py", args)
    
    def train_fever_bert_base(self, output_dir, seed=0):
        args = bert_defaults.fields
        args.update(fever_defaults.fields)
        args.update(dict(output_dir=output_dir, seed=seed))
        execute("exp_glue.py", args)

    def finetune_hard_examples(
        self,
        base_model_path,
        output_dir,
        base_model_type="bert_base",
        hard_path="",
        hard_type="forgettables_b",
        training_examples_ids=None,
        seed=0,
        task='fever'
    ):
        """Finetune a base model, e.g. bert, on hard examples
           from a weaker model, e.g. bow.
        """
        from pathlib import Path

        if base_model_type == "bert_base":
            args = bert_defaults.fields
        else:
            assert False

        if task == 'fever':
            args.update(fever_defaults.fields)
        elif task == 'mnli':
            args.update(mnli_defaults.fields)

        args.update(dict(num_train_epochs=4, learning_rate=5e-6, per_gpu_train_batch_size=100))
        args.update(dict(load_model=base_model_path))
        args.update(dict(output_dir=Path(output_dir) / base_model_type, seed=seed))
        args.update(dict(training_examples_ids=training_examples_ids))
        if hard_path:
            args.update(dict(hard_examples=hard_path, hard_type=hard_type))

        # fine-tune base model on hard examples
        execute("exp_glue.py", args)
        
    def test(
            self, base_model_path,
            base_model_type="bert_base",
            task='fever', dev=False
        ):
        """Test a base model on testset.

           Task can be 'fever', 'fever-symmetric-r1', 'fever-symmetric-r2', 'mnli', etc..
        """
        from pathlib import Path
        if base_model_type == "bert_base":
            args = bert_defaults.fields
        elif base_model_type == "bert_large":
            args = bert_large_defaults.fields
        elif base_model_type == "xlnet_base":
            args = xlnet_defaults.fields
        elif base_model_type == "xlnet_large":
            args = xlnet_large_defaults.fields
        else:
            assert "Base model not valid: %s" % base_model_type
        if task == 'mnli':
            args.update(mnli_defaults.fields)
        elif 'fever' in task:
            args.update(fever_test_defaults.fields)

        args.update(dict(eval_tasks=task))
        args.update(dict(load_model=base_model_path))
        args.update(dict(output_dir=base_model_path))
        args.update(dict(output_dir=base_model_path))
        if 'do_train' in args:
            args.pop('do_train')
        if not dev:
            args.update(dict(test=""))
        args.update(dict(per_gpu_eval_batch_size="100"))

        # train base model on hard examples
        execute("exp_glue.py", args)


if __name__ == '__main__':
    import fire
    fire.Fire(Main)
