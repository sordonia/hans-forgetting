This is the code for our paper 'Increasing Robustness to Spurious Correlations using Forgettable Examples'.

<!---

Example Training on Fever
--------------


```bash
# download fever data
$ sh getdata.sh fever && export FEVER_PATH=fever/

# download glove
$ sh getdata.sh glove

# create embeddings for base weak models (bow, lstm)
$ python exp_cli.py extract_subset_from_glove glove.42B.300d.txt config/dictionary.txt config/

# train bow model
$ python exp_cli.py train_fever_bow --output_dir fever_bow

# extract forgettables from a bow model
$ python exp_cli.py extract_hard_examples fever_bow/ --train_path $FEVER_PATH/fever.train.jsonl --task fever

# fine-tune a bert base model on fever
$ python exp_cli.py train_fever_bert_base --output_dir fever_bert_base/

# fine-tune the model on forgettables
$ python exp_cli.py finetune_hard_examples fever_bert_base/checkpoint-epoch-1/ fever_bert_base_fbow/ --hard_path fever_bow/hard_examples.pkl
```

-->

Reproducing MNLI -> MNLI & HANS results in the paper (one seed)
--------------


```bash
# download fever data
$ sh getdata.sh mnli && export MNLI_PATH=mnli/MNLI/

# fine-tune a bert base model on mnli 
$ python exp_cli.py train_mnli_bert_base --output_dir mnli_bert_base/

# fine-tune the model on bow forgettables
$ python exp_cli.py finetune_hard_examples mnli_bert_base/checkpoint-epoch-3/ mnli_bert_base_fbow/ --training-examples-ids example_ids/mnli/bow/balanced_forg.ids --task mnli 

# fine-tune the model on lstm forgettables
$ python exp_cli.py finetune_hard_examples mnli_bert_base/checkpoint-epoch-3/ mnli_bert_base_flstm/ --training-examples-ids example_ids/mnli/lstm/balanced_forg.ids --task mnli 

# fine-tune the model on bert forgettables
$ python exp_cli.py finetune_hard_examples mnli_bert_base/checkpoint-epoch-3/ mnli_bert_base_fbert/ --training-examples-ids example_ids/mnli/bert/balanced_forg.ids --task mnli 
```
-----------

To generate the BoW forgettables for MNLI, you can run:

```bash
# download glove
$ sh getdata.sh glove

# create embeddings for base weak models (bow, lstm)
$ python exp_cli.py extract_subset_from_glove glove.42B.300d.txt config/dictionary.txt config/

# train bow model
$ python exp_cli.py train_mnli_bow --output_dir mnli_bow

# extract forgettables from a bow model
$ python exp_cli.py extract_hard_examples mnli_bow/ --train_path $MNLI_PATH/train.tsv --task mnli 
```
and then you can fine-tune your mnli_bert_base checkpoint on your BoW forgettables using:
```bash
$ python exp_cli.py finetune_hard_examples mnli_bert_base/checkpoint-epoch-3/ mnli_bert_base_fbow/ --hard_path mnli_bow/hard_examples.pkl
```
