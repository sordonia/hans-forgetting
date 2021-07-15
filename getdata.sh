# get glove
if [ $1 = "glove" ]
then
    wget https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
    unzip glove.42B.300d.zip
fi

if [ $1 = "fever" ]
then
    # get fever
    mkdir fever/
    cd fever/
    wget https://www.dropbox.com/s/v1a0depfg7jp90f/fever.train.jsonl
    wget https://www.dropbox.com/s/bdwf46sa2gcuf6j/fever.dev.jsonl
    wget https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.1/fever_symmetric_generated.jsonl
    wget https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.2/fever_symmetric_dev.jsonl
    wget https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.2/fever_symmetric_test.jsonl
    cd ..
    export FEVER_DATA_PATH=$PWD/fever
fi

if [ $1 = "mnli" ]
then
    mkdir mnli/
    cd mnli/
    wget https://dl.fbaipublicfiles.com/glue/data/MNLI.zip
    unzip MNLI.zip
    cd MNLI
    wget https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt
    wget https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_train_set.txt
    cd ..
    cd ..
    export MNLI_DATA_PATH=$PWD/mnli/MNLI/
fi

# qqp and paws need to be downloaded / built manually
