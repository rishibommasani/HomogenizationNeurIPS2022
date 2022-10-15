CONDA_ENV = "rishi"

nlprun -a rishi -p high -q john -o lsac_svm.out 'python3 lawschool_experiments.py -m svm'
nlprun -a rishi -p high -q john -o lsac_nn.out 'python3 lawschool_experiments.py -m nn'
nlprun -a rishi -p high -q john -o lsac_gbm.out 'python3 lawschool_experiments.py -m gbm'
nlprun -a rishi -p high -q john -o lsac_logi.out 'python3 lawschool_experiments.py -m logistic'