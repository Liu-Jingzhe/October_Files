Regarding the datasets, it is hard to share large files anonymously. We recommand to download the datasets from another publication, "Text-space Graph Foundation Models: Comprehensive Benchmarks and New Insights". Please download the datasets from their [github link](https://github.com/CurryTang/TSGFM). 

After downloading the datasets, please unzip the folder and place them in the same directory with the codes. Note that their datasets share the same pre-processed features with ours on the node classification tasks. Thus we can run experiments on the datasets for zero-shot node classification.

To train a expert model on a dataset with `dataname`, run
```
python train.py --dataset dataname
```

To train a gate model on the same dataset, run
```
python train_gate.py --dataset dataname
```

To do zero-shot predictions on a dataset `testdata`, run
```
python infer.py -dataset testdata
```

During the inference stage, the model will use the following datasets' experts and gates by default:
```
 dataset_list = ['cora','citeseer','dblp','bookchild','bookhis','products','pubmed','sportsfit','wikics']
```
Please change it (line 334 in `infer.py`) based on actual needs.


We list our evironment dependency here
```
Package               Version
--------------------- ----------------
accelerate            0.31.0
aiohttp               3.9.5
aiosignal             1.3.1
annotated-types       0.7.0
args                  0.1.0
asttokens             2.4.1
async-timeout         4.0.3
attrs                 23.2.0
beautifulsoup4        4.12.3
certifi               2022.12.7
charset-normalizer    2.1.1
click                 8.1.7
clint                 0.5.1
cloudpickle           3.0.0
ConfigArgParse        1.7
contourpy             1.2.1
cycler                0.12.1
cytoolz               0.12.3
dask                  2024.5.2
datasets              2.20.0
datasketch            1.6.4
decorator             5.1.1
deepspeed             0.14.0
Deprecated            1.2.14
dill                  0.3.8
docker-pycreds        0.4.0
evaluate              0.4.2
exceptiongroup        1.2.1
executing             2.0.1
fast-pagerank         1.0.0
filelock              3.13.1
fonttools             4.53.0
frozenlist            1.4.1
fsspec                2024.2.0
future                1.0.0
gcl                   0.6.11
gdown                 4.7.1
gitdb                 4.0.11
GitPython             3.1.43
hjson                 3.1.0
huggingface-hub       0.23.2
idna                  3.4
importlib_metadata    7.1.0
ipdb                  0.13.13
ipython               8.25.0
jedi                  0.19.1
Jinja2                3.1.3
joblib                1.3.2
kiwisolver            1.4.5
lightning-utilities   0.11.2
littleutils           0.2.2
llm2vec               0.1.8
llvmlite              0.42.0
lmdb                  1.4.1
locket                1.0.0
MarkupSafe            2.1.5
matplotlib            3.7.2
matplotlib-inline     0.1.7
mpmath                1.3.0
multidict             6.0.5
multiprocess          0.70.16
networkx              3.1
ninja                 1.11.1.1
nltk                  3.8.1
numba                 0.59.1
numpy                 1.22.4
ogb                   1.3.6
outdated              0.2.2
packaging             24.1
pandas                1.5.3
parso                 0.8.4
partd                 1.4.2
peft                  0.11.1
pexpect               4.9.0
pillow                10.2.0
pip                   24.0
platformdirs          4.2.2
prompt_toolkit        3.0.47
protobuf              5.27.1
psutil                5.9.8
ptyprocess            0.7.0
pure-eval             0.2.2
py-cpuinfo            9.0.0
py-tgb                2.0.0
pyarrow               16.1.0
pyarrow-hotfix        0.6
pybind11              2.13.6
pydantic              2.7.4
pydantic_core         2.18.4
pyg-lib               0.4.0+pt21cu118
PyGCL                 0.1.2
Pygments              2.18.0
pynndescent           0.5.12
pynvml                11.5.0
pyparsing             3.0.9
PySocks               1.7.1
python-dateutil       2.9.0.post0
pytorch-lightning     2.2.1
pytorch-warmup        0.1.1
pytz                  2024.1
PyYAML                6.0.1
rdkit                 2023.9.2
regex                 2024.5.15
requests              2.32.3
safetensors           0.4.3
scikit-learn          1.3.2
scipy                 1.9.0
seaborn               0.13.2
sentence-transformers 2.2.2
sentencepiece         0.2.0
sentry-sdk            2.7.1
setproctitle          1.3.3
setuptools            69.5.1
shortuuid             1.0.13
six                   1.16.0
smmap                 5.0.1
soupsieve             2.5
stack-data            0.6.3
sympy                 1.12
tensorboardX          2.6.2.2
threadpoolctl         3.5.0
tokenizers            0.19.1
tomli                 2.0.1
toolz                 0.12.1
torch                 2.1.0+cu118
torch-cluster         1.6.3+pt21cu118
torch_geometric       2.5.3
torch-scatter         2.1.2+pt21cu118
torch-sparse          0.6.18+pt21cu118
torch-spline-conv     1.2.2+pt21cu118
torchaudio            2.1.0+cu118
torchmetrics          1.3.0
torchvision           0.16.0+cu118
tqdm                  4.66.4
traitlets             5.14.3
transformers          4.41.2
triton                2.1.0
typing_extensions     4.9.0
tzdata                2024.1
umap-learn            0.5.6
urllib3               1.26.13
wandb                 0.17.4
wcwidth               0.2.13
wheel                 0.43.0
wrapt                 1.16.0
xxhash                3.4.1
yarl                  1.9.4
zipp                  3.19.2
```
