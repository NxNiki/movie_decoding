## movie decoding

Use a visual transformer (ViT) for brain decoding. This code is adapted from the work by Yuanyi Ding and Chenda Duan.



### How to use

Install conda, python, and poetry:
```
sh src/tools/setup_sge_env.sh
```

Initialize Conda
Once installation is complete, initialize Conda by running:

```
~/miniconda3/bin/conda init
```
This command will add Conda to your PATH, making it available in future terminal sessions.

Activate Conda
To make the changes take effect, you can either open a new terminal or run:
```
source ~/.bashrc
```

Start conda environment:
```
conda create --name movie_decoding python=3.10
```

>> You only need to do the above steps the first time you run this project on your machine.

Activate conda environment:
```
conda activate movie_decoding
```

install libraries with poetry:

```
poetry install
```



