# LossTop
Repository for Skoltech DL project 2021 "Investigation of topological losses for neural text generation"

![The distribution of feature for natural and generated texts during training on 4 layer, 11 head](https://media.giphy.com/media/V9mGnh1b3chIu5YZAT/giphy.gif)

## Requirements installation

The code was test on Python 3.8.5; the requirements can be installed in a new enviroment with:

```
pip3 install -r requirements.txt
```

## GPT-2 finetuning

To finetune GPT-2 on [ShortJokes dataset](https://www.kaggle.com/abhinavmoudgil95/short-jokes), as was done during the project, please refer to [GPT-2 finetune](https://github.com/danchern97/LossTop/blob/main/GPT-2%20finetune.ipynb) notebook.

To examine the MST calculation functions we used, refer to [mst.py](https://github.com/danchern97/LossTop/blob/main/mst.py).
