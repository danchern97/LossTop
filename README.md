# LossTop
Repository for Skoltech Deep Learning course project "Investigation of topological losses for neural text generation" 2021

![The distribution of feature for natural and generated texts during training on 4 layer, 11 head](https://media.giphy.com/media/V9mGnh1b3chIu5YZAT/giphy.gif)

## Authors

Daniil Cherniavskii, Saveliy Galochkin, Yehor Kravets, Lev Kosolapov, Ramilya Sharifullina, Dmitry Kryukov, Georgiy Kozhevniko

## Requirements installation

The code was test on Python 3.8.5; the requirements can be installed in a new enviroment with:

```
pip3 install -r requirements.txt
```

## GPT-2 finetuning

To finetune GPT-2 on [ShortJokes dataset](https://www.kaggle.com/abhinavmoudgil95/short-jokes), as was done during the project, please refer to [GPT-2 finetune](https://github.com/danchern97/LossTop/blob/main/GPT-2%20finetune.ipynb) notebook.

To examine the MST calculation functions we tried using, refer to [mst.py](https://github.com/danchern97/LossTop/blob/main/mst.py).

## Finetuned models

Both default finetuned GPT-2 and the one trained with topological loss are localted [here](https://drive.google.com/drive/folders/1FlkIAoY8zWC7T9E1j18uYZ0N1q32A75a?usp=sharing)

## Some explanations

Calculating original topological features is very time consuming, so we used an approximation to H0s, that is the sum of weights above threshold:

![equation](https://latex.codecogs.com/gif.latex?f%20%3D%20%5Csum_%7Bi%2Cj%3D1%7D%20W_%7Bij%7D%20%5Cmathbf%7B1%7D%5BW_%7Bij%7D%20%3E%20t%5D)

The final topological loss is defined as 

![equation](https://latex.codecogs.com/gif.latex?L_%7B%5Ctext%7BTop%7D%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Ctext%7Bbatch%5C_size%7D%7D%5Csum_%7Bi%3D1%7D%5E%7B%5Ctext%7Bbatch%5C_size%7D%7D%20%28f%5En_i%20-%20f%5Eg_i%29%5E2)
