#!/usr/bin/env python

def count_parameters(model):
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")