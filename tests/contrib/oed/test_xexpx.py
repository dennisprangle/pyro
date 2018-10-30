#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:52:27 2018

@author: charlinelelan
"""
import torch

from pyro.contrib.oed.eig import xexpx


def test_xexpx():
    input1 = torch.tensor([float('-inf')])
    output1 = torch.tensor([0.])
    assert xexpx(input1) == output1
    input2 = torch.tensor([0.])
    output2 = torch.tensor([0.])
    assert xexpx(input2) == output2
    input3 = torch.tensor([1.])
    output3 = torch.exp(torch.tensor([1.]))
    assert xexpx(input3) == output3
