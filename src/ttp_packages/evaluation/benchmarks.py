#!/usr/bin/python
# -*- coding: utf-8 -*-

# src/ttp_packages/evaluation/benchmarks.py

"""Instancias y parámetros usados para benchmarks de evaluación."""

EVAL_INST_FNAMES = [
    
    # a280-ttp
    "a280_n279_bounded-strongly-corr_01.ttp",
    "a280_n279_bounded-strongly-corr_10.ttp",
    "a280_n279_uncorr_01.ttp",
    "a280_n279_uncorr_10.ttp",
    "a280_n279_uncorr-similar-weights_01.ttp",
    "a280_n279_uncorr-similar-weights_10.ttp",
    "a280_n837_bounded-strongly-corr_01.ttp",
    "a280_n837_bounded-strongly-corr_10.ttp",
    "a280_n837_uncorr_01.ttp",
    "a280_n837_uncorr_10.ttp",
    "a280_n837_uncorr-similar-weights_01.ttp",
    "a280_n837_uncorr-similar-weights_10.ttp",


    # berlin52-ttp
    "berlin52_n51_bounded-strongly-corr_01.ttp",
    "berlin52_n51_bounded-strongly-corr_10.ttp",
    "berlin52_n51_uncorr_01.ttp",
    "berlin52_n51_uncorr_10.ttp",
    "berlin52_n51_uncorr-similar-weights_01.ttp",
    "berlin52_n51_uncorr-similar-weights_10.ttp",
    "berlin52_n153_bounded-strongly-corr_01.ttp",
    "berlin52_n153_bounded-strongly-corr_10.ttp",
    "berlin52_n153_uncorr_01.ttp",
    "berlin52_n153_uncorr_10.ttp",
    "berlin52_n153_uncorr-similar-weights_01.ttp",
    "berlin52_n153_uncorr-similar-weights_10.ttp",

    # eil76-ttp
    "eil76_n75_bounded-strongly-corr_01.ttp",
    "eil76_n75_bounded-strongly-corr_10.ttp",
    "eil76_n75_uncorr_01.ttp",
    "eil76_n75_uncorr_10.ttp",
    "eil76_n75_uncorr-similar-weights_01.ttp",
    "eil76_n75_uncorr-similar-weights_10.ttp",

    # u1060-ttp
    "u1060_n1059_bounded-strongly-corr_01.ttp",
    "u1060_n1059_bounded-strongly-corr_10.ttp",
    "u1060_n1059_uncorr_01.ttp",
    "u1060_n1059_uncorr_10.ttp",
    "u1060_n1059_uncorr-similar-weights_01.ttp",
    "u1060_n1059_uncorr-similar-weights_10.ttp",
]

EVAL_CS2SAR_ITERATIONS = 10
EVAL_LIST_TIME_TO_SOL = [30, 60, 300, 420, 600]

EVAL_BIG_SEPARATED = "\n" + ("=" * 100)
EVAL_SEPARATED = "-" * 100