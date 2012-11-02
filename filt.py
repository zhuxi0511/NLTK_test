# -*- coding: utf-8 -*-

import mmseg

for i in mmseg.seg_txt('MMSEG是中文分词中一个常见的、基于词典的分词算法（作者主页：http://chtsai.org/index_tw.html），简单、效果相对较好。由于它的简易直观性，实现起来不是很复杂，运行速度也比较快。'):
    print i
