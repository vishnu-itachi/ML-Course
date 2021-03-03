# %%
import numpy as np
import pandas as pd
import re

msgtype = []
msg = []
with open("Assignment_4_data.txt", 'r') as f, open("stopwords.txt", 'r') as s:
    for line in f:
        # print(line)
        l = list(filter(None, re.split(r'\s|\t|[.,-:]|\d+', line)))
        msg.append(l[1:])
        msgtype.append(l[0])
    stopwords = list(filter(None, re.split(r'\n', s.read())))

# %%
newmsg = []
for i in msg:
    newmsg.append(list(set(i)-set(stopwords)))


# %%
