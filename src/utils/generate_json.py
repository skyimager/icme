#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def get_json_of_duplicates(df, key_col, val_col):

    df['dup_first'] = df.duplicated(subset = val_col, keep='first') #marks all duplicates as True except first entry
    df['dup_all'] = df.duplicated(subset = val_col, keep=False) #marks all duplicates as True
    mykeys = list(df[key_col][[not x for x in df.dup_first] & df.dup_all]) #First ID of each duplicated set of records
    def get_duplicates(k):
        return list(df[key_col][df[val_col].isin(df[val_col][df[key_col]==k])])[1:]
    
    return dict(zip(mykeys, [get_duplicates(k) for k in mykeys]))