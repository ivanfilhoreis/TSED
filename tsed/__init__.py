# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.
"""

class tsed():
    
    def __init__(self, ts, text):
        self.ts = ts;
        self.text = text;
        
    def __str__(self):
        return "Module: TSED (Time-Series Enriched with Domain-specific terms)"
        



def main():
    
    tt = tsed(0, "teste")
    
    print(tt)
    
    
main()