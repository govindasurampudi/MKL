# MKL
## Multiple Kernel Learning Model for Relating Structural and Functional Connectivity in the Brain

This is the code base for the article in [Nature/Scientific Reports](www.nature.com/articles/s41598-018-21456-0). 

In order to use the MKL model on your data set, you need to ensure few things:

- The dataset should have both the structural and functional connectivity matrices. They are denoted by sCall and fCall respectively. Their size is [<#rois>, <#rois>, <#subjects>].
- to run the model use **run_MKL.m** file.
