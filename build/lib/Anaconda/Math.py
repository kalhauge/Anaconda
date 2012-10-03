import math

"""
Name: Math
Author: Christian Gram Kalhauge : kalhauge@cbs.dtu.dk 
Date : 14th September - 2012

Math is a collection of methods used in math.

"""

def gammaln(n):
   return math.ln(factorial(n-1));

def binomialln(n,k):
   return gammaln(n+1) - (gammaln(k+ 1) + gammaln(n-k+1));

def hypergeometric(x,r,b,n):
   return math.exp(binomialln(r,x) + binomialln(b,n-x) - binomialln(r+b,n));