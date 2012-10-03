import math

import unittest

"""
Name: AMath
Author: Christian Gram Kalhauge : kalhauge@cbs.dtu.dk 
Date : 14th September - 2012

Math is a collection of methods used in math.

"""

def gammalog(n):
   return math.log(math.factorial(n-1));

def binomiallog(n,k):
   return gammalog(n+1) - (gammalog(k+ 1) + gammalog(n-k+1));

def binomial(n,k):
   if n < k: return 0;
   return round(math.exp(binomiallog(n,k)));

def hypergeometric(k,G,n,N):
   return math.exp(binomiallog(G,k) + binomiallog(N-G,n-k) - binomiallog(N,n));


class MathTester (unittest.TestCase):
   
   def testBinomial(self):
      self.assertEqual(binomial(2,3),0);
      self.assertEqual(binomial(3,2),3);
      self.assertEqual(binomial(15,4),1365);
      self.assertEqual(binomial(52,16),10363194502115);
      
   
   def testHypergeometric(self):
      self.assertAlmostEqual(hypergeometric(1,4,2,10),0.5333333333,10);
      self.assertAlmostEqual(hypergeometric(10,15,20,64),0.00125782191601423,10);
      self.assertAlmostEqual(hypergeometric(10,15,80,102),0.124300105660462,10);


if __name__ == "__main__":
   unittest.main();