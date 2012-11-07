import time
import numpy as np
import unittest
import logging
import Image,ImageDraw
import AMath

log = logging.getLogger(__name__)

"""
Name: SOM
Author: Christian Gram Kalhauge : kalhauge@cbs.dtu.dk 
Date : 13th August - 2012

An Implementation of Self Organizing Maps (SOM).
"""


   
class SOM:
   def __init__(self):
      self.size = (1,1);
      self.isHexagon = False;
      self.grid = None;
      self.vector_size = 1;
      
   def createGrid(self,size,vector_size,isHexagon=False):
      log.debug("Create grid of size [{},{},{}], hexagon = {} ".format(size[0],size[1],vector_size,isHexagon))
      self.size = size;
      self.isHexagon = isHexagon;
      self.vector_size = vector_size;
      self.grid = np.random.rand(size[0],size[1],vector_size);
#      self.grid = np.zeros([size,size,vector_size]) + 0.5;
   
   def _createGaussian(self,node,learn_val):
      sigma_sqrt = learn_val *  self.size[0] / 8.0;
      (x,y) = np.mgrid[0:self.size[0],0:self.size[1]];
      if self.isHexagon:
         y[0:self.size[1]:2] = y[0:self.size[1]:2] - 0.5;
         x = x * 0.8660254038;
      x_d, y_d = node[0]-x, node[1]-y;
      gauss =np.exp(0 - (x_d*x_d + y_d*y_d / sigma_sqrt));
      return gauss;
      
   
   def getGrid(self):
      return self.grid;
   
   def _feed(self,data,learn_val,sqDistFunc):
      """
      feed(data) -> void,
      Feeds the data to the grid.
      """

#     Calculate the squared distance to each node.    
      dist = sqDistFunc(data,self.grid);
      ndir = np.nan_to_num(data - self.grid);
      
      (x,y) = np.unravel_index(dist.argmin(),dist.shape);
      gauss = self._createGaussian((x,y),learn_val);
      
      corrector = (gauss * learn_val).reshape(self.size[0],self.size[1],1);
      
      self.grid = self.grid + (corrector * ndir)
      
   def train(self,data,iterations,sqDistFunc = AMath.euclideanSqDist):
      log.debug("Training with {} iterations".format(iterations));
      start = time.clock();
      data = np.array(data);
      for i in xrange(iterations):
         learn_val = float(iterations - i) / iterations;
         x = i % len(data);
         self._feed(data[x],learn_val,sqDistFunc);
      dur = time.clock() -start;
      
      log.debug("Done training took {} s, speed {} hz".format(dur,iterations/dur));
   
   @staticmethod
   def drawHexagonImage(values,min_size):
      scale = 128;
      (x_size,y_size) = values.shape;
      (xs,ys) = np.mgrid[0:x_size,0:y_size];
      
      # finding centers.
      (c_xs,c_ys) = (xs*scale*0.8660254038 + scale ,ys*scale + scale);
      
      c_ys[0:y_size:2] = np.float64(c_ys[0:y_size:2]) - ( 0.25* scale)
      c_ys[1:y_size:2] = np.float64(c_ys[1:y_size:2]) + ( 0.25* scale)
      
      (hexa_x, hexa_y) = ([0]*6,[0]*6);
      (hexa_o_x, hexa_o_y) = ([0]*6,[0]*6);
      for r in xrange(6):
         hexa_o_x[r] = (c_xs + np.cos(r*np.pi/3)*scale/2/0.8560254038);
         hexa_o_y[r] = (c_ys + np.sin(r*np.pi/3)*scale/2/0.8560254038);
         hexa_x[r] = (c_xs + np.cos(r*np.pi/3)*scale/2/0.9);
         hexa_y[r] = (c_ys + np.sin(r*np.pi/3)*scale/2/0.9);
      
      size = (int(x_size*scale* 0.8660254038+scale),y_size*scale+scale);
      img = Image.new("L",size,255);
      d = ImageDraw.Draw(img);
      for (x,y) in np.nditer([xs,ys]):
         a = []
         if(not np.isnan(values[x,y])):  
            for r in xrange(6):
               a.append((hexa_o_x[r][x,y],hexa_o_y[r][x,y]));
            d.polygon(a,fill=0);
            a = []
            for r in xrange(6):
               a.append((hexa_x[r][x,y],hexa_y[r][x,y]));
            d.polygon(a,fill=values[x,y]*255);
      
      a = min(float(size[0])/min_size,float(size[1])/min_size);
      size = (int(size[0]/a),int(size[1]/a));
      
      return img.resize(size,Image.ANTIALIAS)

   
   def getGridAsImages(self,size):
      if self.isHexagon:
         log.debug("Outputing images with hexagons");
         return [SOM.drawHexagonImage(self.grid[:,:,i],size) for i in range(self.vector_size)]
      else:
         log.debug("Outputing images with squares");
         imgs = [];
         for i in xrange(self.vector_size):
            img = Image.fromarray(np.uint8(self.grid[:,:,i]*255));
            imgs.append(img.resize(size,Image.NEAREST));
         return imgs;

   def getUMatrixImage(self,size):
      if self.isHexagon:
         log.debug("Outputing U-matrix with hexagons");
         
         (x,y,z) = self.grid[:,:,:].shape;
         s = (x*2-1,y*2)
         umatrix = np.zeros(s);
         
         i,j = np.mgrid[0:x,0:y];
         
         umatrix[:,:] = np.nan;
         
         xs,ys = np.mgrid[0:x,0:y];
         axs,ays = np.mgrid[0:x,0:y-1];
         bxs,bys = np.mgrid[0:x-1,0:y];
         cxs,cys = np.mgrid[0:x-1,1:y];
         
         amaskx, amasky = axs *2,ays*2+(axs % 2) +1;
         bmaskx, bmasky = bxs *2 +1 ,bys*2;
         cmaskx, cmasky = cxs *2 +1 ,cys*2 -1;

         uposx,uposy = xs *2,ys*2+(xs % 2);
         
         avec = self.grid[axs,ays,:] - self.grid[axs,ays+1,:];
         bvec = self.grid[bxs,bys,:] - self.grid[bxs+1,bys,:];
         cvec = self.grid[cxs,cys,:] - self.grid[cxs+1,cys-1,:];
         
         
         umatrix[amaskx,amasky] = np.sqrt(np.sum(avec * avec,2));
         umatrix[bmaskx,bmasky] = np.sqrt(np.sum(bvec * bvec,2));
         umatrix[cmaskx,cmasky] = np.sqrt(np.sum(cvec * cvec,2));
         
         
         temp = np.zeros((x,y));
         
         temp[axs,ays] += umatrix[amaskx,amasky];
         temp[bxs,bys] += umatrix[bmaskx,bmasky];
         temp[cxs,cys] += umatrix[cmaskx,cmasky];
         
         temp[axs,ays+1] += umatrix[amaskx,amasky];
         temp[bxs+1,bys] += umatrix[bmaskx,bmasky];
         temp[cxs+1,cys-1] += umatrix[cmaskx,cmasky];
         
         umatrix[uposx,uposy] = temp[xs,ys] / 6.0;
         
         umax = np.nanmax(umatrix);
         
         umatrix = 1 - umatrix/umax;
         
         return SOM.drawHexagonImage(umatrix,size);
      else:
         log.error("Outputing U-matrix with squares, is not allowed");
         sys.exit(-1);

   
class SOMTester(unittest.TestCase):
   
   def setUp(self):
      self.som = SOM();
      pass;
   
   def testInitalization(self):
      self.assertIsNone(self.som.getGrid());
      self.som.createGrid((2,2),2,False);
      self.assertIsNotNone(0,self.som.getGrid());
      
   def testSimpleFeed(self):
      self.som.createGrid((2,2),2,False);
      self.som._feed(np.array([0.5,0.5]),1,AMath.euclideanSqDist);
   
   def testTrain(self):
      self.som.createGrid((2,2),2,False);
      self.som.train([[0,0],[1,0],[0,1],[1,1]],10);
      
   def testSimpleImages(self):
      self.som.createGrid((10,10),2,True);
      self.som.train([[0,0],[1,0],[0,1],[1,1]],1000);
      imgs = self.som.getGridAsImages(512);
#      imgs[0].show();
#      imgs[1].show();
      
   def testDrawHexagonImage(self):
      self.som.createGrid((10,10),2,True);
      self.som.train([[0,0],[1,0],[0,1],[1,1]],1000);
      img = SOM.drawHexagonImage(self.som.grid[:,:,0],512);
#      img.show();
   
   
   def testGetUMatrix(self):
      self.som.createGrid((10,10),2,True);
      self.som.train([[0,0],[1,0],[0,1],[1,1]],1000);
      img = self.som.getUMatrixImage(512);
      img.show();
      
      imgs = self.som.getGridAsImages(512);
      imgs[0].show();
      imgs[1].show();
      
      
if __name__ == "__main__":
   logging.basicConfig(level=logging.DEBUG);
   unittest.main();