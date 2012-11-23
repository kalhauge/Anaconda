"""
Name: AGraphics
Author: Christian Gram Kalhauge : kalhauge@cbs.dtu.dk 
Date : 13th August - 2012

Used to produce grapichs. 

"""

import AMath
import Image, ImageDraw, ImageFont
import numpy as np

def drawHexagonImage(values,min_size,labels=None,fontpath=None,normfunction=AMath.normalizeData):
   """
   Draws a Image using the data as hexagons. If the data
   is np.nan. It is posible to assing labels. Labels must corespond in
   size with the values.
   """
   values = normfunction(values);
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


   if labels:
      font = ImageFont.truetype(fontpath, 32);
   size = (int(x_size*scale* 0.8660254038+scale),y_size*scale+scale);
   img = Image.new("L",size,255);
   d = ImageDraw.Draw(img);
   for (x,y) in np.nditer([xs,ys]):
      a = []
      color =values[x,y]*255;
      if(not np.isnan(values[x,y])):  
         for r in xrange(6):
            a.append((hexa_o_x[r][x,y],hexa_o_y[r][x,y]));
         d.polygon(a,fill=0);
         a = []
         for r in xrange(6):
            a.append((hexa_x[r][x,y],hexa_y[r][x,y]));
         d.polygon(a,fill=color);

      if labels:
         s = labels[x][y];
         w,h = font.getsize(s);
         v = (c_xs[x,y] - w/2, \
           c_ys[x,y] - h/2);
         d.text(v,s,font=font,fill=255 if color < 128 else 0);

   a = min(float(size[0])/min_size,float(size[1])/min_size);
   size = (int(size[0]/a),int(size[1]/a));

   return img.resize(size,Image.ANTIALIAS)

def drawMatrixImage(matrix,min_size,normfunction=AMath.normalizeData):
   """
   Draws a image of a matrix, with its values normalized between [0,1]. If one need an other
   function to do the illustration, one can set the normfunction with it.
   """
   return Image \
          .fromarray(np.uint8(normfunction(self.grid[:,:,i])*255)) \
          .resize(min_size,NEAREST)