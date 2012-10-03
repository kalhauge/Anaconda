from distutils.core import setup,Extension

setup(name='Anaconda',
      url='https://github.com/kalhauge/Anaconda',
      version='0.1',
      author='Christian Gram Kalhugae',
      author_email='christian@kalhauge.dk',
      packages = ["Anaconda"],
      py_modules=[
         'Anaconda.AssociationMining',
         'Anaconda.Table',
         'Anaconda.__init__',
         'Anaconda.SOM',
         'Anaconda.AMath'],
      )
      