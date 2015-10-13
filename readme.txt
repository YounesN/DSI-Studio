Along the following library you need to download IT++ library and compile it.

Instructions for compiling IT++ for windows is available in here:

http://nejahi.com/1/post/2015/09/compiling-it-with-visual-studio-express-2013-and-cmake-322.html


Qt Creator is needed to edit the GUI.

The following libraries are also needed to build DSI Studio

1. download libraries (required for building DSI Studio)

download https://github.com/frankyeh/TIPL/zipball/master to directory src/image

2. download Boost library (required for building DSI Studio)

www.boost.org


The .pro files has to be edited to match the file system. Other required data to run the program is placed under /data


3. download FA template (required for runtime)


4. download armadillo library.

http://arma.sourceforge.net/

5. Levmar library needed to implement Levenberg-Marquardt optimization algorithm for BSM.

http://users.ics.forth.gr/~lourakis/levmar/
