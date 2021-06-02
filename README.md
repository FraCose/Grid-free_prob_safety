---------------------------------------------------
This Repository contains the Algorithms explained in<br />
Cosentino, Abate, Oberhauser <br />
"Grid-Free Computation of Probabilistic Safety with Malliavin Calculus"<br />
---------------------------------------------------

The files are divided in the following way:<br />
- mc_gpu_cube.py and mc_gpu_sphere.py are the libraries with the code of the Algorithms presented in<br />
 the cited work.<br />
 *_cube.py is focalised on the experiments relative to the cube safety region.<br />
 *_sphere.py is focalised on the experiments relative to the sphere safety region.<br />
 Once completed they both save some npy files used then to plot the figures. <br />
- From the ipython notebooks you can get the same plots of the paper. <br />
 Before running the ipython notebooks you HAVE TO run the *.py file. <br />
 To have the interactive 3d plots run the notebooks in jupyter notebook (not lab). <br />

----------------------------------------------------
Special notes to run the experiments
----------------------------------------------------
- mc_gpu_*.py are set to plot interactive figures via a x11 server, which show the updates of the wanted border.<br />
 Please change the code (line 4 should be enough) accordingly if you run the code locally or something go wrong.<br />
 You can play with the parameters as explained in the work. It is important to remark that if the probability is 0, <br />
 then the gradient is 0 as well, and the algorithm does not converge, therefore you have to change the initial point. <br />
- if a CUDA gpu is not available, in principle the code can be changed easily replacing .cp with .np (importing numpy as np).<br />
 However, running the code on CPU could be quite heavy therefore we do not suggest it. <br />

---------------------------------------------------
Funding
---------------------------------------------------
The authors want to thank The Alan Turing Institute and the University of Oxford<br /> 
for the financial support given. FC is supported by The Alan Turing Institute, [TU/C/000021],<br />
under the EPSRC Grant No. EP/N510129/1. HO is supported by the EPSRC grant Datasig<br />
[EP/S026347/1], The Alan Turing Institute, and the Oxford-Man Institute.
