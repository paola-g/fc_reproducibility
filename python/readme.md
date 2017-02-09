### Current version 
Custom pipeline for fMRI preprocessing.<br>
Files:
<ul>
<li> pipelineCustom.ipynb - notebook
<li> setupPipeline.py - python file for parameters customization and pipeline setup
<li> runPipeline.py - python file for launching the pipeline on a single subject
<li> pipelineCustom_par.py - python file for launching the pipeline on a cohort of subjects, supports SGE.
</ul>

### Older versions
#### First version
Porting of Julien's MATLAB code, mainly based on FSL.<br>
Files:
<ul> 
<li> pipeline.ipynb - notebook
<li> pipeline.py - python version of notebook
<li> init.py, calls.py - files for testing (sequential execution)
<li> init_par.py, calls_par.py, Finn_loadandpreprocess.py - files for testing (on SGE)
</ul>
#### Second version
Implementation of Finn's pipeline (with Legendre polynomials and separate regressors for WM and CSF)<br>
Files:
<ul>
<li> pipelineFinn.ipynb - notebook
</ul>
#### Third version
Porting of Julien's Finn_preprocess2.m  (reduced dependencies on FSL)<br>
Files:
<ul>
<li> pipelineFinn2.ipynb - notebook
<li> pipelineFinn2.py - python version of notebook
</ul>
#### Nilearn version
Implementation based on Nilearn, incomplete.<br>
Files:
<ul>
<li>pipelineNilearn.py
</ul>


