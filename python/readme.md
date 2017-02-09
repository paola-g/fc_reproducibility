### Current version 
Custom pipeline for fMRI preprocessing.<br>
Files:
<ul>
<li> pipelineCustom.ipynb - notebook
<li> setupPipeline.py - python file for parameters customization and pipeline setup
<li> runPipeline.py - python file for launching the pipeline on a single subject
<li> pipelineCustom_par.py - python file for launching the pipeline on a cohort of subjects, supports SGE.
</ul>
Instruction for launching:
<ol>
<li> Customize parameters in class <i>config</i>
<ul>
<li> In the notebook in cell <b>Parameters</b>
<li> In setupPipeline.py at lines <b>35-50</b>
</ul>
<li> Customize functions <i>getDataDir</i>, <i>getParcelDir</i> and <i>build_path</i> to match local paths.
<ul>
<li> In the notebook in cell <b>Parameters</b>
<li> In setupPipeline.py at lines <b>53-72</b>
</ul>
<li> Customize pipeline parameters in struct <b>Operarions</b>
<ul>
<li> In the notebook in cell <b>Pipeline definition</b>
<li> In setupPipeline.py at lines <b>107-118</b>
</ul>
<li> Launch the pipeline
<ul>
<li> In the notebook cells can be executed sequentially (one by one or from the menu Cell->Run All)
<li> From command line with <b>python pipelineCustom_par.py</b> (<i>setupPipeline.py</i> and <i>runPipeline.py</i> need to be in the same path)
<li> To execute jobs in parallel on SGE, <i>queue</i> parameter in class <i>config</i> needs to be set to <b>True</b>. To customize <i>qsub</i> options, edit lines 395-396 in function <i>fnSubmitToCluster</i> in <i>setupPipeline.py</i> or in cell <b>Utils</b> in the notebook.
</ul>
</ol>
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


