---
title: "Simulation of Different Variations of the Gaussian Process"
author: Arthur Lui
date: "21 March 2017"
geometry: margin=1in
fontsize: 12pt

# Uncomment if using natbib:

# bibliography: BIB.bib
# bibliographystyle: plain 

# This is how you use bibtex refs: @nameOfRef
# see: http://www.mdlerch.com/tutorial-for-pandoc-citations-markdown-to-latex.html

header-includes: 
    - \usepackage{bm}
    - \usepackage{bbm}
    - \usepackage{graphicx}
    - \pagestyle{empty}
    - \newcommand{\norm}[1]{\left\lVert#1\right\rVert}
    - \newcommand{\p}[1]{\left(#1\right)}
    - \newcommand{\bk}[1]{\left[#1\right]}
    - \newcommand{\bc}[1]{ \left\{#1\right\} }
    - \newcommand{\abs}[1]{ \left|#1\right| }
    - \newcommand{\mat}{ \begin{pmatrix} }
    - \newcommand{\tam}{ \end{pmatrix} }
    - \newcommand{\suml}{ \sum_{i=1}^n }
    - \newcommand{\prodl}{ \prod_{i=1}^n }
    - \newcommand{\ds}{ \displaystyle }
    - \newcommand{\df}[2]{ \frac{d#1}{d#2} }
    - \newcommand{\ddf}[2]{ \frac{d^2#1}{d{#2}^2} }
    - \newcommand{\pd}[2]{ \frac{\partial#1}{\partial#2} }
    - \newcommand{\pdd}[2]{\frac{\partial^2#1}{\partial{#2}^2} }
    - \newcommand{\N}{ \mathcal{N} }
    - \newcommand{\E}{ \text{E} }
    - \def\given{~\bigg|~}
    # Figures in correct place
    - \usepackage{float}
    - \def\beginmyfig{\begin{figure}[H]\center}
    - \def\endmyfig{\end{figure}}
    - \newcommand{\iid}{\overset{iid}{\sim}}
    - \newcommand{\ind}{\overset{ind}{\sim}}
    - \newcommand{\I}{\mathrm{\mathbf{I}}}
    #
    - \allowdisplaybreaks
    - \def\M{\mathcal{M}}
---

## Model

Data were generated from the following model:

$$
y_i = 0.3 + 9.4x_i + 0.5\sin(2.7x_i) + 1.1 / (1+x_i^2) + \epsilon_i
$$

where $\epsilon_i \sim \N(0,\sigma^2=0.1)$ and $x_i \sim \N(0,1)$.

The following graphs are the posterior summaries of the parameters 
in three different models: (1) Gaussian process, (2) kernel convolution, and
(3) predictive process.


## Gaussian Process
\beginmyfig
\includegraphics[height=1\textwidth]{../img/paramGP.pdf}
\caption{Original Gaussian Process}
\label{fig:paramGP}
\endmyfig

## Kernel Convolution
\beginmyfig
\includegraphics[height=1\textwidth]{../img/paramKC.pdf}
\caption{Kernel Convolution}
\label{fig:paramKC}
\endmyfig

## Predictive Process
\beginmyfig
\includegraphics[height=1\textwidth]{../img/paramPP.pdf}
\caption{Predictive Process}
\label{fig:paramPP}
\endmyfig


## Mean Functions

\beginmyfig
\includegraphics[height=1\textwidth]{../img/pred.pdf}
\caption{Posterior mean functions (red line) for GP (top), kernel convolution
(middle), and predictive process (bottom). The red shaded region being the 95\%
credible intervals. The small grey dots are the 1000 simulated  observations.
The dark grey dots is the \textbf{true} mean function. The vertical grey lines
in the bottom two graphs are the 33 knot points used. Note that the CI for the GP
is a little wider than that for the other two. That is the kernel convolution
and predicted process underestimates the variability in the mean function.}
\label{fig:pred}
\endmyfig
\newpage

## Timings

Note that the simulations were performed are a server so timings may vary according
to the server usages. Also, in the metropolis step, an auto-tuner was used
to determine the proposal step-size. Consequently, the burn-in was different 
for each simulation and the simulation times included the burn-in time.
However, the number of MCMC samples were 5000 for each simulation. 
Finally, before implementing MCMC, the mean functions were analytically integrated
out and estimated after the other parameter estimates were obtained. Those
timings are provided separately. All timings are reported in seconds. Code
for this assignment was written in Julia. And can be provided upon request.
(It's actually on the same Github repository as my notes.) 

| | GP | KC | PP |
|:--- |:---:|:---:|:---:|
Time for Learning Parameters| 2305 | 2422| 2698 |
Time for Estimating Mean Function| 5660 | 4497 | 4196 |

[//]: # ( example image embedding
\beginmyfig
\includegraphics[height=0.5\textwidth]{path/to/img/img.pdf}
\caption{some caption}
\label{fig:mylabel}
% reference by: \ref{fig:mylabel}
\endmyfig
)
[//]: # ( example image embedding
> ![some caption.\label{mylabel}](path/to/img/img.pdf){ height=70% }
)

[//]: # ( example two figs side-by-side
\begin{figure*}
  \begin{minipage}{.45\linewidth}
    \centering \includegraphics[height=1\textwidth]{img1.pdf}
    \caption{some caption}
    \label{fig:myLabel1}
  \end{minipage}\hfill
  \begin{minipage}{.45\linewidth}
    \centering \includegraphics[height=1\textwidth]{img2.pdf}
    \caption{some caption}
    \label{fig:myLabel2}
  \end{minipage}
\end{figure*}
)


[//]: # (Footnotes:)


