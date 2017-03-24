---
title: "Stochastic Process Final Report"
author: Arthur Lui
date: "23 March 2017"
geometry: margin=1in
fontsize: 12pt

# Uncomment if using natbib:

bibliography: ibp.bib
bibliographystyle: plain 

# This is how you use bibtex refs: @nameOfRef
# see: http://www.mdlerch.com/tutorial-for-pandoc-citations-markdown-to-latex.html

header-includes: 
    - \usepackage{bm}
    - \usepackage{bbm}
    - \usepackage{graphicx}
    #- \pagestyle{empty}
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
    #
    - \def\X{\bm X}
    - \def\A{\bm A}
    - \def\Z{\bm Z}
    - \def\F{\bm F}
    - \def\bL{\bm \Lambda}
---

# Introduction

One challenge in latent feature modelling is that the number of latent features
has to be predetermined by the researcher or subject experts. Bayesian
nonparametrics offers methods to learn the number of latent features while
discovering the latent features themselves.  The Indian buffet process (IBP)
provides a distribution over sparse binary matrices of infinite dimensions, and
can be used as a prior for feature matrices in latent feature models.
In this paper, I will first introduce latent feature models. Then, I will
review the IBP, discuss a toy example model and the details of implementing
MCMC to learn parameters of the model, and discuss some of the recent
applications of the IBP.


# Latent Feature Models

The IBP is commonly used in latent feature models. So providing some
introduction here of such models is appropriate. A latent feature model is akin
to factor analysis models where the practitioner observes some observations,
say a matrix $\X$ ($N\times D$), and is attempting to learn the underlying
structure of which generated $\X$. That is, in factor analysis, the model is
$\X = \bm{F\Lambda} + \bm\epsilon$ and the matrices $\F$ and $\bL$
are the parameters to be learned. Usually, the dimensions of $\F$ and $\bL$
are known and fixed. Moreover, $\F$ and $\bL$ are assumed to be continuous
matrices.

In a latent feature model, $\X = \Z\A + \bm \epsilon$, where only $\X$ is
observed, and the parameters to be learned are $\Z$ and $\A$. This differs
from the previous model in that $\Z$ is a binary matrix. In addition, the 
number of columns in $\Z$, which can be interpreted as the number of 
hidden features in the data generating mechanism, is unknown and treated
as a parameter. The IBP, which is a stochastic process, serves as a natural
prior distribution for $\Z$ in such models as it is a distribution over binary
matrices of variable sizes (i.e. variable columns, but fixed number of rows).
I will now introduce the IBP.

# Indian Buffet Process

@griffiths2011indian introduced the IBP by first constructing a matrix $\Z$ 
as follows

$$
\begin{split}
\pi_k &\sim \text{Beta}\p{\frac{\alpha}{K},1} \\
z_{ik} &\sim \text{Bernoulli}\p{\pi_k} \\
\end{split}
$$

where $k=1,\cdots,K$ and $i=1,\cdots,N$. As $K\rightarrow\infty$, 
$\Z\sim \text{IBP}(\alpha)$. It can be shown be integrating out $\pi_k$
for $k=1,\cdots,K$ that $\Z$ has probability mass function (pmf) 

\begin{equation}
\text{P}(\bm Z) = \frac{\alpha^{K_+}}{\prod_{h=1}^{2^N-1} 
                        {K_h}!} 
                        \exp\{-\alpha H_N\}\prod_{k=1}^{K_+}
                                     \frac{(N-m_k)!(m_k-1)!}{N!}
\end{equation}

where $H_N$ is the harmonic number, $\suml{i}{1}{N}i^{-1}$, $K_+$ is the number
of non-zero columns in $\bm Z$, $m_k$ is the $k^{th}$ column sum of $\bm Z$,
and $K_h$ the number of columns having history $h$. The history $h=1$ for a
matrix of $N$ rows is the $N$-dimensional binary vector, or the binary number,
(1,0,0,...,0).  The history $h=2$ is the $N$-dimensional binary vector
(0,1,0,...,0), etc. Finally, the history $h=2^{N-1}$ is the binary vector
(1,1,1,...,1).

# IBP as a Stochastic Process

Like the Dirichlet process (DP), the IBP has a culinary analogy which describes
the stochastic process generating observations from an IBP distribution with
parameter $\alpha$. The description is as follows:

> Customer $i$ taking dish $k$ is analogous to 
> observation $i$ possessing feature $k$. This is indicated by setting the value of
> $z_{ik}$ to 1 if the customer takes the dish, and 0 otherwise.
> 
> An IBP$(\alpha)$ for $N$ observations can be simulate as follows:
> 
> - The $1^{st}$ customer takes Poisson($\alpha$) number of dishes
> - For customers $i=2 \text{ to } N$,
>     1. For each previously sampled dish,
>        customer $i$ takes dish $k$ with probability $m_k / i$
>     2. After sampling the last sampled dish, customer $i$ samples 
>        Poisson($\alpha/i$) new dishes
> 

It can be shown that a matrix generated by this process has the same pmf up to
a proportionality constant as the previous pmf. This is difference is due to
the specific ordering of the columns induced by the data-generating process.
The binary matrices produced are referred to as left-ordered matrices. A
particular sample from the IBP$(\alpha=5)$ with 30 rows can be the matrix in
the left figure of Figure \ref{Z}

\beginmyfig
![Sample from IBP](../img/Z.pdf){ height=40% }
![Expected value of IBP(5)](../img/EZ.pdf){ height=40% }
\caption{Left: sample from IBP(5). Right: Expected value of IBP(5)}
\label{Z}
\endmyfig

Note here that the image represents a binary matrix of 30 rows where the
black squares are 1 and white squares are 0. Moreover, the expected value
of a random matrix with 30 rows and distributed IBP($\alpha$) is shown 
in the right figure in Figure \ref{Z}. Darker values are closer to 1 and lighter
colors are closer to 0.

# Stick-breaking Construction of IBP

Similarly to the DP, there is another commonly used construction of the IBP.
@teh2007stick introduced the stick-breaking construction for the IBP as follows:

$$
\begin{aligned}
  v_l &\sim \text{Beta}(\alpha,1) \\
  z_{ik} &\sim \text{Bernoulli}\p{\prod_{l=1}^k v_l} \\
\end{aligned}
$$

This representation was adopted in the dependent IBP by Williamson et al.
\cite{williamson2010dependent}, which assumes dependence between "customers".

# Toy Example

I now introduce


