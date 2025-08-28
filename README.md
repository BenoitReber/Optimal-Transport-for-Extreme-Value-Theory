# Optimal Transport meets multivariate Extreme Value Theory

## Scope of the project

Multivariate extreme value theory is interested in the regular variation of a measure $`m`$ defined as the convergence of the scaling $`t\,m(b(t)\cdot)_{|\mathbb{R}^{d}\setminus\{0\}}`$, for some appropriate function $`b`$, to some limit measure $`\mu`$ (in the set $`\mathrm{M}_{0}(\mathbb{R}^{d})`$ of Radon measure) on $`\mathcal{B}(\mathbb{R}^{d}\setminus\{0\})`$ finite on sets bounded away from the origin and with (possibly) infinite total mass as defined in [1,2]. This convergence happens in duality with the set $`\mathrm{C}_{b,0}^{+}(\mathbb{R}^{d})`$ of non-negative bounded continuous functions whose support is bounded away from the origin. In practice this allows one to build (simple) approximations of the measure $`m`$ for regions of the space bounded away from the origin for which few data is available.

Center-outward multivariate quantile functions and regions have been introduced for probability measures in [3], developing further ideas from [4] where the Monge-Kantorovich statistical depths are defined using optimal transport with the quadratic distance as cost between a well-chosen reference distribution $`P`$ and the target distribution $`Q`$.  The idea is to take for $`P`$ the spherical uniform distribution (defined by the polar decomposition made up of the uniform on the unit sphere for the angular component and the uniform on the $`[0,1]`$ segment for the radial part) for which Brenier's theorem [5,6] ensures the existence of a transport map $`T`$, called center-outward distribution function, such that $`Q`$ is the push-forward of $`P`$ by $`T`$. A "natural" quantile region of order $`q`$ for the spherical distribution is simply $`\mathbb{B}_{0,q}`$ the ball centered at the origin and of radius $q$. The center-outward quantile region of order $`q`$ for $`Q`$ is then defined by $`T(\mathbb{B}_{0,q})`$.

The aim of this ongoing research project is to bring the above-mentioned ideas to possibly infinite measures in $`\mathrm{M}_{0}(\mathbb{R}^{d})`$ arising as limits in the definition of regular variation in order to use these concepts in the approximations built with the limit measures.

The starting point is some extension of Brenier-McCann's theorem to possibly infinite measures in $`\mathrm{M}_{0}(\mathbb{R}^{d})`$. It is link to the optimal transport problem used in [7,8] to define a Wasserstein distance between Lévy measures in a similar manner as Brenier-McCann's theorem [5,6] zis linked to the Monge-Kantorovich problem with quadratic distance as cost.

## Content of the repository

The repository is structured as follows:

1. The working document contains the theoretical foundations of the project, though is not updated anymore as the project has been split in two since then. It includes an old version of some extension of Brenier-McCann's theorem to possibly infinite measures in $`\mathrm{M}_{0}(\mathbb{R}^{d})`$ under appropriate assumptions and convergence results for a new estimator of high order center-outward quantile regions.

2. In the notebooks extreme_quantile_region_estimation_i.ipynb, i=1,2,3, we illustrate the new method developped to estimate high order center-outward quantile regions of order close to one. Different examples are presented in the 2D case using data sampled using the MLExtreme package. These are first attempts to get some numerical results and further experiments are needed.

3. In the notebook generative_modelling_with_extreme_values.ipynb, an heuristic method is presented to generate data from a learned multivariate extreme value distribution. The idea is to use our theoretical results to build a generative model for the data.

## References

[1] Hult, H. and F. Lindskog (2006). Regular Variation for Measures on Metric Spaces. Publ. Inst. Math. 80 (94), 121–140.

[2] Lindskog, F., S. I. Resnick, and J. Roy (2014). Regularly varying measures on metric spaces: Hidden regular variation and hidden jumps. Probability Surveys 11, 270–314.

[3] Hallin, M., E. Del Barrio, J. Cuesta-Albertos, and C. Matr´an (2021). Distribution and quantile functions, ranks and signs in dimension d: a measure transportation approach. Annals of Statistics 49, 1139–1165.

[4] Chernozhukov, V., A. Galichon, M. HALLIN, and M. HENRY (2017).Monge–kantorovich depth, quantiles, ranks and signs. The Annals of Statistics 45, 223–256.

[5] Brenier, Y. (1991). Polar factorization and monotone rearrangement of vector-valued functions. Communications on pure and applied mathematics 44 (4), 375–417.

[6] McCann, R. J. (1995). Existence and uniqueness of monotone measure-preserving maps. Duke Mathematical Journal 80 (2), 309–323.

[7] Guillen, N., C. Mou, and S. Swiech (2019). Coupling l´evy measures and comparison principles for viscosity solutions. Transactions of the American Mathematical Society 372 (10), 7327–7370

[8] Catalano, M., H. Lavenant, A. Lijoi, and I. Pr¨unster (2021). A wasserstein index of dependence for random measures. arXiv preprint arXiv:2109.06646.
