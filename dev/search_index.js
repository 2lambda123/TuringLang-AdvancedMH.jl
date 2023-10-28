var documenterSearchIndex = {"docs":
[{"location":"api/#AdvancedMH.jl","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"","category":"section"},{"location":"api/","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"Documentation for AdvancedMH.jl","category":"page"},{"location":"api/#Structs","page":"AdvancedMH.jl","title":"Structs","text":"","category":"section"},{"location":"api/","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"MetropolisHastings","category":"page"},{"location":"api/#AdvancedMH.MetropolisHastings","page":"AdvancedMH.jl","title":"AdvancedMH.MetropolisHastings","text":"MetropolisHastings{D}\n\nMetropolisHastings has one field, proposal.  proposal is a Proposal, NamedTuple of Proposal, or Array{Proposal} in the shape of your data. For example, if you wanted the sampler to return a NamedTuple with shape\n\nx = (a = 1.0, b=3.8)\n\nThe proposal would be\n\nproposal = (a=StaticProposal(Normal(0,1)), b=StaticProposal(Normal(0,1)))\n````\n\nOther allowed proposals are\n\n\np1 = StaticProposal(Normal(0,1)) p2 = StaticProposal([Normal(0,1), InverseGamma(2,3)]) p3 = StaticProposal((a=Normal(0,1), b=InverseGamma(2,3))) p4 = StaticProposal((x=1.0) -> Normal(x, 1))\n\n\nThe sampler is constructed using\n\n\njulia spl = MetropolisHastings(proposal) ```\n\nWhen using MetropolisHastings with the function sample, the following keyword arguments are allowed:\n\ninit_params defines the initial parameterization for your model. If\n\nnone is given, the initial parameters will be drawn from the sampler's proposals.\n\nparam_names is a vector of strings to be assigned to parameters. This is only\n\nused if chain_type=Chains.\n\nchain_type is the type of chain you would like returned to you. Supported\n\ntypes are chain_type=Chains if MCMCChains is imported, or  chain_type=StructArray if StructArrays is imported.\n\n\n\n\n\n","category":"type"},{"location":"api/#Functions","page":"AdvancedMH.jl","title":"Functions","text":"","category":"section"},{"location":"api/","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"DensityModel","category":"page"},{"location":"api/#AdvancedMH.DensityModel","page":"AdvancedMH.jl","title":"AdvancedMH.DensityModel","text":"DensityModel{F} <: AbstractModel\n\nDensityModel wraps around a self-contained log-liklihood function logdensity.\n\nExample:\n\nl(x) = logpdf(Normal(), x)\nDensityModel(l)\n\n\n\n\n\n","category":"type"},{"location":"#AdvancedMH.jl","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"","category":"section"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"(Image: Stable) (Image: Dev) (Image: AdvancedMH-CI)","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"AdvancedMH.jl currently provides a robust implementation of random walk Metropolis-Hastings samplers.","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"Further development aims to provide a suite of adaptive Metropolis-Hastings implementations.","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"AdvancedMH works by allowing users to define composable Proposal structs in different formats.","category":"page"},{"location":"#Usage","page":"AdvancedMH.jl","title":"Usage","text":"","category":"section"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"First, construct a DensityModel, which is a wrapper around the log density function for your inference problem. The DensityModel is then used in a sample call.","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"# Import the package.\nusing AdvancedMH\nusing Distributions\nusing MCMCChains\n\nusing LinearAlgebra\n\n# Generate a set of data from the posterior we want to estimate.\ndata = rand(Normal(0, 1), 30)\n\n# Define the components of a basic model.\ninsupport(θ) = θ[2] >= 0\ndist(θ) = Normal(θ[1], θ[2])\ndensity(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf\n\n# Construct a DensityModel.\nmodel = DensityModel(density)\n\n# Set up our sampler with a joint multivariate Normal proposal.\nspl = RWMH(MvNormal(zeros(2), I))\n\n# Sample from the posterior.\nchain = sample(model, spl, 100000; param_names=[\"μ\", \"σ\"], chain_type=Chains)","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"Output:","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"Object of type Chains, with data of type 100000×3×1 Array{Float64,3}\n\nIterations        = 1:100000\nThinning interval = 1\nChains            = 1\nSamples per chain = 100000\ninternals         = lp\nparameters        = μ, σ\n\n2-element Array{ChainDataFrame,1}\n\nSummary Statistics\n\n│ Row │ parameters │ mean     │ std      │ naive_se    │ mcse       │ ess     │ r_hat   │\n│     │ Symbol     │ Float64  │ Float64  │ Float64     │ Float64    │ Any     │ Any     │\n├─────┼────────────┼──────────┼──────────┼─────────────┼────────────┼─────────┼─────────┤\n│ 1   │ μ          │ 0.156152 │ 0.19963  │ 0.000631285 │ 0.00323033 │ 3911.73 │ 1.00009 │\n│ 2   │ σ          │ 1.07493  │ 0.150111 │ 0.000474693 │ 0.00240317 │ 3707.73 │ 1.00027 │\n\nQuantiles\n\n│ Row │ parameters │ 2.5%     │ 25.0%     │ 50.0%    │ 75.0%    │ 97.5%    │\n│     │ Symbol     │ Float64  │ Float64   │ Float64  │ Float64  │ Float64  │\n├─────┼────────────┼──────────┼───────────┼──────────┼──────────┼──────────┤\n│ 1   │ μ          │ -0.23361 │ 0.0297006 │ 0.159139 │ 0.283493 │ 0.558694 │\n│ 2   │ σ          │ 0.828288 │ 0.972682  │ 1.05804  │ 1.16155  │ 1.41349  │\n","category":"page"},{"location":"#Usage-with-[LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl)","page":"AdvancedMH.jl","title":"Usage with LogDensityProblems.jl","text":"","category":"section"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"Alternatively, you can define your model with the LogDensityProblems.jl interface:","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"using LogDensityProblems\n\n# Use a struct instead of `typeof(density)` for sake of readability.\nstruct LogTargetDensity end\n\nLogDensityProblems.logdensity(p::LogTargetDensity, θ) = density(θ)  # standard multivariate normal\nLogDensityProblems.dimension(p::LogTargetDensity) = 2\nLogDensityProblems.capabilities(::LogTargetDensity) = LogDensityProblems.LogDensityOrder{0}()\n\nsample(LogTargetDensity(), spl, 100000; param_names=[\"μ\", \"σ\"], chain_type=Chains)","category":"page"},{"location":"#Proposals","page":"AdvancedMH.jl","title":"Proposals","text":"","category":"section"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"AdvancedMH offers various methods of defining your inference problem. Behind the scenes, a MetropolisHastings sampler simply holds some set of Proposal structs. AdvancedMH will return posterior samples in the \"shape\" of the proposal provided – currently supported methods are Array{Proposal}, Proposal, and NamedTuple{Proposal}. For example, proposals can be created as:","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"# Provide a univariate proposal.\nm1 = DensityModel(x -> logpdf(Normal(x,1), 1.0))\np1 = StaticProposal(Normal(0,1))\nc1 = sample(m1, MetropolisHastings(p1), 100; chain_type=Vector{NamedTuple})\n\n# Draw from a vector of distributions.\nm2 = DensityModel(x -> logpdf(Normal(x[1], x[2]), 1.0))\np2 = StaticProposal([Normal(0,1), InverseGamma(2,3)])\nc2 = sample(m2, MetropolisHastings(p2), 100; chain_type=Vector{NamedTuple})\n\n# Draw from a `NamedTuple` of distributions.\nm3 = DensityModel(x -> logpdf(Normal(x.a, x.b), 1.0))\np3 = (a=StaticProposal(Normal(0,1)), b=StaticProposal(InverseGamma(2,3)))\nc3 = sample(m3, MetropolisHastings(p3), 100; chain_type=Vector{NamedTuple})\n\n# Draw from a functional proposal.\nm4 = DensityModel(x -> logpdf(Normal(x,1), 1.0))\np4 = StaticProposal((x=1.0) -> Normal(x, 1))\nc4 = sample(m4, MetropolisHastings(p4), 100; chain_type=Vector{NamedTuple})","category":"page"},{"location":"#Static-vs.-Random-Walk","page":"AdvancedMH.jl","title":"Static vs. Random Walk","text":"","category":"section"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"Currently there are only two methods of inference available. Static MH simply draws from the prior, with no conditioning on the previous sample. Random walk will add the proposal to the previously observed value. If you are constructing a Proposal by hand, you can determine whether the proposal is a StaticProposal or a RandomWalkProposal using","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"static_prop = StaticProposal(Normal(0,1))\nrw_prop = RandomWalkProposal(Normal(0,1))","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"Different methods are easily composeable. One parameter can be static and another can be a random walk, each of which may be drawn from separate distributions.","category":"page"},{"location":"#Multiple-chains","page":"AdvancedMH.jl","title":"Multiple chains","text":"","category":"section"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"AdvancedMH.jl implements the interface of AbstractMCMC which means sampling of multiple chains is supported for free:","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"# Sample 4 chains from the posterior serially, without thread or process parallelism.\nchain = sample(model, RWMH(init_params), MCMCSerial(), 100000, 4; param_names=[\"μ\",\"σ\"], chain_type=Chains)\n\n# Sample 4 chains from the posterior using multiple threads.\nchain = sample(model, RWMH(init_params), MCMCThreads(), 100000, 4; param_names=[\"μ\",\"σ\"], chain_type=Chains)\n\n# Sample 4 chains from the posterior using multiple processes.\nchain = sample(model, RWMH(init_params), MCMCDistributed(), 100000, 4; param_names=[\"μ\",\"σ\"], chain_type=Chains)","category":"page"},{"location":"#Metropolis-adjusted-Langevin-algorithm-(MALA)","page":"AdvancedMH.jl","title":"Metropolis-adjusted Langevin algorithm (MALA)","text":"","category":"section"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"AdvancedMH.jl also offers an implementation of MALA if the ForwardDiff and DiffResults packages are available. ","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"A MALA sampler can be constructed by MALA(proposal) where proposal is a function that takes the gradient computed at the current sample. It is required to specify an initial sample init_params when calling sample.","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"# Import the package.\nusing AdvancedMH\nusing Distributions\nusing MCMCChains\nusing ForwardDiff\nusing StructArrays\n\nusing LinearAlgebra\n\n# Generate a set of data from the posterior we want to estimate.\ndata = rand(Normal(0, 1), 30)\n\n# Define the components of a basic model.\ninsupport(θ) = θ[2] >= 0\ndist(θ) = Normal(θ[1], θ[2])\ndensity(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf\n\n# Construct a DensityModel.\nmodel = DensityModel(density)\n\n# Set up the sampler with a multivariate Gaussian proposal.\nσ² = 0.01\nspl = MALA(x -> MvNormal((σ² / 2) .* x, σ² * I))\n\n# Sample from the posterior.\nchain = sample(model, spl, 100000; init_params=ones(2), chain_type=StructArray, param_names=[\"μ\", \"σ\"])","category":"page"},{"location":"#Usage-with-[LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl)-2","page":"AdvancedMH.jl","title":"Usage with LogDensityProblems.jl","text":"","category":"section"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"As above, we can define the model with the LogDensityProblems.jl interface. We can implement the gradient of the log density function manually, or use LogDensityProblemsAD.jl to provide us with the gradient computation used in MALA. Using our implementation of the LogDensityProblems.jl interface above:","category":"page"},{"location":"","page":"AdvancedMH.jl","title":"AdvancedMH.jl","text":"using LogDensityProblemsAD\nmodel_with_ad = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), LogTargetDensity())\nsample(model_with_ad, spl, 100000; init_params=ones(2), chain_type=StructArray, param_names=[\"μ\", \"σ\"])","category":"page"}]
}