module Entropyest
using DensityRatioEstimation, Statistics, HiQGA.transD_GP, Distributed, DelimitedFiles, Dates, Random

function getkldfromsamples(x1, x2; σ=[0.5], b=[20], nfolds=10, debug=false, nfit=6000)
    t = time()
    nx1, nx2 = length(x1), length(x2)
    nfit = min(nx1, nx2, nfit)
    debug && (@info "fitting $nfit samples")
    K_dre = fit(KLIEP, x1[randperm(nx1)[1:nfit]], x2[randperm(nx2)[1:nfit]], LCV((;σ,b), nfolds))
    # lr = ln(p(x1)/q(x2))
    rfunc = densratiofunc(x1, x2, K_dre) 
    # KLD(x1~p1||x2~q) = expectation of lr when samples are drawn from x1
    KLD = mean(log.(rfunc.(x1)))
    debug && (@info "process $(myid()) took $(time()-t) seconds")
    # [KLD, K_dre.σ, K_dre.b]
end    

function getkldfromopt(opt::transD_GP.Options, x2::AbstractVector, pids::UnitRange; 
            σ=[0.5], b=[20], nfolds=2, burninfrac=0.5, debug=false, restrictto=2, nuse=6000, nfit=6000)
    # open file
    debug && @info "OPENING "*opt.fdataname*"at pids $pids at $(Dates.now())"
    x1 = reduce(vcat, transD_GP.CommonToAll.assembleTat1(opt, :fstar; burninfrac, temperaturenum=1))
    x2 = reduce(vcat, x2)
    @assert size(x1, 2) == size(x2, 2) # same number of variables
    nx1, nx2 = size(x1,1), size(x2,1) # ensure there aren't too many samples
    debug && ((x1, x2) = map(x->x[:,1:restrictto], (x1, x2)))
    nuse = min(nx1, nx2, nuse)
    debug && (@info "using $nuse samples")
    x1, x2 = map(x->x[1:nuse,:], (x1, x2))
    # get kld from prior samples in x2
    A = reduce(hcat, pmap((x, y)->getkldfromsamples(x, y; σ, b, nfolds, nfit, debug), 
                                    WorkerPool(collect(pids)), eachcol(x1), eachcol(x2)))'
    debug && @info "WRITING "*opt.fdataname*" at $(Dates.now())"
    writedlm(opt.fdataname*"kld.txt", A)
    nothing
end    

function getkldfromfilenames(fnames::Vector{String}, opt_in::transD_GP.Options, x2::AbstractVector; 
                        burninfrac=0.5, σ=[0.5], b=[20], nfolds=10, 
                        debug=false, ncorespersounding=3, nuse=6000, nfit=6000)
    nsoundings = length(fnames)
    ncores = nworkers()
    nsequentialiters, nparallelsoundings = splittasks(;nsoundings, ncores, ncorespersounding)
    
    @info "done 0 out of $nsequentialiters at $(Dates.now())"
    for iter = 1:nsequentialiters
        ss = getss(iter, nsequentialiters, nparallelsoundings, nsoundings)
        @sync for (i, s) in enumerate(ss)
            pids = getpids(i, ncorespersounding)
            opt = deepcopy(opt_in)
            opt.fdataname = fnames[s]*"_"
            @async remotecall_wait(getkldfromopt, pids[1], opt, x2, pids[2:end]; 
                                    σ, b, nfolds, burninfrac, nuse, nfit, debug)
        end # @sync
        @info "done $iter out of $nsequentialiters at $(Dates.now())"
    end
    nothing    
end    

function getkldfromsoundings(soundings::Vector{S}, opt_in::transD_GP.Options, x2::AbstractVector; burninfrac=0.5, σ=[0.5], b=[20], nfolds=10) where S<:Sounding
    fnames = [s.sounding_string for s in soundings]
    getkldfromfilenames(fnames, opt_in, x2; burninfrac, σ, b, nfolds)
end

## parallel stuff

# global splitters across soundings
function splittasks(;nsoundings=nothing, ncores=nothing, ncorespersounding=nothing)
    # split into sequential iterations of parallel soundings
    @assert !any(isnothing.([nsoundings, ncores, ncorespersounding]))
    @assert ncores >= ncorespersounding
    nparallelsoundings = floor(Int, (ncores)/(ncorespersounding)) - 1 
    nsequentialiters = ceil(Int, nsoundings/nparallelsoundings)
    @info "will require $nsequentialiters iterations of $nparallelsoundings soundings in one iteration"
    nsequentialiters, nparallelsoundings
end    

function getss(iter, nsequentialiters, nparallelsoundings, nsoundings)
    if iter<nsequentialiters
        ss = (iter-1)*nparallelsoundings+1:iter*nparallelsoundings
    else
        ss = (iter-1)*nparallelsoundings+1:nsoundings
    end
    ss
end

function getpids(i, ncorespersounding)
    ncorespersounding*(i-1) + 2 : ncorespersounding*i + 1
end

end # module Entropyest
