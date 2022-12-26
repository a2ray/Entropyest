module Entropyest
using DensityRatioEstimation, Statistics, HiQGA.transD_GP, Distributed, DelimitedFiles, Dates, Random

function getkldfromsamples(x1, x2; σ=[0.5], b=[20], nfolds=10, debug=false, Kfolds=5)
    t = time()
    nx1, nx2 = length(x1), length(x2)
    debug && (@info "fitting $nx1, $nx2 samples")
    nperfold_x1, nperfold_x2 = map(nx->getnperfold(nx, Kfolds), (nx1, nx2))
    debug && (@info "perfold are $nperfold_x1, $nperfold_x2 samples")
    ridx_x1, ridx_x2 = map(nx->getrandomizedidx(nx), (nx1, nx2))
    K_dre = Vector{Any}(undef, Kfolds)
    KLD = zeros(Kfolds)
    for k = 1:Kfolds
        idx_x1 = ridx_x1[getfoldidx(k, nperfold_x1, Kfolds, nx1)]
        idx_x2 = ridx_x2[getfoldidx(k, nperfold_x2, Kfolds, nx2)]
        K_dre[k] = fit(KLIEP, x1[idx_x1], x2[idx_x2], LCV((;σ,b), nfolds))
        rfunc = densratiofunc(x1[idx_x1], x2[idx_x2], K_dre[k])
        KLD[k] = mean(log.(rfunc.(x1[idx_x1])))
    end    
    debug && (@info "process $(myid()) took $(time()-t) seconds")
    [mean(KLD), mean(getproperty.(K_dre, :σ)), mean(getproperty.(K_dre, :b))]
end    

getrandomizedidx(totalnum) = randperm(totalnum)
getnperfold(totalnum, K) = floor(Int, totalnum/K)
getfoldidx(k, nperfold, K, totalnum) = k < K ? ((k-1)nperfold + 1 : k*nperfold) : ((K-1)nperfold + 1 : totalnum) 

function getkldfromopt(opt::transD_GP.Options, x2::AbstractVector, pids::UnitRange; 
            σ=[0.5], b=[20], nfolds=2, burninfrac=0.5, debug=false, restrictto=2, nuse=6000, Kfolds=5)
    # open file
    debug && @info "OPENING "*opt.fdataname*"at pids $pids with burin: $burninfrac at $(Dates.now())"
    x1 = reduce(vcat, transD_GP.CommonToAll.assembleTat1(opt, :fstar; burninfrac, temperaturenum=1))
    x2 = reduce(vcat, x2)
    @assert size(x1, 2) == size(x2, 2) # same number of variables
    nx1, nx2 = size(x1,1), size(x2,1) 
    debug && ((x1, x2) = map(x->x[:,1:restrictto], (x1, x2)))
    nx1, nx2 = map(n -> ( n > nuse ? nuse : n ), (nx1, nx2))  # make sure nsamples isn't too huge
    debug && (@info "using $nx1, $nx2 samples")
    x1, x2 = map((x, n) -> x[1:n,:], (x1, x2), (nx1, nx2))
    # get kld from prior samples in x2
    A = reduce(hcat, pmap((x, y)->getkldfromsamples(x, y; σ, b, nfolds, Kfolds, debug), 
                                    WorkerPool(collect(pids)), eachcol(x1), eachcol(x2)))'
    debug && @info "WRITING "*opt.fdataname*" at $(Dates.now())"
    writedlm(opt.fdataname*"kld.txt", A)
    nothing
end    

function getkldfromfilenames(fnames::Vector{String}, opt_in::transD_GP.Options, x2::AbstractVector; 
                        burninfrac=0.5, σ=[0.5], b=[20], nfolds=10, 
                        debug=false, ncorespersounding=3, nuse=6000, Kfolds=5)
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
                                    σ, b, nfolds, burninfrac, nuse, Kfolds, debug)
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
