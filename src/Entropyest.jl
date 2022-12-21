module Entropyest
using DensityRatioEstimation, Statistics, HiQGA.transD_GP, Distributed, DelimitedFiles, Dates
using HiQGA.transD_GP.SoundingDistributor

function getkldfromsamples(x1, x2; σ=[0.5], b=[20], nfolds=10, debug=false)
    t = time()
    K_dre = fit(KLIEP, x1, x2, LCV((;σ,b), nfolds))
    # lr = ln(p(x1)/q(x2))
    rfunc = densratiofunc(x1, x2, K_dre) 
    # KLD(x1~p1||x2~q) = expectation of lr when samples are drawn from x1
    KLD = mean(log.(rfunc.(x1)))
    debug && (@info "$process (myid()) took $(time()-t) seconds")
    [KLD, K_dre.σ, K_dre.b]
end    

function getkldfromopt(opt::transD_GP.Options, x2::AbstractVector, pids::UnitRange; 
            σ=[0.5], b=[20], nfolds=10, burninfrac=0.5, debug=false, restrictto=2)
    # open file
    @info "OPENING "*opt.fdataname*"at pids $pids at $(Dates.now())"
    x1 = reduce(vcat, transD_GP.CommonToAll.assembleTat1(opt, :fstar; burninfrac, temperaturenum=1))
    x2 = reduce(vcat, x2)
    @assert size(x1, 2) == size(x2, 2)
    debug && (x1, x2 = map(x->x[1:restrictto,:]))
    # get kld from prior samples in x2
    A = reduce(hcat, pmap((x, y)->getkldfromsamples(x, y; σ, b, nfolds, debug), 
                                    WorkerPool(collect(pids)), eachcol(x1), eachcol(x2)))'
    @info "WRITING "*opt.fdataname*" at $(Dates.now())"
    writedlm(opt.fdataname*"kld.txt", A)
    nothing
end    

function getkldfromfilenames(fnames::Vector{String}, opt_in::transD_GP.Options, x2::AbstractVector; 
                        burninfrac=0.5, σ=[0.5], b=[20], nfolds=10, debug=false)
    nsoundings = length(fnames)
    ncorespersounding = length(opt_in.xall)
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
                                    σ, b, nfolds, burninfrac, debug)
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
function splittasks(soundings::AbstractVector; ncorespersounding=nothing)
    nsoundings = length(soundings)
    ncores = nworkers()
    splittasks(;nsoundings, ncores, ncorespersounding)
end

function splittasks(;nsoundings=nothing, ncores=nothing, ncorespersounding=nothing)
    # split into sequential iterations of parallel soundings
    @assert !any(isnothing.([nsoundings, ncores, ncorespersounding]))
    @assert ncores >= ncorespersounding
    nparallelsoundings = floor(Int, (ncores)/(ncorespersounding))
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
