module Entropyest
using DensityRatioEstimation, Statistics, HiQGA.transD_GP, Distributed, DelimitedFiles, Dates

function getkldfromsamples(x1, x2; σ=[0.5], b=[20], nfolds=10)
    K_dre = fit(KLIEP, x1, x2, LCV((;σ,b), nfolds))
    # lr = ln(p(x1)/q(x2))
    rfunc = densratiofunc(x1, x2, K_dre) 
    # KLD(x1~p1||x2~q) = expectation of lr when samples are drawn from x1
    KLD = mean(log.(rfunc.(x1)))
    [KLD, K_dre.σ, K_dre.b]
end    

function getkldfromopt(opt::transD_GP.Options, x2::AbstractVector, pids::Array{Int, 1}; σ=[0.5], b=[20], nfolds=10, burninfrac=0.5)
    # open file
    x1 = reduce(vcat, transD_GP.CommonToAll.assembleTat1(opt, :fstar; burninfrac, temperaturenum=1))
    x2 = reduce(vcat, x2)
    @assert size(x1, 2) == size(x2, 2)
    # get kld from prior samples in x2
    A = reduce(hcat, pmap((x, y)->getkldfromsamples(x, y; σ, b, nfolds), WorkerPool(pids), eachcol(x1), eachcol(x2)))'
    writedlm(opt.fdataname*"_kld.txt", A)
    nothing
end    

function getkldfromfilenames(fnames::Vector{String}, opt_in::transD_GP.Options, x2::AbstractVector; burninfrac=0.5, σ=[0.5], b=[20], nfolds=10)
    nsoundings = length(fnames)
    nchainspersounding = length(opt_in.xall)
    ncores = nworkers()
    nsequentialiters, nparallelsoundings = splittasks(;nsoundings, ncores, nchainspersounding, ppn=nchainspersounding+1)
    
    opt = deepcopy(opt_in)
    @info "done 0 out of $nsequentialiters at $(Dates.now())"
    for iter = 1:nsequentialiters
        ss = getss(iter, nsequentialiters, nparallelsoundings, nsoundings)
        @sync for (i, s) in enumerate(ss)
            pids = getpids(i, nchainspersounding)
            opt.fdataname = fnames[s]*"_"
            @async remotecall_wait(getkldfromopt, pids[1], opt, x2, pids[2:end]; σ, b, nfolds, burninfrac)
        end # @sync
        @info "done $iter out of $nsequentialiters at $(Dates.now())"
    end
    nothing    
end    

function getkldfromsoundings(soundings::Vector{S}, opt_in::transD_GP.Options, x2::AbstractVector; burninfrac=0.5, σ=[0.5], b=[20], nfolds=10) where S<:Sounding
    fnames = [s.sounding_string for s in soundings]
    getkldfromfilenames(fnames, opt_in, x2; burninfrac, σ, b, nfolds)
end

end # module Entropyest
