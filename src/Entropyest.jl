module Entropyest
using DensityRatioEstimation, Statistics, HiQGA.transD_GP, Distributed, DelimitedFiles, 
Dates, Random, KernelDensity, KernelDensitySJ, PyPlot, Printf

function getkldfromsamples(x1, x2; σ=[0.5], b=[20], nfolds=10, debug=false)
    t = time()
    nx1, nx2 = length(x1), length(x2)
    K_dre = fit(KLIEP, x1, x2, LCV((;σ,b), nfolds))
    # lr = ln(p(x1)/q(x2))
    rfunc = densratiofunc(x1, x2, K_dre) 
    # KLD(x1~p1||x2~q) = expectation of lr when samples are drawn from x1
    KLD = mean(log.(rfunc.(x1)))
    debug && (@info "process $(myid()) took $(time()-t) seconds")
    [KLD, K_dre.σ, K_dre.b]
end    

function getkldfromopt(opt::transD_GP.Options, x2::AbstractVector, pids::UnitRange; 
            σ=[0.5], b=[20], nfolds=2, burninfrac=0.5, debug=false, restrictto=2, nuse=6000)
    # open file
    debug && @info "OPENING "*opt.fdataname*"at pids $pids at $(Dates.now())"
    x1 = reduce(vcat, transD_GP.CommonToAll.assembleTat1(opt, :fstar; burninfrac, temperaturenum=1))
    x2 = reduce(vcat, x2)
    @assert size(x1, 2) == size(x2, 2) # same number of variables
    debug && ((x1, x2) = map(x->x[:,1:restrictto], (x1, x2)))
    nx1, nx2 = size(x1, 1), size(x2, 1)
    nx1, nx2 = map(n -> ( n > nuse ? nuse : n ), (nx1, nx2))  # make sure nsamples isn't too huge
    debug && (@info "using $nx1, $nx2 samples")
    x1, x2 = map((x, n) -> x[randperm(size(x, 1))[1:n],:], (x1, x2), (nx1, nx2)) # shuffle to break correlations in McMC
    # get kld from prior samples in x2
    A = reduce(hcat, pmap((x, y)->getkldfromsamples(x, y; σ, b, nfolds, debug), 
                                    WorkerPool(collect(pids)), eachcol(x1), eachcol(x2)))'
    debug && @info "WRITING "*opt.fdataname*" at $(Dates.now())"
    writedlm(opt.fdataname*"kld.txt", A)
    nothing
end    

function getkldfromfilenames(fnames::Vector{String}, opt_in::transD_GP.Options, x2::AbstractVector; 
                        burninfrac=0.5, σ=[0.5], b=[20], nfolds=10, 
                        debug=false, ncorespersounding=3, nuse=6000)
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
                                    σ, b, nfolds, burninfrac, nuse, debug)
        end # @sync
        @info "done $iter out of $nsequentialiters at $(Dates.now())"
    end
    nothing    
end    

function getkldfromsoundings(soundings::Vector{S}, opt_in::transD_GP.Options, x2::AbstractVector; 
            burninfrac=0.5, σ=[0.5], b=[20], nfolds=10, debug=false, ncorespersounding=3, nuse=6000) where S<:Sounding
    fnames = [s.sounding_string for s in soundings]
    getkldfromfilenames(fnames, opt_in, x2; burninfrac, σ, b, nfolds, debug, ncorespersounding, nuse)
end

## parallel stuff

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

# scoring stuff

function getscore(xtrue, pdfx::transD_GP.CommonToAll.KDEstimator)
    log(pdfx(xtrue))
end

function findlayer(ztrue::Real, zboundaries::Vector)
    findfirst(zboundaries .>= ztrue)
end    

function getscore(ρtrue::Vector, ztrue::Vector, zboundaries::Vector, ρpost::AbstractArray, K::transD_GP.CommonToAll.KDEtype)
    @assert length(ρtrue) == length(ztrue)
    map((ρ, z)->begin
        ilayer = findlayer(z, zboundaries)
        isnothing(ilayer) && (ilayer=length(zboundaries))
        s = getscore(ρ, transD_GP.CommonToAll.kde_(K, ρpost[:,ilayer]))
        # (ilayer == 23 || ilayer == 24)&& (@info "check" z, ρ, transD_GP.CommonToAll.kde_(K, ρpost[:,ilayer])(ρ), s)
        # s
    end,
    ρtrue, ztrue)        
end

function getscorefromopt(opt::transD_GP.Options; 
        ρtrue=zeros(0), ztrue=zeros(0), zboundaries=zeros(0), K=transD_GP.CommonToAll.SJ(), burninfrac=0.5)
    @assert !isempty(ρtrue)
    @assert !isempty(ztrue)
    @assert !isempty(zboundaries)
    @info "OPENING "*opt.fdataname*" at $(Dates.now())"
    ρpost = reduce(vcat, transD_GP.CommonToAll.assembleTat1(opt, :fstar; burninfrac, temperaturenum=1))
    getscore(ρtrue, ztrue, zboundaries, ρpost, K)
end

function getscorefromfnames(fnames::Vector{String}, opt::transD_GP.Options; 
    ρtrue=zeros(0), ztrue=zeros(0), zboundaries=zeros(0), K=transD_GP.CommonToAll.SJ(), burninfrac=0.5, doplot=false)
    map(fn->begin
        opt.fdataname = fn*"_"
        s = getscorefromopt(opt; 
        ρtrue, ztrue, zboundaries, K, burninfrac)
        writedlm(opt.fdataname*"score.txt", s)
    end,
    fnames)
    nothing
end

nattobit(x) = log2(exp(1.))*x
bittonat(x) = x/log2(exp(1.))
logscoretoignorance(x) = -x  

function plotscoresandkld(ρ, zb, z, zall_kld, zboundaries_kld, fnames::Vector{String}; isboundary=false, figsize=(10,5), fontsize=11, zjitter=10, 
    optprior=nothing, nbins=50, cmappdf="inferno", x2=nothing, CIcolor = ["w", "k"], plotCI=true, lwidth=2)
    f, ax = plt.subplots(1, 3; sharey=true, figsize)
    till = findlast(zb[end] .>= zboundaries_kld)
    if !isnothing(optprior)
        @assert !isnothing(x2) # prior samples vector
        himage, edges, CI, meanimage, = transD_GP.CommonToAll.gethimage(transD_GP.LineRegression.Line([1.]), x2, optprior; temperaturenum=1,
                nbins, usekde=true)
        im1 = ax[1].pcolormesh(edges[:], [zboundaries_kld[1:till]; zboundaries_kld[till+1]], himage[1:till,:], cmap=cmappdf, vmin=0.)
        # plotCI && ax[1].plot(CI, zall_kld, linewidth=lwidth, color=CIcolor[1])
        # plotCI && ax[1].plot(CI, zall_kld, linewidth=lwidth, color=CIcolor[2], linestyle="--")
        # # ax[1].plot(meanimage[:], zall_kld, linewidth=lwidth, color="r", linestyle="--")
        # @info mean(meanimage[:])                
    end    
    ax[1].step(log10.(ρ), zb, color="w", linewidth=3)
    ax[1].step(log10.(ρ), zb, color="k", linewidth=1.5)
    cb1 = colorbar(im1, ax=ax[1])
    cb1.ax.set_title("prior\npdf")
    for (i,fn) in enumerate(fnames)
        kld = readdlm(fn*"_kld.txt")[:,1]
        s = readdlm(fn*"_score.txt")
        @assert length(kld) == length(zall_kld)
        isboundary ? ax[2].step(s, zb) : ax[2].plot(s, z)
        plotq(ax[2], vec(s), [0.05, 0.5, 0.95], zboundaries_kld[end]-zjitter*i)
        ax[3].plot(kld[1:till], zall_kld[1:till])
        plotq(ax[3], vec(kld), [0.05, 0.5, 0.95], zboundaries_kld[end]-zjitter*i)
    end    
    ax[1].invert_xaxis()
    ax[1].invert_yaxis()
    ax[1].set_title("True resistivity")
    ax[1].set_xlabel("Log₁₀ ρ")
    ax[1].set_ylabel("Depth m")
    ax[2].set_xlabel("log score")
    secax2 = ax[2].secondary_xaxis("top", functions=(logscoretoignorance, logscoretoignorance))
    secax2.set_xlabel("ignorance score"; fontsize)
    secax2.tick_params(labelsize=fontsize)
    ax[3].set_xlabel("information gain nats")
    secax3 = ax[3].secondary_xaxis("top", functions=(nattobit, bittonat))
    secax3.set_xlabel("information gain bits"; fontsize)
    secax3.tick_params(labelsize=fontsize)
    ax[3].legend(fnames)
    transD_GP.nicenup(f, fsize=fontsize)
end

function plotq(ax, x, whichqs, whichz)
    q = quantile(x, whichqs)
    Δless = q[2] - q[1]
    Δmore = q[3] - q[2]
    ax.errorbar(q[2], whichz, xerr=[[Δless], [Δmore]]; capsize=4, capthick=4, elinewidth=1, linewidth=2, marker="v", 
        color = ax.lines[end].get_color())
end

# plot kld stuff
function kldcompare(parentdir, subdirs; yl=nothing, topowidth=2, fontsize=11, idxshow=[], legendstring=[],
        figsize=(15, 4), zall=[1.], dr=[1.], dz=[1.], vmin=1e-2, cmap="gist_ncar", usemax=nothing, figsize_hist=(5,7),
        preferEright=true, preferNright=false, histogram_depth_ranges=([0., 300],), histogram_kld_ranges=([0:0.1:2],))
    # has to be run from the parent directory containing all survey subdirs
    @assert length(histogram_depth_ranges) == length(histogram_kld_ranges)
    nsurveys, nhists = length(subdirs), length(histogram_depth_ranges)
    !isempty(idxshow) && @assert length(idxshow) == nsurveys
    isempty(legendstring) &&(legendstring = subdirs)
    heightratios = ones(nsurveys+1); heightratios[end]=0.1
    fig, ax = plt.subplots(nsurveys+1, 1, gridspec_kw=Dict("height_ratios" => heightratios), figsize=figsize)
    fig2, ax2 = plt.subplots(nhists, 1, sharex="all", figsize=figsize_hist, squeeze=false)
    imh = Vector{Any}(undef, nsurveys)
    foundmax = -Inf
    for (i, dir) in enumerate(subdirs)
        include(joinpath(parentdir, dir, "01_read_data.jl"))
        A = [readdlm(joinpath(parentdir, dir, s.sounding_string*"_kld.txt"))[:,1] for s in soundings]
        img = reduce(hcat, A)
        linename = "_line_$(soundings[1].linenum)_summary.txt"
        chi2fname = "phid_mean"*linename
        chi2mean = readdlm(joinpath(parentdir, dir, chi2fname))[:,1]
        idxgoodchi2 = chi2mean .<= 1.21
        img[img.<0] .= 0.
        maximg = maximum(img[:])
        (maximg > foundmax) && (foundmax = maximg)
        A, gridx, gridz, topofine, R = transD_GP.CommonToAll.makegrid(img, soundings; zall, dz, dr)
        Z = [s.Z for s in soundings]
        imh[i] = ax[i].imshow(A; extent=[gridx[1], gridx[end], gridz[end], gridz[1]], aspect="auto", cmap, vmin)
        ax[i].plot(gridx, topofine, linewidth=topowidth, "-k")
        ax[i].set_ylabel("Height m")
        !isnothing(yl) && ax[i].set_ylim(yl)
        !isempty(idxshow) && transD_GP.CommonToAll.plotprofile(ax[i], [idxshow[i]], Z, R)
        transD_GP.CommonToAll.plotNEWSlabels(soundings, gridx, gridz, [ax[i]]; preferEright, preferNright)
        i != nsurveys && ax[i].set_xticklabels([])
        # now for the histograms with depth
        for j = 1:nhists
            idx = histogram_depth_ranges[j][1] .<= zall .<histogram_depth_ranges[j][2]
            # ax2[j,1].hist(img[idx,:][:], histogram_kld_ranges[j]..., density=true, histtype="step", linewidth=2)
            qs = reduce(hcat, [quantile(in_img[idxgoodchi2], [0.05, .5, 0.95]) for in_img in eachrow(img)])'
            Δless = qs[idx,2] - qs[idx,1]
            Δmore = qs[idx,3] - qs[idx,2]
            ax2[j].errorbar(qs[idx,2],zall[idx], xerr=[Δless, Δmore], errorevery=(0+i, 5), capsize=4, capthick=4, elinewidth=1, linewidth=2, label=legendstring[i])
            !isempty(idxshow) && ax2[j].plot(img[idx,idxshow[i]], zall[idx], color = ax2[j].lines[end].get_color(), "--")
            secax2 = ax2[j].secondary_xaxis("top", functions=(nattobit, bittonat))
            secax2.set_xlabel("KLD bits"; fontsize)
            secax2.tick_params(labelsize=fontsize)
            ax2[j].set_xlabel("KLD nats")
            ax2[j].set_ylabel("Depth m")
        end
        ax2[end].legend(framealpha=0.1)
        ax[i].set_title(legendstring[i])
    end
    ax[end-1].set_xlabel("Line distance m")
    map(k->ax2[k, end].invert_yaxis(), 1:nhists)
    for i = 1:nsurveys
        usemax = isnothing(usemax) ? foundmax : usemax
        imh[i].set_clim(vmin, usemax) 
    end
    cb = colorbar(imh[nsurveys], cax=ax[end], orientation="horizontal")
    cb.set_label("KLD nats", labelpad=0)
    map(x->transD_GP.nicenup(x, fsize=fontsize, h_pad=0), (fig, fig2))
end    

end # module Entropyest
