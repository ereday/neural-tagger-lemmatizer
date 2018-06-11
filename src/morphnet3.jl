### output only pos not feats 
using Knet,ArgParse,Logging,JLD
include("data.jl")

import JLD: writeas, readas
import Knet: RNN
type RNNJLD; inputSize; hiddenSize; numLayers; dropout; inputMode; direction; mode; algo; dataType; end
writeas(r::RNN) = RNNJLD(r.inputSize, r.hiddenSize, r.numLayers, r.dropout, r.inputMode, r.direction, r.mode, r.algo, r.dataType)
readas(r::RNNJLD) = rnninit(r.inputSize, r.hiddenSize, numLayers=r.numLayers, dropout=r.dropout, skipInput=(r.inputMode==1), bidirectional=(r.direction==1), rnnType=(:relu,:tanh,:lstm,:gru)[1+r.mode], algo=r.algo, dataType=r.dataType)[1]
type KnetJLD; a::Array; end
writeas(c::KnetArray) = KnetJLD(Array(c))
readas(d::KnetJLD) = (gpu() >= 0 ? KnetArray(d.a) : d.a)

function parseargs()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--datafiles";arg_type=String ;nargs='+';help="parsed data if available")
        ("--bestmodel";default="bestmodel_debug.jld";help="Save best model to file")
        ("--logfile";arg_type=String; default="debug.out";help="log file")
        ("--hs";nargs='+';arg_type=Int;default=[512,512,256];help="hidden sizes")
        ("--es";nargs='+';arg_type=Int;default=[64,256];help="embedding size")
        ("--bs";arg_type=Int; default=1;help="batch size")
        ("--atype";default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--seed";arg_type=Int; default=31;help="Random number seed.")
        ("--pdrop";arg_type = Float64;default= 0.4;help="Dropout")
        ("--decayrate";arg_type=Float64;default=1.0)
        ("--optim";arg_type=String;default="Sgd(;lr=1.6,gclip=60.0)";help="Sgd|Adam|RMSProp")
        ("--patiance";arg_type=Int;default=10;help="number of validation to wait")
        ("--mode";arg_type=Int;default=1;help="mode=1 training 2 else")
        ("--epoch";arg_type=Int; default=1; help="Number of epochs for training.")
        ("--wordlimit";arg_type=Int; default=100; help="Max number of words in sentence.")
        ("--threshold";arg_type=Float64; default=0.0; help="Threshold to be considered")
        ("--transfer_from";arg_type=String;default="source_model.jld"; help="Initial model for TL.")
        ("--transferdatafiles";arg_type=String ;nargs='+';help="parsed data if available")
        ("--random_ix";arg_type=Int;nargs='+';default=[1,2,9,10,11];help="weight ix to init randomly")
    end
    return parse_args(s;as_symbols=true)
end

function main()
    opts = parseargs()
    setseed(opts[:seed])
    opts[:atype] = eval(parse(opts[:atype]))
    opts[:model] = "morphnet3.jl"
    Logging.configure(level=DEBUG)
    Logging.configure(output=open(opts[:logfile], "a"))
    enc_data,c2i,o2i = prepare_data(opts)
    i2o = Dict(); for (u,v) in o2i;i2o[v]=u;end
    i2oarr = Array{Any}(length(i2o))
    for (k,v) in o2i; i2oarr[v] = k;end
    if opts[:mode] == 2
        # generation mode
        model = JLD.load(opts[:bestmodel])
        weights = model["weights"]
        rsettings = model["rsettings"]
        params    = model["params"]
        opts[:lr] = params[1].lr
        opts[:gc] = params[1].gclip
        i2o = Dict(); for (u,v) in o2i;i2o[v]=u;end
        i2c = Dict(); for (u,v) in c2i;i2c[v]=u;end
        i2oarr = Array{Any}(length(i2o))
        for (k,v) in o2i; i2oarr[v] = k;end
        for (u,v) in opts
            info(u,"=>",v)
        end
        info("input vocabulary length:",length(c2i))
        info("output vocabulary length:",length(o2i))
        besties = zeros(2)
        data  = minibatch(enc_data,opts,o2i;st=false)
        words1 = map(x->map(y->y[1],x),enc_data[3])
        words  = map(x->map(y->join(map(z->i2c[z],y),""),x),words1)
        generate(weights,rsettings,data[3],words,i2o,i2oarr,i2c,opts)
        return 1
    elseif opts[:mode] == 3
        # transfer learning
        data = minibatch(enc_data,opts,o2i)
        rsettings,weights = initweights(opts,c2i,o2i,xavier)
        params = map(wi->eval(parse(opts[:optim])),weights)
        opts[:lr] = params[1].lr
        opts[:gc] = params[1].gclip
        for (u,v) in opts
            info(u,"=>",v)
        end
        info("input vocabulary length:",length(c2i))
        info("output vocabulary length:",length(o2i))
        besties = zeros(2)
        besties[2] = opts[:threshold]
        tfname = opts[:transfer_from]
        model = JLD.load(tfname)
        source_weights = model["weights"]
        source_bestacc  = model["bestacc"]
        source_bestacc2 = model["bestacc2"]
        info("source lang:",split(split(tfname,"_")[1],"/")[end]," dev acc:",source_bestacc)
        # To get source_c2i and source_o2i
        source_opts = deepcopy(opts)
        source_opts[:datafiles] = opts[:transferdatafiles]
        _,source_c2i,source_o2i = prepare_data(source_opts)

        # Do not transfer input embedding and output
        #=
        for i=1:length(weights)
            weights[i] = (i in opts[:random_ix])? weights[i] : source_weights[i]
        end
        =#
        # Transfer everyting common
        for i=1:length(weights)
            if i == 1
                for sk in keys(source_c2i)
                    if haskey(c2i,sk)
                        weights[i][:,c2i[sk]] = source_weights[i][:,source_c2i[sk]]
                    end
                end
            elseif i == 2
                for sk in keys(source_o2i)
                    if haskey(o2i,sk)
                        weights[i][:,o2i[sk]] = source_weights[i][:,source_o2i[sk]]
                    end
                end
            elseif i == 9
                continue
            elseif i in [10,11]
                for sk in keys(source_o2i)
                    if haskey(o2i,sk)
                        weights[i][o2i[sk],:] = source_weights[i][source_o2i[sk],:]
                    end
                end
            else
                weights[i] = source_weights[i]
            end
        end
    elseif isfile(opts[:bestmodel])
        model = JLD.load(opts[:bestmodel])
        weights = model["weights"]
        rsettings = model["rsettings"]
        params    = model["params"]
        opts[:lr] = params[1].lr
        opts[:gc] = params[1].gclip
        i2o = Dict(); for (u,v) in o2i;i2o[v]=u;end
        i2c = Dict(); for (u,v) in c2i;i2c[v]=u;end
        i2oarr = Array{Any}(length(i2o))
        for (k,v) in o2i; i2oarr[v] = k;end
        for (u,v) in opts
            info(u,"=>",v)
        end
        info("input vocabulary length:",length(c2i))
        info("output vocabulary length:",length(o2i))
        data = minibatch(enc_data,opts,o2i)
        besties = zeros(2)
        besties[1] = model["bestacc"]
        besties[2] = model["bestacc2"]
    else
        data = minibatch(enc_data,opts,o2i)
        rsettings,weights = initweights(opts,c2i,o2i,xavier)
        params = map(wi->eval(parse(opts[:optim])),weights)
        opts[:lr] = params[1].lr
        opts[:gc] = params[1].gclip
        for (u,v) in opts
            info(u,"=>",v)
        end
        info("input vocabulary length:",length(c2i))
        info("output vocabulary length:",length(o2i))
        besties = zeros(2)
        besties[2] = opts[:threshold]
    end
    dlss,dacc = evaluate(weights,rsettings,data[2],i2o,i2oarr,opts)
    info(@sprintf "[dev-0] acc:%.4f lss:%.4f"  (100*dacc[1])/dacc[2] dlss[1]/dlss[2])
    tlss,tacc = evaluate(weights,rsettings,data[3],i2o,i2oarr,opts)
    info(@sprintf "[tst-0] acc:%.4f lss:%.4f"  (100*tacc[1])/tacc[2] tlss[1]/tlss[2])

    patiance = [opts[:patiance]]
    for i=1:opts[:epoch]
        lss = train(weights,rsettings,params,data[1],opts)
        info(@sprintf "epoch:%d trnlss:%.4f:" i lss[1]/lss[2])

        dlss,dacc = evaluate(weights,rsettings,data[2],i2o,i2oarr,opts)
        info(@sprintf "[dev-%d] acc:%.4f lss:%.4f" i (100*dacc[1])/dacc[2] dlss[1]/dlss[2])
        tlss,tacc = evaluate(weights,rsettings,data[3],i2o,i2oarr,opts)
        info(@sprintf "[tst-%d] acc:%.4f lss:%.4f" i (100*tacc[1])/tacc[2] tlss[1]/tlss[2])
        if (100*tacc[1])/tacc[2] > besties[2]
            besties[2] = (100*tacc[1])/tacc[2]
            info("best test acc:",besties[2])
            JLD.save(string(opts[:bestmodel],".besttestacc.jld"),
                     "weights",weights,
                     "rsettings",rsettings,
                     "params",params,
                     "lr",opts[:lr],
                     "bestacc",besties[1],
                     "bestacc2",besties[2])
        end
        if (100*dacc[1])/dacc[2] > besties[1]
            patiance[1] = opts[:patiance]
            besties[1] = (100 *dacc[1])/dacc[2]
            info("best dev acc:",besties[1]," test acc:",(100*tacc[1])/tacc[2])
            JLD.save(opts[:bestmodel],
                     "weights",weights,
                     "rsettings",rsettings,
                     "params",params,
                     "lr",opts[:lr],
                     "bestacc",besties[1],
                     "bestacc2",besties[2])
        else
            patiance[1] =  patiance[1] - 1
            if patiance[1] < 0
                info(@sprintf "Patiance goes below zero, training finalized, best dev acc:%.3f" besties[1])
                break
            end
            if patiance[1] == div(opts[:patiance],2)
                opts[:lr] = max(0.1,opts[:lr]*opts[:decayrate])
                params = map(wi->eval(parse("Sgd(;lr=$(opts[:lr]),gclip=$(opts[:gc]))")),weights)
                info(@sprintf "learning rate has been set to: %.4f" opts[:lr])
            end
            #=
            patiance[1] =  patiance[1] - 1
            if patiance[1] < 0
                opts[:lr] = opts[:lr]*opts[:decayrate]
                params = map(wi->eval(parse(opts[:optim])),weights)
                for i=1:length(params)
                    params[i].lr = opts[:lr]
                end
                info(@sprintf "learning rate has been set to: %.4f" opts[:lr])
                patiance[1] = opts[:patiance]
            end
            =#
        end
    end
end

function train(weights,rsettings,params,data,opts)
    lssval = zeros(Float32,2)
    T = length(data)
    datax = shuffle(data)
    ho = KnetArray(zeros(Float32,opts[:hs][4],1,1))
    co = KnetArray(zeros(Float32,opts[:hs][4],1,1))
    for i=1:T
        ho[:] = 0.0;co[:] = 0.0
        (size(datax[i][4],1) > opts[:wordlimit]) && continue
        grads = lossgradient(weights,rsettings,datax[i],ho,co;pdrop=opts[:pdrop],lss=lssval)
        update!(weights,grads,params)
    end
    return lssval
end

function evaluate(weights,rsettings,data,i2o,i2oarr,opts)
    lssval  = zeros(Float32,2)
    predval = zeros(Float32,2)
    T = length(data)
    ho = KnetArray(zeros(Float32,opts[:hs][4],1))
    co = KnetArray(zeros(Float32,opts[:hs][4],1))
    corrnum = 0.0
    count = 0.0
    total = 0.0
    for i=1:T
        ho[:] = 0.0;co[:] = 0.0
        preds = predict(weights,rsettings,data[i],i2o,ho,co;lss=lssval)
        # accuracy calculation
        output   = data[i][4]
        outmask  = data[i][5]
        outmask  = outmask .* (1 - map(x->all(isdigit,x),i2oarr[output]))
        output2  = output .* outmask
        preds2   = vec(all((preds.*outmask).==output2,2))
        valids   = 1 - vec(all(output2.==0,2))
        corrects = preds2 .* valids
        predval[2] += sum(valids)
        predval[1] += sum(corrects)
    end
    return lssval,predval
end

function loss(weights,rsettings,data,ho,co;pdrop=0.0,lss=nothing,preds=nothing,prevt=5)
    input,bs,indices,labels,outmasks,tagrange,decoder_start = data
    total = 0.0; count = 0.0
    omask = convert(KnetArray{Float32},outmasks)
    # Word Encoder
    rnninput = weights[1][:,input]
    rnninput = dropout(rnninput,pdrop;training=true)
    y,he,ce,_ = rnnforw(rsettings[1],weights[3],rnninput;batchSizes=bs,hy=true,cy=true)
    # hidden states in the correct order
    hd1 = he[:,:,end][:,indices]
    cd1 = ce[:,:,end][:,indices]
    # bilstm
    bi_input = reshape(hd1,size(hd1,1),1,size(hd1,2))
    bi_input = dropout(bi_input,pdrop;training=true)
    bi_out,_,_,_ = rnnforw(rsettings[4],weights[6],bi_input)
    bi_out = reshape(bi_out,size(bi_out,1),size(bi_out,3))
    decoder_h1  = relu.((weights[7] * bi_out) .+ weights[8])
    decoder_c1  = KnetArray(zeros(Float32,size(decoder_h1)))
    wordnumber = size(labels,1)
    outlen     = size(weights[10],1)
    # output encoder
    all_ho = Array{Any}(wordnumber)
    all_co = Array{Any}(wordnumber)
    for i=1:wordnumber
        ho = fill!(similar(ho),0.0)
        co = fill!(similar(co),0.0)
        startix = max(i-prevt,1)
        for j = startix:i-1
            outemb0    = weights[2][:,labels[j,tagrange[j]]]
            outemb0    = dropout(outemb0,pdrop;training=true)
            outemb     = reshape(outemb0,size(outemb0,1),1,size(outemb0,2))
            yo,ho,co,_ = rnnforw(rsettings[5],weights[9],outemb,ho,co;hy=true,cy=true)
        end
        all_ho[i]  = ho[:,:,end]
        all_co[i]  = co[:,:,end]
    end
    output_encodings_h = hcat(all_ho...)
    output_encodings_c = hcat(all_co...)
    # Decoder
    decoder_ts = size(labels,2)
    decoder_input = decoder_start
    hd1 = hd1 .+ output_encodings_h
    cd1 = cd1 .+ output_encodings_c
    for i=1:decoder_ts
        rnn2input = weights[2][:,decoder_input]
        rnn2input = dropout(rnn2input,pdrop;training=true)
        yd1,decoder_h1,decoder_c1,_ = rnnforw(rsettings[2],weights[4],rnn2input,decoder_h1,decoder_c1)
        #x2 = dropout(decoder_h1,pdrop)
        x2 = dropout(yd1,pdrop;training=true)
        yd2,hd1,cd1,_ = rnnforw(rsettings[3],weights[5],x2,hd1,cd1)
        #yd2 = dropout(yd2,pdrop) ###
        #ypred = weights[10] * yd2 .+ weights[11] #
        hd1 = dropout(hd1,pdrop;training=true)
        hd1 = reshape(hd1,size(hd1,1),size(hd1,2))
        ypred = weights[10] * hd1 .+ weights[11] #
        # loss calculation
        ynorm = logp(ypred,1)
        index = (0:wordnumber-1)*outlen  + labels[:,i]
        loss_current = ynorm[index] .* omask[:,i]
        total += sum(loss_current)
        count += sum(omask[:,i])
        decoder_input = labels[:,i]
    end
    if lss != nothing
        lss[1] += AutoGrad.getval(-total)
        lss[2] += AutoGrad.getval(count)
    end
    -total/count
end

lossgradient = grad(loss)

function predict(weights,rsettings,data,rov,ho,co;pdrop=0.0,lss=nothing,preds=nothing,prevt=5)
    input,bs,indices,labels,outmasks,tagrange,decoder_start = data
    total = 0.0; count = 0.0
    omask = convert(KnetArray{Float32},outmasks)
    # Word Encoder
    rnninput = weights[1][:,input]
    y,he,ce,_ = rnnforw(rsettings[1],weights[3],rnninput;batchSizes=bs,hy=true,cy=true)
    # hidden states in the correct order
    hd1 = he[:,:,end][:,indices]
    cd1 = ce[:,:,end][:,indices]
    # bilstm
    bi_input = reshape(hd1,size(hd1,1),1,size(hd1,2))
    bi_out,_,_,_ = rnnforw(rsettings[4],weights[6],bi_input)
    bi_out = reshape(bi_out,size(bi_out,1),size(bi_out,3))
    decoder_h1  = relu.((weights[7] * bi_out) .+ weights[8])
    decoder_c1  = KnetArray(zeros(Float32,size(decoder_h1)))
    wordnumber = size(labels,1)
    outlen     = size(weights[10],1)
    # Decoder
    decoder_ts = size(labels,2)
    decoder_input = decoder_start
    preds = Array{Int}(wordnumber,decoder_ts)
    prevPreds = Any[]
    for i=1:wordnumber
        hd1_single = hd1[:,i:i] .+ ho
        cd1_single = cd1[:,i:i] .+ co
        dec_h1_s = decoder_h1[:,i:i]
        dec_c1_s = decoder_c1[:,i:i]
        decoder_input_single = decoder_start[1]
        for j=1:decoder_ts
            rnn2input = weights[2][:,decoder_input_single:decoder_input_single]
            y1,dec_h1_s,dec_c1_s,_ = rnnforw(rsettings[2],weights[4],rnn2input,dec_h1_s,dec_c1_s)
            y2,hd1_single,cd1_single,_ = rnnforw(rsettings[3],weights[5],y1,hd1_single,cd1_single)
            ypred = weights[10] * y2 .+ weights[11]
            ynorm = logp(ypred,1)
            index = labels[i,j]
            total += sum(ynorm[index] .* omask[i,j])
            count += sum(omask[i,j])
            decoder_input_single = indmax(convert(Array{Float32},ynorm))
            preds[i,j] = decoder_input_single
        end
        # eleminate non morpheme tokens
        # MASK,UNK,EOW,chars
        eows = find(x->rov[x]=="EOW",preds[i,:])
        eid  = (length(eows)>0)? eows[1]-1 : length(preds[i,:])
        o1 = filter(x->!(rov[x] in ["UNK","MASK","EOW","BOW"]),preds[i,:][1:eid])
        o2 = filter(x->length(rov[x])>1,o1)
        # output encoder
        if length(prevPreds) > prevt - 1
            prevPreds = prevPreds[2:end]
        end
        push!(prevPreds,o2)
        ho = fill!(similar(ho),0.0)
        co = fill!(similar(co),0.0)
        for j = 1:length(prevPreds)
            length(prevPreds[j]) == 0 && continue
            outemb = weights[2][:,prevPreds[j]]
            outemb2 = reshape(outemb,size(outemb,1),1,size(outemb,2))
            _,ho,co,_ = rnnforw(rsettings[5],weights[9],outemb2,ho,co)
            ho = reshape(ho,size(ho,1),1)
            co = reshape(co,size(co,1),1)
        end
    end
    if lss != nothing
        lss[1] += AutoGrad.getval(-total)
        lss[2] += AutoGrad.getval(count)
    end
    return preds
end

function initweights(o,c2i,o2i,init)
    w = Any[]
    rsettings = Any[]
    # embedding
    push!(w,init(o[:es][1],length(c2i))) #1
    push!(w,init(o[:es][2],length(o2i))) #2
    # encoder
    r1,wenc = rnninit(o[:es][1],o[:hs][1];rnnType=:lstm,seed=o[:seed]) #3
    push!(rsettings,r1)
    push!(w,wenc)
    # decoder
    r21,wdec1 = rnninit(o[:es][2],o[:hs][3];rnnType=:lstm,seed=o[:seed]) #4
    push!(rsettings,r21)
    push!(w,wdec1)
    r22,wdec2 = rnninit(o[:hs][3],o[:hs][3];rnnType=:lstm,seed=o[:seed]) #5
    push!(rsettings,r22)
    push!(w,wdec2)
    # bilstm
    rbi,wbi = rnninit(o[:hs][1],o[:hs][2];rnnType=:lstm,bidirectional=true,seed=o[:seed]) #6
    push!(rsettings,rbi)
    push!(w,wbi)
    # bilstm reducer
    push!(w,init(o[:hs][3],2o[:hs][2])) #7
    push!(w,zeros(o[:hs][3],1))         #8
    # output encoder
    roe,woe = rnninit(o[:es][2],o[:hs][4];rnnType=:lstm,seed=o[:seed]) #9
    push!(rsettings,roe)
    push!(w,wbi)
    # output layer
    push!(w,init(length(o2i),o[:hs][3])) #10
    push!(w,zeros(length(o2i),1))        #11

    # convert to atype
    w0 = map(wi->convert(o[:atype], wi), w)
    weights = convert(Array{Any}, w0)
    return rsettings,weights
end

function minibatch(edf,opts,o2i;st=true)
    all_batches = Any[]
    for df in edf
        batches = _minibatch(df,opts,st,o2i)
        push!(all_batches,batches)
    end
    return all_batches
end

function _minibatch(df,opts,st,o2i)
    batches = Any[]
    sorted_df = (st == true) ? sort(df;lt=(x,y)->length(x)<length(y)) : df
    for i=1:length(sorted_df)
        # words
        words = map(x->x[1],sorted_df[i])
        sort_indices = sortperm(words;lt=(x,y)->length(x)<length(y),rev=true)
        reverse_indices = sortperm(sort_indices)
        sorted_words = words[sort_indices]
        batch_words = Int[]
        batchsizes = Int[]
        for k = 1:length(sorted_words[1])
            bs = 0
            for t=1:length(sorted_words)
                if k<=length(sorted_words[t])
                    push!(batch_words,sorted_words[t][k])
                    bs += 1
                end
            end
            push!(batchsizes,bs)
        end
        # output
        labels = map(x->vcat(x[2],x[3]),sorted_df[i])
        #map(x->unshift!(x,o2i[BOW]),labels)
        map(x->push!(x,o2i[EOW]),labels)
        decoder_start = [o2i[BOW] for k=1:length(labels)]
        tagrange = map(x->(length(x[2])+1):(length(x[2])+1),sorted_df[i])
        maxoutlen = maximum(map(x->length(x),labels))
        outmasks  = ones(Float32,maxoutlen,length(labels))
        for k=1:length(labels)
            maxoutlen == length(labels[k]) && continue
            si = length(labels[k])+1
            outmasks[si:end,k] = 0.0
            for j=si:maxoutlen
                push!(labels[k],o2i[MASK])
            end
        end
        labels_2d = transpose(hcat(labels...)) # wordnumber x timestep
        outmasks_2d  = transpose(outmasks)     # wordnumber x timestep
        push!(batches,(batch_words,batchsizes,reverse_indices,labels_2d,outmasks_2d,tagrange,decoder_start))
    end
    return batches
end

function generate(weights,rsettings,data,words,i2o,i2oarr,i2c,opts;conlluformat=false)
    lssval = zeros(Float32,2)
    T = length(data)
    if conlluformat
        fname = string(opts[:bestmodel],".generation")
    else
        fname = string(opts[:bestmodel],".generation_truefalse")
    end
    fout = open(fname,"w")
    ho = KnetArray(zeros(Float32,opts[:hs][4],1,1))
    co = KnetArray(zeros(Float32,opts[:hs][4],1,1))
    for i=1:T
        ho[:] = 0.0;co[:] = 0.0
        preds = predict(weights,rsettings,data[i],i2o,ho,co;lss=lssval)
        if conlluformat
            write2file(words[i],preds,i2o,opts,i2c,fout)
        else
            write2file(words[i],data[i],preds,i2o,i2oarr,opts,i2c,fout)
        end
        if i != T
            write(fout,"\n")
        end
    end
    close(fout)
end


function write2file(words,preds,i2o,opts,i2c,fout)
    text = string("# text = ",join(words," "))
    write(fout,text,"\n")
    idictlen = length(i2c)
    for i=1:length(words)
        prediction = preds[i,:]
        eows = find(x->i2o[x]=="EOW",prediction)
        eid  = (length(eows)>0)? eows[1]-1 : length(prediction)
        o1 = filter(x->!(i2o[x] in ["UNK","MASK","EOW","BOW"]),prediction[1:eid])
        lemma     = filter(x->x<=idictlen,o1)
        lemma_x   = (length(lemma) == 0) ? "X": join(map(x->i2c[x],lemma),"")
        notlemma  = filter(x->x>idictlen,o1)
        postag    = (length(notlemma) == 0)? "X" : i2o[notlemma[1]]
        morpfeats = (length(notlemma) < 2) ? "_" : join(map(x->i2o[x],notlemma[2:end]),"|")

        write(fout,string(i),"\t",words[i],"\t",lemma_x,"\t",postag,"\t","_","\t",morpfeats,"\t_\t_\t_\t_\n")
    end
end
#=
function write2file(words,data,preds,i2o,i2oarr,opts,i2c,fout)
    output   = data[4]
    outmask  = data[5]
    outmask  = outmask .* (1 - map(x->all(isdigit,x),i2oarr[output]))
    output2  = output .* outmask
    preds2   = vec(all((preds.*outmask).==output2,2))
    valids   = 1 - vec(all(output2.==0,2))
    corrects = preds2 .* valids
    corrects2 = (1 .== corrects)
    text = string("# text = ",join(words," "))
    write(fout,text,"\n")
    idictlen = length(i2c)
    for i=1:length(words)
        # gold label
        go = output[i,:]
        goldoutput = filter(x->!(i2o[x] in ["MASK","EOW","BOW"]) ,go)
        goldlemma  = map(y->i2o[y],filter(x->x<=idictlen,goldoutput))
        goldposandfeat = map(y->i2o[y],filter(x->x>idictlen,goldoutput))
        goldoutput = string(join(goldlemma,""),"\t",join(goldposandfeat,"|"))
        # prediction
        prediction = preds[i,:]
        eows = find(x->i2o[x]=="EOW",prediction)
        eid  = (length(eows)>0)? eows[1]-1 : length(prediction)
        o1 = filter(x->!(i2o[x] in ["UNK","MASK","EOW","BOW"]),prediction[1:eid])
        lemma     = filter(x->x<=idictlen,o1)
        lemma_x   = (length(lemma) == 0) ? "X": join(map(x->i2c[x],lemma),"")
        notlemma  = filter(x->x>idictlen,o1)
        postag    = (length(notlemma) == 0)? "X" : i2o[notlemma[1]]
        morpfeats = (length(notlemma) < 2) ? "_" : join(map(x->i2o[x],notlemma[2:end]),"|")

        write(fout,string(i),"\t",words[i],"\t",lemma_x,"\t",postag,"\t",morpfeats,"\t",goldoutput,"\t",string(corrects2[i]),"\n")
    end
end
=#
function write2file(words,data,preds,i2o,i2oarr,opts,i2c,fout)
    output   = data[4]
    outmask  = data[5]
    outmask  = outmask .* (1 - map(x->all(isdigit,x),i2oarr[output]))
    output2  = output .* outmask
    preds2   = vec(all((preds.*outmask).==output2,2))
    valids   = 1 - vec(all(output2.==0,2))
    corrects = preds2 .* valids
    corrects2 = (1 .== corrects)
    text = string("# text = ",join(words," "))
    write(fout,text,"\n")
    idictlen = length(i2c)
    for i=1:length(words)
        # gold label
        go = output[i,:]
        goldoutput = filter(x->!(i2o[x] in ["MASK","EOW","BOW"]) ,go)
        goldlemma  = map(y->i2o[y],filter(x->x<=idictlen,goldoutput))
        go2 = filter(x->x>idictlen,goldoutput)
        gold_postag = (length(go2) == 0)? "X" : i2o[go2[1]]
        gold_morpfeats = (length(go2) < 2) ? "_" : join(map(x->i2o[x],go2[2:end]),"|")
        goldoutput = string(join(goldlemma,""),"\t",gold_postag,"\t",gold_morpfeats)
        # prediction
        prediction = preds[i,:]
        eows = find(x->i2o[x]=="EOW",prediction)
        eid  = (length(eows)>0)? eows[1]-1 : length(prediction)
        o1 = filter(x->!(i2o[x] in ["UNK","MASK","EOW","BOW"]),prediction[1:eid])
        lemma     = filter(x->x<=idictlen,o1)
        lemma_x   = (length(lemma) == 0) ? "X": join(map(x->i2c[x],lemma),"")
        notlemma  = filter(x->x>idictlen,o1)
        postag    = (length(notlemma) == 0)? "X" : i2o[notlemma[1]]
        morpfeats = (length(notlemma) < 2) ? "_" : join(map(x->i2o[x],notlemma[2:end]),"|")

        write(fout,string(i),"\t",words[i],"\t",lemma_x,"\t",postag,"\t",morpfeats,"\t",goldoutput,"\t",string(corrects2[i]),"\n")
    end
end

!isinteractive() && main()
