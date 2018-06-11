#tr
lcase(x::Char) = x=='I' ? 'ı':lowercase(x);
lcase(s::AbstractString) = map(lcase,s)

function convert_data_format(df,tr_corpus)
    if tr_corpus == "trmor2018"
        _S_ = "<s"
        _SS_= "</s"
        SEP = "\t"
        UNK  = "*unknown*"
    else
        _S_  = "<S"
        _SS_ = "</S"
        SEP  = " "
        UNK  = "*UNKNOWN*"
    end
    converted_data = Any[]
    for f in df
        open(f,"r") do fin
            sentence  = Any[]
            sentences = Any[]
            while !eof(fin)
                line = readline(fin)
                if startswith(line,"<DOC")||startswith(line,"</DOC")||
                    startswith(line,"<TITLE")||startswith(line,"</TITLE") || line == ""
                    continue
                end
                if startswith(line,_S_)
                    sentence = Any[]
                elseif startswith(line,_SS_)
                    if length(sentence) == 0
                        continue
                    end
                    if length(filter(x->x[3][1] != UNK,sentence)) != 0
                        push!(sentences,sentence)
                    end
                    sentence = Any[]
                else
                    parts = split(line,SEP)
                    tags =  Any[]
                    isamb =  (length(parts) > 2) ? true : false
                    word  = parts[1]
                    analysis = parts[2]
                    if contains(analysis ,UNK)
                        lemma = "u"
                        push!(tags,UNK)
                    elseif contains(analysis,"++")
                        println(line," ",word," ",analysis)
                        lemma = "+"
                        push!(tags,"+Punc")
                    else
                        tmp = split(analysis,"+")
                        lemma = tmp[1]
                        for i=2:length(tmp)
                            if contains(tmp[i],"^DB")
                                push!(tags,split(tmp[i],"^DB")[1])
                                push!(tags,"^DB")
                            else
                                push!(tags,tmp[i])
                            end
                        end
                    end
                    push!(sentence,(lcase(word),lcase(lemma),tags,isamb))
                end
            end
            push!(converted_data,sentences)
        end
    end
    return converted_data
end

function create_vocabulary(cdf)
    c2i = Dict()
    o2i = Dict()
    global UNKTAG = "UNK"
    global MASK   = "MASK"
    global EOW    = "EOW"
    global BOW    = "BOW"
    # for tag file use only training data
    # for characters use all data
    for (ix,cf) in enumerate(cdf)
        for sentence in cf
            # word,label,tags,type(amb or unamb)
            for (w,l,t,ty) in sentence
                map(x->get!(c2i,x,length(c2i)+1),collect(w))
                map(x->get!(o2i,x,length(o2i)+1),collect(w))
            end
        end
    end
    for (ix,cf) in enumerate(cdf)
        for sentence in cf
            for (w,l,t,ty) in sentence
                map(x->get!(o2i,x,length(o2i)+1),t)
            end
        end
    end
    get!(o2i,UNKTAG,length(o2i)+1)
    get!(o2i,MASK,length(o2i)+1)
    get!(o2i,BOW,length(o2i)+1)
    get!(o2i,EOW,length(o2i)+1)
    return c2i,o2i
end

function encode_data(cdf,c2i,o2i)
    edf = Any[]
    for cf in cdf
        esentences = Any[]
        for sentence in cf
            esentence = Any[]
            # word,label,tags,type(amb or unamb)
            for (w,l,t,ty) in sentence
                eword  = map(x->c2i[x],collect(w))
                elemma = map(x->get(o2i,x,o2i[UNKTAG]),collect(l))
                etag   = map(x->o2i[x],t)
                push!(esentence,(eword,elemma,etag,ty))
            end
            push!(esentences,esentence)
        end
        push!(edf,esentences)
    end
    return edf
end

function prepare_data(opts)
    df  = opts[:datafiles]
    cdf = convert_data_format(df,opts[:treebank])
    c2i,o2i = create_vocabulary(cdf)
    edf = encode_data(cdf,c2i,o2i)
    return edf,c2i,o2i
end
