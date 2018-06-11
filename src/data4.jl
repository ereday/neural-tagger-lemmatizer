lcase(x::Char) = x=='I' ? 'ı':lowercase(x)
lcase(s::AbstractString) = map(lcase,s)
function convert_data_format(df)
    converted_data = Any[]
    for f in df
        open(f,"r") do fin
            sentence = Any[]
            sentences = Any[]
            while !eof(fin)
                line = readline(fin)
                if line == "" # sentence is over new sentence is about to start
                    push!(sentences,sentence)
                    sentence = Any[]
                elseif (m=match(r"^(\d-*\d*)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)(:.+)?\t",line)) != nothing
                    id  = m.captures[1]
                    contains(id,"-") && continue
                    word   = lcase(m.captures[2])
                    lemma  = lcase(m.captures[3])
                    postag = m.captures[4]
                    morphemes = m.captures[6]
                    ms = split(morphemes,"|")
                    #if postag == "X"
                    #    postag = "*UNKNOWN*"
                    #end
                    if morphemes != "_"
                        label = unshift!(ms,postag)
                    else
                        label = [postag]
                    end
                    push!(sentence,(word,lemma,label))
                else
                    # raw sentence starting with # symbol
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
    composite_tag_dict = Dict()
    global UNKTAG = "UNK"
    global MASK   = "MASK"
    global EOW    = "EOW"
    global BOW    = "BOW"
    # for tag file use only training data
    # for characters use all data
    for (ix,cf) in enumerate(cdf)
        for sentence in cf
            for (w,l,t) in sentence
                map(x->get!(c2i,x,length(c2i)+1),collect(w))
                map(x->get!(o2i,x,length(o2i)+1),collect(w))
            end
        end
    end
    for (ix,cf) in enumerate(cdf)
        for sentence in cf
            for (w,l,t) in sentence
                map(x->get!(o2i,x,length(o2i)+1),t)
                if ix != 3
                    get!(composite_tag_dict,join(t,"|"),length(composite_tag_dict)+1)
                end
            end
        end
    end
    get!(composite_tag_dict,UNKTAG,length(composite_tag_dict)+1)
    get!(o2i,UNKTAG,length(o2i)+1)
    get!(o2i,MASK,length(o2i)+1)
    get!(o2i,BOW,length(o2i)+1)
    get!(o2i,EOW,length(o2i)+1)
    return c2i,o2i,composite_tag_dict
end

function encode_data(cdf,c2i,o2i)
    edf = Any[]
    for cf in cdf
        esentences = Any[]
        for sentence in cf
            esentence = Any[]
            for (w,l,t) in sentence
                eword  = map(x->c2i[x],collect(w))
                elemma = map(x->get(o2i,x,o2i[UNKTAG]),collect(l))
                etag   = map(x->o2i[x],t)
                push!(esentence,(eword,elemma,etag))
            end
            push!(esentences,esentence)
        end
        push!(edf,esentences)
    end
    return edf
end

function prepare_data(opts)
    df  = opts[:datafiles]
    cdf = convert_data_format(df)
    c2i,o2i,composite_tag_dict = create_vocabulary(cdf)
    edf = encode_data(cdf,c2i,o2i)
    return edf,c2i,o2i,composite_tag_dict
end
